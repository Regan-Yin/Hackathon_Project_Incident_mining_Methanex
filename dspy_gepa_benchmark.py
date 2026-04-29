"""DSPy + GEPA optimization for the Methanex Generative AI Safety Analyst.

Pipeline (mirrors the agentic-movie-recommender setup):

  1. Build a stratified eval set from `events_clean.csv` — each case feeds the
     model a partial input (keywords / what-happened snippet / mini-report) and
     keeps the gold risk_level / severity / category_type / root_cause / actions
     / lessons for scoring.
  2. Sweep four hand-authored prompt styles against the *real* analyzer pipeline
     (TF-IDF + Ollama Cloud) and pick the best.
  3. Wrap a `dspy.ChainOfThought` module around the strict-JSON signature and
     run `dspy.GEPA` with a high-temperature reflection LM, using the metric's
     structured feedback string so reflection targets concrete failure modes
     (label mismatch, missing action structure, non-grounded root cause, …).
  4. Persist `dspy_gepa_best_config.json` (loaded by `safety_analyst.py` at
     import time) and `dspy_gepa_eval_cases.json` for reproducibility.

Run:

    export OLLAMA_API_KEY=...
    python dspy_gepa_benchmark.py --num-cases 24 --auto light          # quickest
    python dspy_gepa_benchmark.py --num-cases 36 --auto medium         # better
    python dspy_gepa_benchmark.py --prepare-only                       # just dump cases
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from safety_analyst import (
    ACTIONS_DF,
    DEFAULT_INSTRUCTION,
    EVENTS_DF,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    TUNED_CONFIG_PATH,
    VALID_CATEGORY,
    VALID_RISK,
    VALID_SEVERITY,
    _extract_json_payload,
    _normalize_payload,
    _ollama_chat,
    build_prompt,
    fallback_response,
    format_history_block,
    retrieve_similar_events,
)

load_dotenv()

dspy = importlib.import_module("dspy")

EVAL_DUMP_PATH = Path(__file__).resolve().parent / "dspy_gepa_eval_cases.json"


# ---------------------------------------------------------------------------
# Prompt style sweep candidates (the GEPA "starting points")
# ---------------------------------------------------------------------------

PROMPT_STYLES: dict[str, str] = {
    "balanced": (
        "Be precise, structured, and grounded in the historical context. Match every "
        "label exactly to the canonical vocabularies."
    ),
    "strict_format": (
        "Strict format compliance is paramount. Always emit valid JSON with all six "
        "required keys. Each suggested action MUST contain Action, Owner, Timing, "
        "and Verification — no exceptions."
    ),
    "evidence_grounded": (
        "Ground every conclusion in the provided historical context. Reference case "
        "numbers when helpful. Prefer modal labels across the most-similar cases."
    ),
    "operational": (
        "Lead with operational priority: identify the root cause first, then enumerate "
        "actions ordered by urgency (Immediate → <30 days → 30-90 days → >90 days). "
        "Best practices last, drawn from the lessons column of the retrieved cases."
    ),
}


# ---------------------------------------------------------------------------
# Eval-case construction
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    event_id: str
    input_text: str
    input_style: str
    gold_risk: str
    gold_severity: str
    gold_category: str
    gold_root_cause: str
    gold_lessons: str


def _coerce(label: str, options: list[str]) -> str:
    """Return the canonical option that matches `label` (case-insensitive contains)."""
    if not isinstance(label, str):
        return ""
    s = label.strip()
    if not s:
        return ""
    sl = s.lower()
    for opt in options:
        if opt.lower() == sl:
            return opt
    for opt in options:
        if opt.lower() in sl or sl in opt.lower():
            return opt
    return ""


def _make_input(row: pd.Series, style: str) -> str:
    title = str(row.get("title", "")).strip()
    what = str(row.get("what_happened", "")).strip()
    causes = str(row.get("root_causes", "")).strip()
    if style == "keywords":
        keywords = [w for w in re.split(r"[\s,;]+", title.lower()) if len(w) >= 4]
        return " ".join(keywords[:8]) or title
    if style == "title":
        return title
    if style == "snippet":
        return what[:400]
    if style == "mini_report":
        return f"{title}. {what[:280]}"
    return f"{title}\n{what[:360]}\nRoot cause hint: {causes[:200]}"


def _build_eval_cases(num_cases: int, seed: int) -> list[EvalCase]:
    rng = random.Random(seed)
    df = EVENTS_DF.dropna(subset=["risk_level", "severity", "category_type", "what_happened"]).copy()

    cats = sorted(df["category_type"].unique().tolist())
    per_cat = max(1, num_cases // max(1, len(cats)))
    style_cycle = ["snippet", "mini_report", "title", "keywords", "full"]

    selected: list[EvalCase] = []
    seen_ids: set[str] = set()
    for cat in cats:
        sub = df[df["category_type"] == cat]
        if sub.empty:
            continue
        idx = list(sub.index)
        rng.shuffle(idx)
        for i in idx[:per_cat]:
            r = df.loc[i]
            ev_id = str(r["event_id"])
            if ev_id in seen_ids:
                continue
            seen_ids.add(ev_id)
            style = style_cycle[len(selected) % len(style_cycle)]
            text = _make_input(r, style)
            selected.append(
                EvalCase(
                    event_id=ev_id,
                    input_text=text,
                    input_style=style,
                    gold_risk=_coerce(str(r["risk_level"]), VALID_RISK) or "Medium",
                    gold_severity=_coerce(str(r["severity"]), VALID_SEVERITY) or "Potentially Significant",
                    gold_category=_coerce(str(r["category_type"]), VALID_CATEGORY) or "Other",
                    gold_root_cause=str(r.get("root_causes", "")).strip(),
                    gold_lessons=str(r.get("lessons", "")).strip(),
                )
            )
            if len(selected) >= num_cases:
                break
        if len(selected) >= num_cases:
            break

    rng.shuffle(selected)
    return selected[:num_cases]


# ---------------------------------------------------------------------------
# Metric (used both as the GEPA objective and for native-style scoring)
# ---------------------------------------------------------------------------

def _label_score(pred: str, gold: str, options: list[str]) -> float:
    if not pred or not gold:
        return 0.0
    p = _coerce(pred, options)
    g = _coerce(gold, options)
    if not p or not g:
        return 0.0
    if p == g:
        return 1.0
    # Severity tiers are ordinal — give partial credit for adjacent buckets.
    try:
        if options is VALID_SEVERITY:
            tier = {"Minor": 0, "Near Miss": 0, "Potentially Significant": 1, "Serious": 2, "Major": 3}
            return max(0.0, 1.0 - 0.4 * abs(tier[p] - tier[g]))
        if options is VALID_RISK:
            tier = {"Low": 0, "Medium": 1, "High": 2}
            return max(0.0, 1.0 - 0.5 * abs(tier[p] - tier[g]))
    except KeyError:
        pass
    return 0.3


_ACTION_NUM_RE = re.compile(r"(?m)^\s*\d+\s*[\.\)]")


def _action_score(text: str) -> float:
    if not text:
        return 0.0
    s = str(text)
    n_actions = min(len(_ACTION_NUM_RE.findall(s)), 5)
    if n_actions == 0:
        n_actions = min(s.count("\n") + 1, 3)
    base = min(1.0, n_actions / 3.0)
    sl = s.lower()
    structure_bits = sum(int(k in sl) for k in ("action:", "owner:", "timing:", "verif"))
    structure = structure_bits / 4.0
    return 0.45 * base + 0.55 * structure


def _length_score(text: str, lo: int, hi: int) -> float:
    if not text:
        return 0.0
    n = len(str(text).strip())
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return n / max(1.0, lo)
    return max(0.0, 1.0 - (n - hi) / max(1.0, hi))


def _grounding_score(text: str, gold_root: str, gold_lessons: str) -> float:
    if not text:
        return 0.0
    text_l = str(text).lower()
    pool = (gold_root + " " + gold_lessons).lower()
    pool_tokens = {t for t in re.findall(r"[a-z]+", pool) if len(t) >= 5}
    if not pool_tokens:
        return 0.5
    text_tokens = {t for t in re.findall(r"[a-z]+", text_l) if len(t) >= 5}
    overlap = len(pool_tokens & text_tokens) / max(1, len(pool_tokens))
    return min(1.0, 1.6 * overlap + 0.2)


def metric_from_payload(case: EvalCase, payload: dict[str, Any]) -> tuple[float, str]:
    risk = _label_score(payload.get("risk_level", ""), case.gold_risk, VALID_RISK)
    sev = _label_score(payload.get("severity", ""), case.gold_severity, VALID_SEVERITY)
    cat = _label_score(payload.get("category_type", ""), case.gold_category, VALID_CATEGORY)
    rc_len = _length_score(payload.get("root_cause", ""), 80, 600)
    rc_ground = _grounding_score(payload.get("root_cause", ""), case.gold_root_cause, case.gold_lessons)
    rc = 0.5 * rc_len + 0.5 * rc_ground
    act = _action_score(payload.get("suggested_actions", ""))
    bp = 0.6 * _length_score(payload.get("best_practices", ""), 50, 480) + 0.4 * _grounding_score(
        payload.get("best_practices", ""), "", case.gold_lessons
    )

    final = 0.22 * risk + 0.20 * sev + 0.16 * cat + 0.16 * rc + 0.18 * act + 0.08 * bp
    final = max(0.0, min(1.0, final))

    feedback = (
        f"risk={risk:.2f} (pred='{payload.get('risk_level','')}', gold='{case.gold_risk}'), "
        f"severity={sev:.2f} (pred='{payload.get('severity','')}', gold='{case.gold_severity}'), "
        f"category={cat:.2f} (pred='{payload.get('category_type','')}', gold='{case.gold_category}'), "
        f"root_cause_grounding={rc_ground:.2f}, root_cause_length={rc_len:.2f}, "
        f"action_structure={act:.2f}, best_practices={bp:.2f}. "
        "Match risk/severity/category EXACTLY to the canonical vocabulary. Each suggested "
        "action MUST contain Action, Owner, Timing, AND Verification on one line. Ground "
        "the root cause in the historical cases (use procedure / isolation / training / "
        "fatigue / congestion language that appears in the retrieved evidence)."
    )
    return final, feedback


# ---------------------------------------------------------------------------
# DSPy module
# ---------------------------------------------------------------------------

class AnalyzeSafetyEvent(dspy.Signature):
    """Methanex EPSSC safety analyst: predict labels and write a structured report
    grounded in retrieved historical cases. Output strict JSON-compatible fields."""

    incident_input: str = dspy.InputField(
        desc="User-provided report. May be keywords, a short 'what happened' snippet, or a full report."
    )
    historical_context: str = dspy.InputField(
        desc="Top similar historical Methanex cases (TF-IDF retrieved), with risk/severity/lessons/actions."
    )
    style_guide: str = dspy.InputField(desc="Stylistic guidance for the response.")

    risk_level: str = dspy.OutputField(desc='Exactly one of: "Low" | "Medium" | "High".')
    severity: str = dspy.OutputField(
        desc='Exactly one of: "Minor" | "Potentially Significant" | "Serious" | "Major" | "Near Miss".'
    )
    category_type: str = dspy.OutputField(desc='Exactly one of: "Incident" | "Near Miss" | "Other".')
    root_cause: str = dspy.OutputField(
        desc="2-4 sentences identifying the most likely root cause(s); ground in historical context."
    )
    suggested_actions: str = dspy.OutputField(
        desc=(
            "3-5 numbered actions. Each EXACTLY: 'N. Action: <text>. Owner: <role>. "
            "Timing: <Immediate | <30 days | 30-90 days | >90 days>. Verification: <text>.'"
        )
    )
    best_practices: str = dspy.OutputField(
        desc="2-3 short bullet lessons drawn from the historical cases (lessons / what_went_well)."
    )


class SafetyProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeSafetyEvent)

    def forward(self, incident_input: str, historical_context: str, style_guide: str):
        return self.analyze(
            incident_input=incident_input,
            historical_context=historical_context,
            style_guide=style_guide,
        )


def _build_dspy_example(case: EvalCase, style: str) -> Any:
    top = retrieve_similar_events(case.input_text, k=8)
    block = format_history_block(top)
    return dspy.Example(
        incident_input=case.input_text,
        historical_context=block,
        style_guide=PROMPT_STYLES.get(style, PROMPT_STYLES["balanced"]),
        gold_risk=case.gold_risk,
        gold_severity=case.gold_severity,
        gold_category=case.gold_category,
        gold_root_cause=case.gold_root_cause,
        gold_lessons=case.gold_lessons,
        event_id=case.event_id,
    ).with_inputs("incident_input", "historical_context", "style_guide")


def _example_to_case(example: Any) -> EvalCase:
    return EvalCase(
        event_id=str(getattr(example, "event_id", "")),
        input_text=str(getattr(example, "incident_input", "")),
        input_style="dspy",
        gold_risk=str(getattr(example, "gold_risk", "")),
        gold_severity=str(getattr(example, "gold_severity", "")),
        gold_category=str(getattr(example, "gold_category", "")),
        gold_root_cause=str(getattr(example, "gold_root_cause", "")),
        gold_lessons=str(getattr(example, "gold_lessons", "")),
    )


def dspy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    case = _example_to_case(gold)
    payload = {
        "risk_level": getattr(pred, "risk_level", ""),
        "severity": getattr(pred, "severity", ""),
        "category_type": getattr(pred, "category_type", ""),
        "root_cause": getattr(pred, "root_cause", ""),
        "suggested_actions": getattr(pred, "suggested_actions", ""),
        "best_practices": getattr(pred, "best_practices", ""),
    }
    score, feedback = metric_from_payload(case, payload)
    return dspy.Prediction(score=float(score), feedback=feedback)


# ---------------------------------------------------------------------------
# Native-style scoring (uses the live `safety_analyst` pipeline w/ Ollama)
# ---------------------------------------------------------------------------

def _evaluate_native_style(style: str, cases: list[EvalCase]) -> float:
    style_block = PROMPT_STYLES[style]
    system_prompt = f"{DEFAULT_INSTRUCTION}\n\nStyle guide: {style_block}"
    scores: list[float] = []
    for case in cases:
        top = retrieve_similar_events(case.input_text, k=8)
        history_block = format_history_block(top)
        prompt = build_prompt(case.input_text, history_block)
        try:
            raw = _ollama_chat(prompt, system_prompt)
            payload = _normalize_payload(_extract_json_payload(raw))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{style}] LLM failed for {case.event_id}: {exc} — using fallback")
            payload = fallback_response(top)
        score, _ = metric_from_payload(case, payload)
        scores.append(score)
    return sum(scores) / max(1, len(scores))


def _evaluate_program(program: SafetyProgram, dataset: list[Any]) -> float:
    scores: list[float] = []
    for ex in dataset:
        try:
            pred = program(
                incident_input=ex.incident_input,
                historical_context=ex.historical_context,
                style_guide=ex.style_guide,
            )
            r = dspy_metric(ex, pred)
            scores.append(float(r.score))
        except Exception as exc:  # noqa: BLE001
            print(f"  program eval err: {exc}")
            scores.append(0.0)
    return sum(scores) / max(1, len(scores))


# ---------------------------------------------------------------------------
# DSPy LM configuration (Ollama Cloud)
# ---------------------------------------------------------------------------

def _configure_dspy_models() -> Any:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is required")

    model = os.getenv("DSPY_MODEL", f"ollama_chat/{OLLAMA_MODEL}")
    api_base = os.getenv("DSPY_API_BASE", OLLAMA_API_BASE)
    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.2,
        max_tokens=900,
    )
    reflection_model = os.getenv("DSPY_REFLECTION_MODEL", model)
    reflection_lm = dspy.LM(
        model=reflection_model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.9,
        max_tokens=1500,
    )
    dspy.configure(lm=lm)
    return reflection_lm


def _extract_best_instruction(program: Any) -> str:
    try:
        text = str(program.analyze.predict.signature.instructions).strip()
        if text:
            return text
    except Exception:  # noqa: BLE001
        pass
    try:
        details = getattr(program, "detailed_results", None)
        best_candidate = getattr(details, "best_candidate", None) if details is not None else None
        if isinstance(best_candidate, dict):
            for key in ("analyze.predict", "analyze", "predict"):
                if key in best_candidate and str(best_candidate[key]).strip():
                    return str(best_candidate[key]).strip()
            for value in best_candidate.values():
                if str(value).strip():
                    return str(value).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    cases = _build_eval_cases(num_cases=args.num_cases, seed=args.seed)
    if len(cases) < 6:
        raise RuntimeError("Need at least 6 eval cases for a meaningful GEPA run")

    EVAL_DUMP_PATH.write_text(
        json.dumps([asdict(c) for c in cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.prepare_only:
        print(json.dumps(
            {"num_cases": len(cases), "mode": "prepare-only", "dump_path": str(EVAL_DUMP_PATH)},
            indent=2,
        ))
        return

    reflection_lm = _configure_dspy_models()

    style_scores: dict[str, float] = {}
    if args.skip_style_sweep:
        best_style = "balanced"
        print("Skipping style sweep — using 'balanced' style.")
    else:
        sweep_cases = cases[: max(6, min(len(cases), args.style_sweep_cases))]
        for s in PROMPT_STYLES:
            score = _evaluate_native_style(s, sweep_cases)
            style_scores[s] = score
            print(f"native style {s:>17}: {score:.4f}")
        best_style = max(style_scores, key=lambda x: style_scores[x])

    split = max(4, int(0.6 * len(cases)))
    train_cases = cases[:split]
    val_cases = cases[split:] if len(cases) > split else cases[: max(2, len(cases) // 3)]

    trainset = [_build_dspy_example(c, best_style) for c in train_cases]
    valset = [_build_dspy_example(c, best_style) for c in val_cases]

    student = SafetyProgram()
    baseline_score = _evaluate_program(student, valset)
    print(f"DSPy baseline (untrained) val score: {baseline_score:.4f}")

    gepa_kwargs: dict[str, Any] = {
        "metric": dspy_metric,
        "reflection_lm": reflection_lm,
        "track_stats": True,
        "seed": args.seed,
        "num_threads": args.num_threads,
    }
    budget_mode = "auto"
    if args.max_metric_calls > 0:
        gepa_kwargs["max_metric_calls"] = int(args.max_metric_calls)
        budget_mode = "max_metric_calls"
    elif args.max_full_evals > 0:
        gepa_kwargs["max_full_evals"] = int(args.max_full_evals)
        budget_mode = "max_full_evals"
    else:
        gepa_kwargs["auto"] = args.auto

    gepa = dspy.GEPA(**gepa_kwargs)
    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    optimized_score = _evaluate_program(optimized, valset)
    best_instruction = _extract_best_instruction(optimized)

    summary = {
        "best_style": best_style,
        "best_instruction": best_instruction,
        "native_style_scores": style_scores,
        "dspy_baseline_score": baseline_score,
        "dspy_gepa_score": optimized_score,
        "num_cases": len(cases),
        "auto_budget": args.auto,
        "budget_mode": budget_mode,
        "max_metric_calls": int(args.max_metric_calls or 0),
        "max_full_evals": int(args.max_full_evals or 0),
        "seed": args.seed,
        "model": OLLAMA_MODEL,
        "eval_case_dump": str(EVAL_DUMP_PATH.name),
    }

    TUNED_CONFIG_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSPy + GEPA optimization for the Methanex Safety Analyst.")
    parser.add_argument("--num-cases", type=int, default=24, help="Total eval cases to build.")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light", help="GEPA auto budget.")
    parser.add_argument("--num-threads", type=int, default=1, help="Threads for GEPA evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build eval cases; skip GEPA.")
    parser.add_argument("--skip-style-sweep", action="store_true", help="Skip the native-style sweep (faster).")
    parser.add_argument("--style-sweep-cases", type=int, default=8, help="Cases used for the style sweep.")
    parser.add_argument("--max-metric-calls", type=int, default=0, help="Override GEPA budget with max metric calls.")
    parser.add_argument("--max-full-evals", type=int, default=0, help="Override GEPA budget with max full evals.")
    run(parser.parse_args())
