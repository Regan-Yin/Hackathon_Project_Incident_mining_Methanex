"""Generative AI Safety Analyst engine for Methanex EPSSC.

Replaces the previous Vertex AI Vector Search RAG with:
  1. TF-IDF cosine similarity over the local 2019-2024 events corpus.
  2. An Ollama Cloud LLM call with a strict JSON contract.
  3. A DSPy + GEPA-tuned instruction (loaded from `dspy_gepa_best_config.json`).
  4. A deterministic, label-only fallback when the LLM is unreachable so the
     UI always renders a complete, well-formatted report.

Public API (kept identical to the old `rag_engine` so `app.py` did not change):

    analyze_new_event(text: str, events_df: pd.DataFrame | None = None,
                      k: int = 10) -> tuple[str, pd.DataFrame]

Returns `(markdown_response, top_k_events_df)`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — emits to stdout so the running Dash process surfaces every LLM
# decision (success, fallback, model swap, parse error). Without this the
# fallback notice was the only signal the user got, which made it impossible
# to tell whether a real LLM response was generated or not.
# ---------------------------------------------------------------------------

log = logging.getLogger("safety_analyst")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[safety_analyst] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TUNED_CONFIG_PATH = ROOT / "dspy_gepa_best_config.json"

# Free Ollama Cloud models, ordered for SPEED first then QUALITY fallback.
#
# Empirically benchmarked on the actual safety-analyst prompt
# (~24k chars: 1 system msg + 10 retrieved cases):
#   gpt-oss:20b-cloud           ~5s   ✓ clean 6-key JSON, deterministic
#   gemini-3-flash-preview:cloud ~9s  ✓ usually clean, occasional truncation
#   gpt-oss:120b-cloud          ~9s   ✓ clean (big reasoning, content + thinking both populated)
#   qwen3-coder:480b-cloud      ~80s  ✓ very high quality but slow
#
# `deepseek-v4-flash:cloud` and `deepseek-v4-pro:cloud` look attractive but
# require a paid Ollama Cloud subscription (HTTP 403 on the free tier as of
# 2026-04). They can be added by users on a paid plan via OLLAMA_MODEL env var.
#
# We try the primary first, then fall through the ordered fallback list. Each
# entry is consulted only if the previous model raised, returned empty content,
# or produced an unparseable / heavily-incomplete payload. Override via env
# vars OLLAMA_MODEL and OLLAMA_FALLBACK_MODELS (comma-separated).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
OLLAMA_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "gemini-3-flash-preview:cloud,gpt-oss:120b-cloud,qwen3-coder:480b-cloud",
    ).split(",")
    if m.strip()
]
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
# Reasoning models (gpt-oss:120b) routinely take 90-150s for our prompt size
# (1 system msg + 10 retrieved cases). 60s was too aggressive and was the
# primary reason the UI showed "LLM unavailable" even when the API was healthy.
LLM_TIMEOUT_S = int(os.getenv("ANALYST_LLM_TIMEOUT", "180"))
# When a model returns *some* fields but is missing 1-2 sections, we merge
# the gaps from the deterministic corpus fallback rather than throwing the
# whole response away. Only when too many fields are missing do we fall back
# entirely.
MAX_MISSING_FIELDS_BEFORE_FULL_FALLBACK = int(os.getenv("ANALYST_MAX_MISSING", "2"))

# Canonical label vocabularies (match the dataset's casing exactly).
VALID_RISK = ["Low", "Medium", "High"]
VALID_SEVERITY = ["Minor", "Potentially Significant", "Serious", "Major", "Near Miss"]
VALID_CATEGORY = ["Incident", "Near Miss", "Other"]

# Sections every response must contain — used by training metric AND fallback.
RESPONSE_KEYS = (
    "risk_level",
    "severity",
    "category_type",
    "root_cause",
    "suggested_actions",
    "best_practices",
)

DEFAULT_INSTRUCTION = """You are an expert Process Safety Engineer for Methanex EPSSC.
You analyze a user-provided incident or near-miss report by grounding every claim in the
historical Methanex 2019-2024 corpus passed to you as `historical_context`.

Decision policy:
- ALWAYS infer risk_level, severity, and category_type from the closest historical cases
  (use the modal label across the most-similar cases, broken by the highest-similarity match).
- The user input may be terse keywords, a short "what happened" snippet, or a full report.
  Adapt: when the input is sparse, lean harder on the historical context for grounding.
- Root cause must reference concrete factors visible in the historical cases (isolation,
  procedure gaps, training, fatigue, congestion, design, etc.). Do not invent equipment.

Formatting rules (CRITICAL — used by an automated post-processor):
- NEVER use the unicode bullet characters '●', '○', '•', '▪', '■', or any similar marker.
  Even though the historical context contains them, you must NOT echo them back. Use
  plain markdown bullets ("- ") if you list items.
- "suggested_actions" must be 3-5 numbered actions, each on its own line. Each action MUST
  follow EXACTLY this format (and must NOT contain the literal word "Action:"):
      "N. <imperative action sentence>. Owner: <role>. Timing: <Immediate | <30 days | 30-90 days | >90 days>. Verification: <text>."
- "best_practices" must be a list of 2-3 plain bullet strings (no '●'), one per line.
- "root_cause" should be a 2-4 sentence paragraph; use markdown bullets only if you must
  enumerate multiple distinct causes.
- Output ONLY valid JSON with the six required keys. No prose, no markdown headings, no code fences."""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(DATA_DIR / "events_clean.csv")
    actions = pd.read_csv(DATA_DIR / "actions_clean.csv")
    cluster_path = DATA_DIR / "case_cluster_map.csv"
    if cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
        if "cluster_name" in clusters.columns and "cluster_name" not in events.columns:
            events = events.merge(clusters[["event_id", "cluster_name"]], on="event_id", how="left")
    return events, actions


EVENTS_DF, ACTIONS_DF = _load_events()


_TEXT_COLUMNS = (
    "title",
    "what_happened",
    "what_could_have_happened",
    "root_causes",
    "causal_factors",
    "lessons",
    "primary_classification",
    "primary_classification_clean",
    "setting",
    "category",
)


def _row_to_doc(row: pd.Series) -> str:
    parts: list[str] = []
    for col in _TEXT_COLUMNS:
        if col not in row:
            continue
        val = row[col]
        if bool(pd.notna(val)):
            parts.append(str(val))
    return " ".join(parts)


def _build_tfidf(df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    corpus = [_row_to_doc(r) for _, r in df.iterrows()]
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        sublinear_tf=True,
        max_features=20000,
    )
    matrix = vec.fit_transform(corpus)
    return vec, matrix


VECTORIZER, EVENT_MATRIX = _build_tfidf(EVENTS_DF)


def retrieve_similar_events(query: str, k: int = 10) -> pd.DataFrame:
    """Return the top-k most similar events to `query` with a `similarity` column."""
    query = (query or "").strip()
    if not query:
        return EVENTS_DF.head(0).assign(similarity=pd.Series(dtype="float64"))
    qvec = VECTORIZER.transform([query])
    sims = cosine_similarity(qvec, EVENT_MATRIX).ravel()
    k = max(1, min(int(k), len(sims)))
    idx = np.argsort(sims)[::-1][:k]
    out = EVENTS_DF.iloc[idx].copy()
    out["similarity"] = np.round(sims[idx], 4)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _short(text: Any, n: int) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().replace("\n", " ")
    return (s[: n - 1] + "…") if len(s) > n else s


def _actions_for(event_id: str, max_actions: int = 5) -> list[str]:
    rows = ACTIONS_DF[ACTIONS_DF["event_id"] == event_id]
    out = []
    for r in rows.itertuples():
        text = _short(getattr(r, "action_text", ""), 90)
        owner = _short(getattr(r, "owner_role", ""), 40)
        timing = _short(getattr(r, "timing", ""), 24)
        verif = _short(getattr(r, "verification", ""), 60)
        bits = [text]
        if owner:
            bits.append(f"Owner: {owner}")
        if timing:
            bits.append(f"Timing: {timing}")
        if verif:
            bits.append(f"Verification: {verif}")
        out.append(" | ".join(bits))
        if len(out) >= max_actions:
            break
    return out


def format_history_block(top_df: pd.DataFrame) -> str:
    """Compact, LLM-friendly representation of the retrieved cases."""
    if top_df.empty:
        return "No historical cases retrieved."
    chunks: list[str] = []
    for i, row in enumerate(top_df.itertuples(), 1):
        event_id = getattr(row, "event_id", "")
        similarity = float(getattr(row, "similarity", 0.0) or 0.0)
        actions = _actions_for(event_id)
        actions_block = "\n    ".join(f"- {a}" for a in actions) if actions else "- (no recorded actions)"
        chunks.append(
            f"Case {i} [event_id={event_id} | similarity={similarity:.3f}]\n"
            f"  Title: {_short(getattr(row, 'title', ''), 140)}\n"
            f"  Category: {_short(getattr(row, 'category_type', ''), 24)} | "
            f"Risk: {_short(getattr(row, 'risk_level', ''), 12)} | "
            f"Severity: {_short(getattr(row, 'severity', ''), 32)}\n"
            f"  Primary Classification: {_short(getattr(row, 'primary_classification_clean', ''), 80)}\n"
            f"  What Happened: {_short(getattr(row, 'what_happened', ''), 520)}\n"
            f"  Root Causes: {_short(getattr(row, 'root_causes', ''), 380)}\n"
            f"  Lessons: {_short(getattr(row, 'lessons', ''), 380)}\n"
            f"  Recorded Actions:\n    {actions_block}"
        )
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Prompt construction (loads the GEPA-tuned instruction at import time)
# ---------------------------------------------------------------------------

def _load_tuned_instruction() -> str:
    if not TUNED_CONFIG_PATH.exists():
        return DEFAULT_INSTRUCTION
    try:
        cfg = json.loads(TUNED_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_INSTRUCTION
    tuned = str(cfg.get("best_instruction") or "").strip()
    return tuned or DEFAULT_INSTRUCTION


SYSTEM_INSTRUCTION = os.getenv("ANALYST_INSTRUCTION") or _load_tuned_instruction()


_JSON_SCHEMA_HINT = (
    "Return STRICT JSON only, with EXACTLY these six keys (no extras, no markdown, no code fences):\n"
    f'  "risk_level": one of {VALID_RISK}\n'
    f'  "severity": one of {VALID_SEVERITY}\n'
    f'  "category_type": one of {VALID_CATEGORY}\n'
    '  "root_cause": 2-4 sentence paragraph identifying the most likely root cause(s).\n'
    '  "suggested_actions": 3-5 numbered actions, each on its own line, EXACTLY this shape:\n'
    '      "N. <imperative action sentence>. Owner: <role>. Timing: <Immediate | <30 days | 30-90 days | >90 days>. Verification: <text>."\n'
    '      DO NOT include the literal word "Action:". DO NOT use any unicode bullet characters.\n'
    '  "best_practices": 2-3 short lessons drawn from the historical cases. Plain strings\n'
    '      (or a JSON array of strings). DO NOT use "●", "•", "○" etc. — the renderer\n'
    '      will add its own bullet markers.\n'
)


def build_prompt(user_input: str, history_block: str) -> str:
    return (
        "User incident report (may be keywords, a short 'what happened', or a full report):\n"
        f"<<<\n{user_input.strip()}\n>>>\n\n"
        "Historical Methanex EPSSC cases (already TF-IDF retrieved, ranked by similarity):\n"
        f"{history_block}\n\n"
        f"{_JSON_SCHEMA_HINT}"
    )


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _ollama_chat(prompt: str, system: str, model: str | None = None) -> str:
    """Call Ollama Cloud with strict JSON formatting; raises on hard failure.

    Returns whatever textual content the model produced. Reasoning models
    (gpt-oss:*) frequently emit `message.content == ""` when `format="json"`
    is forced and instead place their JSON object inside `message.thinking`,
    so we read both fields and prefer the longer one.
    """
    if not OLLAMA_API_KEY:
        raise RuntimeError("OLLAMA_API_KEY is not set")

    from ollama import Client  # local import keeps cold-start fast

    model = model or OLLAMA_MODEL
    client = Client(
        host=OLLAMA_API_BASE,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        timeout=LLM_TIMEOUT_S,
    )
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        format="json",
        # 2500 tokens is enough for our 6-key JSON contract even when reasoning
        # models burn budget in `thinking`. 1100 used to truncate them mid-JSON.
        options={"temperature": 0.2, "num_predict": 2500},
    )

    msg = response.get("message") if isinstance(response, dict) else None
    if msg is None and hasattr(response, "message"):  # ollama.ChatResponse object
        msg = response.message

    content = ""
    thinking = ""
    if isinstance(msg, dict):
        content = (msg.get("content") or "").strip()
        thinking = (msg.get("thinking") or "").strip()
    elif msg is not None:
        content = (getattr(msg, "content", "") or "").strip()
        thinking = (getattr(msg, "thinking", "") or "").strip()

    # Prefer the channel that actually carries a JSON object. If `content` is
    # empty but `thinking` has braces, route that through the JSON extractor.
    if not content and thinking:
        log.info(
            "%s returned empty content; reading JSON from thinking channel (%d chars)",
            model, len(thinking),
        )
        return thinking
    if content and thinking and "{" not in content and "{" in thinking:
        log.info(
            "%s content has no JSON object; falling back to thinking channel",
            model,
        )
        return thinking
    return content


# ---------------------------------------------------------------------------
# Output sanitization
# ---------------------------------------------------------------------------

def _extract_json_payload(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _closest_label(value: Any, options: list[str], default: str) -> str:
    if value is None:
        return default
    val = str(value).strip()
    if not val:
        return default
    val_low = val.lower()
    for opt in options:
        if val_low == opt.lower():
            return opt
    for opt in options:
        if opt.lower() in val_low or val_low in opt.lower():
            return opt
    return default


# Unicode bullet characters that frequently leak in from the corpus or that
# different LLMs sometimes emit instead of plain markdown bullets. We normalize
# all of these to clean markdown lists so the Dash <Markdown> renders them
# consistently with the Methanex theme.
_BULLET_CHARS = "●○◯◉▪■□◦•‣⁃·"
_BULLET_CHARSET = set(_BULLET_CHARS)
_BULLET_SPLIT_RE = re.compile(rf"\s*[{re.escape(_BULLET_CHARS)}]+\s*")
_LEADING_LIST_NOISE_RE = re.compile(rf"^[\s\-\*{re.escape(_BULLET_CHARS)}]+")
_NUM_PREFIX_RE = re.compile(r"^\s*\(?\s*\d+\s*[\.\):\-]\s*")
_ACTION_PREFIX_RE = re.compile(r"^\s*(?:action|step|recommendation)\s*[:\-]?\s*", re.IGNORECASE)


def _has_unicode_bullets(text: str) -> bool:
    return any(c in text for c in _BULLET_CHARSET)


def _strip_list_noise(line: str) -> str:
    return _LEADING_LIST_NOISE_RE.sub("", line).strip()


def _ensure_terminal_punct(text: str) -> str:
    text = text.rstrip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _split_unicode_bullet_items(text: str) -> list[str]:
    """Split a string on '●'-style bullets, returning trimmed non-empty items."""
    parts = _BULLET_SPLIT_RE.split(str(text))
    return [p.strip() for p in parts if p and p.strip()]


def _to_markdown_bullets(value: Any) -> str:
    """Coerce a string / list / list-of-dicts into a clean markdown bullet list.

    - Unicode bullets ('●', '•', etc.) are converted to '- '.
    - Each item lives on its own line so dcc.Markdown renders a real <ul>.
    - If the input is plain prose with no bullets, it is returned untouched
      (so a single-paragraph root_cause stays a paragraph).
    """
    if value is None:
        return ""

    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("lesson") or item.get("practice") or item.get("description") or "").strip()
            else:
                text = str(item).strip()
            text = _strip_list_noise(text)
            if not text:
                continue
            items.append(_ensure_terminal_punct(text))
        return "\n".join(f"- {it}" for it in items)

    s = str(value).strip()
    if not s:
        return ""

    if _has_unicode_bullets(s):
        items = _split_unicode_bullet_items(s)
        if items:
            return "\n".join(f"- {_ensure_terminal_punct(it)}" for it in items)

    # Single line that already starts with "-" / "*" → light-touch normalization.
    if "\n" in s and any(line.strip().startswith(("-", "*")) for line in s.splitlines() if line.strip()):
        items = []
        for line in s.splitlines():
            line = _strip_list_noise(line)
            if line:
                items.append(_ensure_terminal_punct(line))
        return "\n".join(f"- {it}" for it in items)

    return s


def _to_numbered_actions(value: Any) -> str:
    """Coerce suggested_actions into '1. ... Owner: ... Timing: ... Verification: ...'.

    Removes any leading 'Action:' / 'Step:' prefix, strips '●', renumbers from 1,
    and puts each action on its own line so the Markdown renders a real <ol>.
    """
    if value is None:
        return ""

    raw_items: list[str] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                bits: list[str] = []
                main = str(item.get("action") or item.get("text") or item.get("description") or "").strip()
                if main:
                    bits.append(main.rstrip("."))
                owner = str(item.get("owner") or item.get("owner_role") or "").strip()
                if owner:
                    bits.append(f"Owner: {owner.rstrip('.')}")
                timing = str(item.get("timing") or item.get("timeframe") or "").strip()
                if timing:
                    bits.append(f"Timing: {timing.rstrip('.')}")
                verif = str(item.get("verification") or item.get("verify") or "").strip()
                if verif:
                    bits.append(f"Verification: {verif.rstrip('.')}")
                if bits:
                    raw_items.append(". ".join(bits) + ".")
            elif isinstance(item, str):
                raw_items.append(item.strip())
    else:
        s = str(value).strip()
        if not s:
            return ""
        # Replace unicode bullets with newlines so we can split uniformly.
        if _has_unicode_bullets(s):
            s = _BULLET_SPLIT_RE.sub("\n", s)
        # Models sometimes pack everything on one line ("1. ... 2. ... 3. ..."). Split that too.
        s_compact = re.sub(r"\s+", " ", s.replace("\n", " ")).strip()
        if "\n" not in s and re.search(r"\b\d+\s*[\.\)]\s", s_compact):
            s = re.sub(r"(?<=[.!?])\s+(?=\d+\s*[\.\)]\s)", "\n", s)
        for line in s.splitlines():
            line = line.strip()
            if line:
                raw_items.append(line)

    cleaned: list[str] = []
    for item in raw_items:
        item = _NUM_PREFIX_RE.sub("", item).strip()
        item = _ACTION_PREFIX_RE.sub("", item).strip()
        item = _strip_list_noise(item)
        if not item:
            continue
        # Capitalize first letter for visual consistency.
        if item[0].islower():
            item = item[0].upper() + item[1:]
        cleaned.append(_ensure_terminal_punct(item))

    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(cleaned))


def _to_clean_paragraph_or_list(value: Any) -> str:
    """Root-cause normalizer: returns a paragraph if the model gave one, otherwise bullets."""
    if isinstance(value, list):
        return _to_markdown_bullets(value)
    s = str(value or "").strip()
    if not s:
        return ""
    if _has_unicode_bullets(s):
        return _to_markdown_bullets(s)
    return s


def _normalize_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "risk_level": _closest_label(payload.get("risk_level"), VALID_RISK, "Medium"),
        "severity": _closest_label(payload.get("severity"), VALID_SEVERITY, "Potentially Significant"),
        "category_type": _closest_label(payload.get("category_type"), VALID_CATEGORY, "Near Miss"),
        "root_cause": _to_clean_paragraph_or_list(payload.get("root_cause")),
        "suggested_actions": _to_numbered_actions(payload.get("suggested_actions")),
        "best_practices": _to_markdown_bullets(payload.get("best_practices")),
    }


# ---------------------------------------------------------------------------
# Deterministic fallback (also feeds the prediction-only training metric)
# ---------------------------------------------------------------------------

def _modal_label(series: pd.Series, options: list[str], default: str) -> str:
    cleaned = [v for v in series.dropna().astype(str).tolist() if v]
    if not cleaned:
        return default
    counter = Counter(cleaned)
    for label, _ in counter.most_common():
        match = _closest_label(label, options, "")
        if match:
            return match
    return default


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if present, otherwise an empty Series.

    Wraps pandas' ambiguous `DataFrame.get(...)` typing (which Pyright sees as
    `Series | None`, and `df[name]` as `Series | DataFrame`) so call sites can
    rely on a concrete `pd.Series`.
    """
    if name in df.columns:
        col = df[name]
        if isinstance(col, pd.Series):
            return col
    return pd.Series(dtype="object")


def fallback_response(top_df: pd.DataFrame) -> dict[str, str]:
    """Generate a complete, valid response from the top-k cases without an LLM.

    All output strings are returned as already-clean markdown so the rendered
    Dash output looks identical whether the LLM responded or this fallback ran.
    """
    if top_df.empty:
        raw = {
            "risk_level": "Medium",
            "severity": "Potentially Significant",
            "category_type": "Near Miss",
            "root_cause": "Insufficient historical context retrieved. Treat as a precautionary near miss until a senior process safety engineer reviews.",
            "suggested_actions": [
                {
                    "action": "Conduct a structured pre-job hazard review for the described task",
                    "owner": "Operations Supervisor",
                    "timing": "Immediate",
                    "verification": "Permit-to-work signoff",
                },
                {
                    "action": "Verify isolation, venting, and energy control before any line break",
                    "owner": "Maintenance Lead",
                    "timing": "Immediate",
                    "verification": "Field walkdown checklist",
                },
                {
                    "action": "Refresh stop-work authority training for the affected crew",
                    "owner": "HSE Advisor",
                    "timing": "<30 days",
                    "verification": "Attendance log and post-training quiz",
                },
            ],
            "best_practices": [
                "Always confirm energy isolation physically; never rely on verbal handover",
                "Maintain housekeeping and clear sightlines around critical valves and vents",
                "Use formal stop-work authority as soon as conditions deviate from plan",
            ],
        }
        return _normalize_payload(raw)

    top5 = top_df.head(5)
    risk = _modal_label(_col(top5, "risk_level"), VALID_RISK, "Medium")
    sev = _modal_label(_col(top5, "severity"), VALID_SEVERITY, "Potentially Significant")
    cat = _modal_label(_col(top5, "category_type"), VALID_CATEGORY, "Near Miss")

    cause_items: list[str] = []
    for raw_cause in _col(top5, "root_causes").dropna().astype(str).tolist():
        for item in _split_unicode_bullet_items(raw_cause) or [raw_cause]:
            item = _strip_list_noise(item)
            if item and item.lower() not in {c.lower() for c in cause_items}:
                cause_items.append(item)
        if len(cause_items) >= 5:
            break
    cause_items = cause_items[:5]
    root_cause_value: Any = (
        cause_items
        if cause_items
        else "Pattern across similar historical cases points to a procedure / isolation gap combined with time pressure or limited refresher training."
    )

    action_dicts: list[dict[str, str]] = []
    seen_text: set[str] = set()
    for ev_id in top5["event_id"].tolist():
        for action in _actions_for(ev_id, max_actions=2):
            primary = action.split(" | ")[0].strip()
            key = primary[:80].lower()
            if not primary or key in seen_text:
                continue
            seen_text.add(key)
            owner = re.search(r"Owner: ([^|]+?)(?: \|| $)", action)
            timing = re.search(r"Timing: ([^|]+?)(?: \|| $)", action)
            verif = re.search(r"Verification: ([^|]+?)(?: \|| $)", action)
            action_dicts.append({
                "action": primary,
                "owner": owner.group(1).strip() if owner else "Operations Supervisor",
                "timing": timing.group(1).strip() if timing else "<30 days",
                "verification": verif.group(1).strip() if verif else "Compliance audit and signoff",
            })
            if len(action_dicts) >= 4:
                break
        if len(action_dicts) >= 4:
            break

    if not action_dicts:
        action_dicts.append({
            "action": "Re-verify isolation, venting, and energy control before any line break",
            "owner": "Operations Supervisor",
            "timing": "Immediate",
            "verification": "Permit-to-work field audit",
        })

    lessons_pool: list[str] = []
    for raw_lesson in _col(top5, "lessons").dropna().astype(str).tolist():
        for item in _split_unicode_bullet_items(raw_lesson) or [raw_lesson]:
            item = _strip_list_noise(item)
            # Keep only the first sentence of each lesson for readability.
            first_sentence = re.split(r"(?<=[.!?])\s+", item, maxsplit=1)[0].strip()
            if first_sentence and first_sentence.lower() not in {l.lower() for l in lessons_pool}:
                lessons_pool.append(first_sentence[:240])
        if len(lessons_pool) >= 3:
            break
    if not lessons_pool:
        lessons_pool = [
            "Treat all small-bore tubing as pressure-retaining until positively vented",
            "Reinforce structured pause points and stop-work authority before loosening fittings",
        ]
    lessons_pool = lessons_pool[:3]

    return _normalize_payload({
        "risk_level": risk,
        "severity": sev,
        "category_type": cat,
        "root_cause": root_cause_value,
        "suggested_actions": action_dicts,
        "best_practices": lessons_pool,
    })


# ---------------------------------------------------------------------------
# Markdown rendering (matches the existing Dash UI layout)
# ---------------------------------------------------------------------------

def render_markdown(
    payload: dict[str, str],
    top_df: pd.DataFrame,
    used_fallback: bool = False,
    model_used: str | None = None,
    merged_fields: list[str] | None = None,
) -> str:
    sim_summary = ""
    if not top_df.empty and "similarity" in top_df.columns:
        avg_sim = float(top_df["similarity"].head(10).mean())
        sim_summary = f" _(top-{len(top_df.head(10))} TF-IDF avg similarity {avg_sim:.2f})_"

    if used_fallback:
        notice = (
            "\n\n> ⚠️ _LLM unavailable — response generated deterministically from the historical corpus._"
        )
    elif model_used:
        merged_note = ""
        if merged_fields:
            merged_note = (
                f" Sections **{', '.join(merged_fields)}** were merged from the corpus to fill gaps."
            )
        notice = (
            f"\n\n> ✅ _Generated by `{model_used}` grounded on the top-{len(top_df.head(10))} "
            f"TF-IDF retrieved cases from the 2019-2024 Methanex corpus._{merged_note}"
        )
    else:
        notice = ""

    return (
        "### Predicted Risk Level & Severity\n"
        f"- **Risk Level:** {payload['risk_level']}\n"
        f"- **Severity:** {payload['severity']}\n"
        f"- **Category Type:** {payload['category_type']}\n\n"
        "### Potential Root Cause\n"
        f"{payload['root_cause']}\n\n"
        "### Suggested Actions\n"
        f"{payload['suggested_actions']}\n\n"
        "### Recommended Best Practices\n"
        f"{payload['best_practices']}"
        f"{sim_summary}"
        f"{notice}"
    )


# ---------------------------------------------------------------------------
# Public entry point — kept signature-compatible with the old rag_engine
# ---------------------------------------------------------------------------

# Sections that must be non-empty for a payload to be considered "LLM-generated"
# rather than discarded. Risk/severity/category are label-only and trivially
# normalized, so they are not part of this set; we instead require the
# narrative fields to be present.
_REQUIRED_LLM_FIELDS = ("root_cause", "suggested_actions", "best_practices")


def _try_llm(prompt: str, model: str) -> tuple[dict[str, str], str]:
    """Call one model; return (normalized_payload, raw_text). Raises on failure."""
    raw = _ollama_chat(prompt, SYSTEM_INSTRUCTION, model=model)
    if not raw or not raw.strip():
        raise RuntimeError("model returned empty response body")
    parsed = _extract_json_payload(raw)
    if not parsed:
        snippet = raw[:200].replace("\n", " ")
        raise RuntimeError(f"could not parse JSON from model output (head: {snippet!r})")
    normalized = _normalize_payload(parsed)
    return normalized, raw


def analyze_new_event(
    text: str,
    events_df: pd.DataFrame | None = None,  # accepted for backwards compatibility
    k: int = 10,
) -> tuple[str, pd.DataFrame]:
    """Generate the AI Safety Analyst response for a free-text incident query.

    Returns `(markdown_response, top_k_events_df)` so the existing Dash callback
    can populate the typewriter and the "Top 10 Similar Events" table.

    Pipeline:
      1. TF-IDF retrieval over the local 2019-2024 corpus → top-k cases (this
         is the local replacement for Vertex Vector Search).
      2. Build a RAG prompt that injects those cases as `historical_context`.
      3. Try the primary Ollama Cloud model with a strict JSON contract.
         If it fails (timeout, rate limit, parse error, missing fields),
         fall through the configured fallback model lineup.
      4. If the LLM produced *most* of the schema but is missing 1-2 sections,
         fill those gaps from the deterministic corpus fallback and STILL
         credit the response to the LLM.
      5. If every model fails, emit the deterministic corpus-only fallback
         (so the dashboard always renders a complete, well-formatted report).
    """
    text = (text or "").strip()
    if not text:
        return "Please enter an incident description.", EVENTS_DF.head(0).assign(similarity=[])

    top_k = retrieve_similar_events(text, k=k)
    history_block = format_history_block(top_k)
    prompt = build_prompt(text, history_block)
    log.info(
        "retrieved %d cases (avg sim %.3f); prompt length %d chars",
        len(top_k),
        float(top_k["similarity"].mean()) if not top_k.empty else 0.0,
        len(prompt),
    )

    candidate_models: list[str] = []
    for m in [OLLAMA_MODEL, *OLLAMA_FALLBACK_MODELS]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    payload: dict[str, str] | None = None
    used_model: str | None = None
    merged_fields: list[str] = []
    last_exc: Exception | None = None

    for model in candidate_models:
        try:
            log.info("calling Ollama Cloud model %s …", model)
            normalized, _ = _try_llm(prompt, model)
            missing = [f for f in _REQUIRED_LLM_FIELDS if not normalized.get(f)]
            if len(missing) > MAX_MISSING_FIELDS_BEFORE_FULL_FALLBACK:
                raise RuntimeError(
                    f"LLM payload missing too many sections: {missing}"
                )
            if missing:
                fb = fallback_response(top_k)
                for fld in missing:
                    normalized[fld] = fb[fld]
                merged_fields = missing
                log.warning(
                    "%s response missing %s; merged from corpus fallback",
                    model, missing,
                )
            payload = normalized
            used_model = model
            log.info("LLM analysis succeeded via %s", model)
            break
        except Exception as exc:  # noqa: BLE001 — any failure must degrade gracefully
            last_exc = exc
            log.warning("%s failed: %s: %s", model, type(exc).__name__, exc)
            continue

    used_fallback = payload is None
    if used_fallback:
        log.error(
            "all %d candidate models failed; using deterministic corpus fallback. "
            "Last error: %s",
            len(candidate_models), last_exc,
        )
        payload = fallback_response(top_k)
        if os.getenv("ANALYST_DEBUG") and last_exc is not None:
            payload["root_cause"] = f"[fallback: {last_exc}] " + payload["root_cause"]

    markdown = render_markdown(
        payload,
        top_k,
        used_fallback=used_fallback,
        model_used=used_model,
        merged_fields=merged_fields or None,
    )
    return markdown, top_k


__all__ = [
    "analyze_new_event",
    "retrieve_similar_events",
    "format_history_block",
    "build_prompt",
    "fallback_response",
    "render_markdown",
    "DEFAULT_INSTRUCTION",
    "SYSTEM_INSTRUCTION",
    "VALID_RISK",
    "VALID_SEVERITY",
    "VALID_CATEGORY",
    "RESPONSE_KEYS",
    "OLLAMA_MODEL",
    "OLLAMA_FALLBACK_MODELS",
    "OLLAMA_API_BASE",
    "LLM_TIMEOUT_S",
    "EVENTS_DF",
    "ACTIONS_DF",
    "TUNED_CONFIG_PATH",
    "_extract_json_payload",
    "_normalize_payload",
    "_ollama_chat",
    "_try_llm",
]
