# Methanex Safety Intelligence: Incident & Near-Miss Pattern Mining

Welcome to the repository for **Regan's (Team 9)** solution to the Methanex Challenge #2: **Safety Incident & Near-Miss Pattern Mining**, developed during the UBC MBAn Hackathon 2026.

**Live MVP Dashboard:** [ilovemethanex.ca](http://ilovemethanex.ca)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Architecture (v2 — Apr 2026)](#2-solution-architecture-v2--apr-2026)
3. [What's New Since v1](#3-whats-new-since-v1)
4. [Repository Layout](#4-repository-layout)
5. [Step-by-Step Replication](#5-step-by-step-replication)
6. [Running the DSPy + GEPA Prompt Optimizer](#6-running-the-dspy--gepa-prompt-optimizer)
7. [Deploying to Google Cloud Run](#7-deploying-to-google-cloud-run)
8. [Key Findings & Recommendations](#8-key-findings--recommendations)
9. [Legacy Vertex AI Stack](#9-legacy-vertex-ai-stack)

---

## 1. Problem Statement

Methanex collects vast amounts of historical safety records — incidents, near-misses, and root-cause analyses. While this unstructured text data is valuable, it is usually reviewed case-by-case, so systemic patterns stay hidden. Our job was to bridge that gap by:

1. Identifying clusters of similar events (e.g., AI system failures, HR privacy exposures, chemical-transfer leaks).
2. Quantifying which operational areas drive higher severity (Incidents) versus high-potential warnings (Near Misses).
3. Surfacing data-driven recommendations for where Methanex should focus prevention efforts.
4. Letting an investigator paste a hypothetical "what happened" snippet and immediately get a grounded, structured AI risk assessment with comparable historical cases.

---

## 2. Solution Architecture (v2 — Apr 2026)

The end-to-end pipeline is composed of four production-ready layers, all running locally in a single Dockerized Dash app:

```text
                      ┌─────────────────────────────────────────────┐
   User input  ─────▶ │  app.py  (Dash UI + Plotly visuals)         │
   (text query)       └────────────────┬────────────────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────────────────┐
                      │  safety_analyst.analyze_new_event()         │
                      │  ─────────────────────────────────────────  │
                      │  1. TF-IDF retrieval over the 2019-2024     │
                      │     events corpus (top-k = 10)              │
                      │  2. Build a strict-JSON RAG prompt          │
                      │  3. Call Ollama Cloud with cascading model  │
                      │     fallback (gpt-oss:20b → gemini-3-flash  │
                      │     → gpt-oss:120b → qwen3-coder:480b)      │
                      │  4. Validate / normalize the JSON output    │
                      │  5. Merge gaps from the deterministic       │
                      │     corpus fallback if a section is missing │
                      └────────────────┬────────────────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────────────────┐
                      │  dspy_gepa_best_config.json                 │
                      │  (GEPA-optimized system instruction —       │
                      │   loaded at import time)                    │
                      └─────────────────────────────────────────────┘
```

### Why this matters

| Concern | Old v1 (Vertex AI / Gemini) | New v2 (Ollama Cloud + DSPy/GEPA) |
|---|---|---|
| Retrieval | `MatchingEngineIndexEndpoint.find_neighbors()` | Local TF-IDF on `data/events_clean.csv` |
| Embeddings | `text-embedding-004` (Vertex AI, paid) | None — TF-IDF, $0 |
| LLM | `gemini-2.5-flash` via `langchain_google_vertexai` | Free-tier Ollama Cloud cascade |
| Prompt | Hand-written | DSPy + GEPA reflective optimization |
| Cold-start | 30-45 min one-time GCP index build | < 5 s in-memory TF-IDF build |
| Cloud-deploy | Vertex endpoints (always-on) | Cloud Run + Docker (scale-to-zero) |
| Cost to reproduce | GCP project + billing + endpoint hosting | **Zero** — free Ollama key only |
| Failure mode | Hard 5xx if GCP quota / billing fails | Deterministic corpus-only fallback |

Anyone can now clone the repo, paste a free Ollama Cloud API key into `.env`, and run the dashboard locally without GCP, billing accounts, or service-account keys.

---

## 3. What's New Since v1

This commit migrates the project from the original Vertex AI / Gemini stack to a leaner, free-tier-friendly stack. Concretely:

### Added
- **`safety_analyst.py`** — drop-in replacement for the old `rag_engine.py`. Implements TF-IDF retrieval, the cascading Ollama Cloud LLM call, JSON sanitization, and the deterministic corpus-only fallback. Public API (`analyze_new_event(text, events_df=None, k=10)`) is **unchanged**, so the Dash app needed only a one-line import swap.
- **`dspy_gepa_benchmark.py`** — DSPy + GEPA optimization driver. Builds a stratified eval set, sweeps four hand-authored prompt styles against the live analyzer, then runs `dspy.GEPA` with a high-temperature reflection LM and structured metric feedback to iteratively improve the system instruction.
- **`dspy_gepa_best_config.json`** — the persisted "winner" system instruction (currently the `operational` style with GEPA-tuned format rules). Loaded by `safety_analyst.py` at import time.
- **`dspy_gepa_eval_cases.json`** — the stratified eval cases dump used for reproducibility and scoring.
- **`Dockerfile`** — production-ready slim image (Python 3.11, gunicorn, 1 worker × 8 threads, 120s timeout) for Google Cloud Run.
- **`.gcloudignore`** — excludes legacy code, dev artifacts, virtualenvs, and dotfiles from `gcloud run deploy` uploads.
- **`legacy_rag_engine/`** — archive folder containing the old Vertex AI / Gemini stack (`rag_engine.py`, `setup_cloud.py`, `data_processing.py`, `gcp-key.json` template) plus a folder-level README documenting why it was replaced and how to re-enable it.

### Changed
- **`app.py`** — the only functional change is a one-line import swap (`from rag_engine import analyze_new_event` → `from safety_analyst import analyze_new_event`) and removal of the `GOOGLE_APPLICATION_CREDENTIALS` bootstrap (no longer required). All visuals, KPIs, callbacks, and the Dash layout are identical.
- **`requirements.txt`** — replaced GCP / Vertex AI / LangChain dependencies with `ollama`, `dspy>=2.6.0`, and `gepa>=0.0.4`. Added `gunicorn` for Cloud Run.
- **`.env.example`** — rewritten around `OLLAMA_API_KEY` and the optional cascade-control variables. Old GCP entries are kept commented at the bottom for users who want to re-enable the legacy stack.
- **`.gitignore`** — modernized: stricter dotenv handling, more Python tooling caches, and explicit exclusion of `gcp-key.json` from the legacy folder.
- **`README.md`** — this file. Documents v2 end-to-end with replication steps.

### Removed (moved to `legacy_rag_engine/`)
- `rag_engine.py`, `setup_cloud.py`, `data_processing.py`, `gcp-key.json`. Still tracked in git for historical reference but **excluded from Cloud Run uploads** via `.gcloudignore`.

---

## 4. Repository Layout

```text
methanex-safety-intelligence/
│
├── app.py                          # Dash dashboard (UI, KPIs, AI Analyst tab)
├── safety_analyst.py               # NEW — TF-IDF + Ollama Cloud RAG engine
├── dspy_gepa_benchmark.py          # NEW — DSPy + GEPA prompt-optimization driver
├── dspy_gepa_best_config.json      # NEW — persisted GEPA-tuned system instruction
├── dspy_gepa_eval_cases.json       # NEW — stratified eval cases (reproducibility)
│
├── Dockerfile                      # NEW — Cloud Run production image
├── .gcloudignore                   # NEW — excludes legacy / dev files from deploy
├── .env.example                    # Updated — Ollama Cloud + optional legacy vars
├── .gitignore                      # Updated — broader Python tooling coverage
├── requirements.txt                # Updated — Ollama, DSPy, GEPA, gunicorn
│
├── data/
│   ├── events_clean.csv            # Cleaned 2019-2024 event corpus
│   ├── actions_clean.csv           # Cleaned recommended actions
│   ├── case_cluster_map.csv        # NLP-generated cluster mappings
│   ├── case_priority_scores.csv
│   ├── cluster_profile_sorted.csv
│   ├── cluster_summary_with_terms_examples.csv
│   └── near_miss_early_warning_dashboard.csv
│
├── assets/
│   ├── style.css                   # Methanex corporate CSS
│   ├── logo.svg
│   └── favicon.ico
│
├── legacy_rag_engine/              # ARCHIVED — Vertex AI / Gemini v1 stack
│   ├── README.md                   # Why this folder exists + how to re-enable
│   ├── rag_engine.py               # Original Vertex Vector Search RAG
│   ├── setup_cloud.py              # One-time GCP index/endpoint bootstrap
│   ├── data_processing.py          # Historical KPI helper (no longer imported)
│   └── gcp-key.json                # Template service-account JSON
│
├── LICENSE
└── README.md                       # This file
```

---

## 5. Step-by-Step Replication

The default path requires only Python 3.11+ and a free Ollama Cloud key — **no GCP project**.

### 5.1. Clone and create a virtualenv

```bash
git clone https://github.com/<your-fork>/Hackathon_Project_Incident_mining_Methanex.git
cd Hackathon_Project_Incident_mining_Methanex

python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2. Get a free Ollama Cloud API key

1. Sign in at <https://ollama.com>.
2. Open **Settings → API Keys** and create a new key.
3. Copy the key (it looks like `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.YYYYY...`).

### 5.3. Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```dotenv
OLLAMA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.YYYYYYYYYYYYYYYYYYYYY
```

That is the **only required variable**. Every other setting falls back to a sensible default (see `.env.example` for the full list).

### 5.4. Run the dashboard

```bash
python app.py
```

The console will print:

```text
Dash is running on http://127.0.0.1:8050/
```

Open <http://127.0.0.1:8050/> in your browser.

### 5.5. Verify the AI Safety Analyst tab

1. Navigate to the **Generative AI Safety Analyst** tab.
2. Paste any short snippet, e.g. *"Operator slipped near unsealed valve during night shift"*, into the textarea.
3. Click **Analyze**. Within ~5–10 s you should see:
    - A predicted **Risk Level**, **Severity**, and **Category Type**.
    - A grounded **Root Cause** paragraph.
    - 3–5 **Suggested Actions** with Owner / Timing / Verification.
    - 2–3 corpus-derived **Best Practices**.
    - The **Top 10 Similar Historical Events** table at the bottom.

If the LLM is unreachable, the same UI still renders — the deterministic corpus-only fallback fills every section and the response is annotated *"LLM unavailable — response generated deterministically from the historical corpus."*

---

## 6. Running the DSPy + GEPA Prompt Optimizer

The system instruction shipping in `dspy_gepa_best_config.json` was produced by `dspy_gepa_benchmark.py`. To re-tune it on your own machine:

```bash
# Quickest sanity check (~5 min on free-tier Ollama Cloud)
python dspy_gepa_benchmark.py --num-cases 24 --auto light

# Better quality (~20-30 min)
python dspy_gepa_benchmark.py --num-cases 36 --auto medium

# Just dump the eval cases without running GEPA
python dspy_gepa_benchmark.py --prepare-only
```

The script:

1. Builds a stratified eval set from `data/events_clean.csv` (5 input styles × 3 categories).
2. Sweeps four hand-authored prompt styles (`balanced`, `strict_format`, `evidence_grounded`, `operational`) against the live analyzer pipeline and picks the best.
3. Wraps a `dspy.ChainOfThought` module around a strict-JSON signature.
4. Runs `dspy.GEPA` with a high-temperature reflection LM and a metric that scores label exactness, action structure, root-cause grounding, and length.
5. Persists the winning system instruction to `dspy_gepa_best_config.json` (the next `python app.py` will pick it up automatically).

Key environment overrides for this run:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | _(required)_ | Auth for both DSPy LM and reflection LM |
| `DSPY_MODEL` | `ollama_chat/$OLLAMA_MODEL` | Override the optimization target model |
| `DSPY_REFLECTION_MODEL` | same as `DSPY_MODEL` | Override the GEPA reflection model |

---

## 7. Deploying to Google Cloud Run

The included `Dockerfile` produces a slim Python 3.11 image that binds `gunicorn` to `$PORT` (Cloud Run's default 8080):

```bash
# 1. Build and push to Artifact Registry (one-time setup of the registry omitted)
PROJECT_ID=your-gcp-project
REGION=us-west1

gcloud builds submit \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/methanex/epssc-dashboard:latest

# 2. Deploy to Cloud Run with the Ollama key wired in via Secret Manager
gcloud secrets create OLLAMA_API_KEY --data-file=- <<< "<paste-your-key>"

gcloud run deploy methanex-epssc \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/methanex/epssc-dashboard:latest \
  --region $REGION \
  --allow-unauthenticated \
  --update-secrets OLLAMA_API_KEY=OLLAMA_API_KEY:latest \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 40 \
  --timeout 180
```

`.gcloudignore` ensures the upload skips:

- The entire `legacy_rag_engine/` folder (no Vertex AI dependency leaks).
- Local `.env` files (so the Secret Manager mount wins).
- Caches, virtualenvs, notebooks, and the dev-only `dspy_gepa_eval_cases.json`.

---

## 8. Key Findings & Recommendations

The dashboard makes it easy for safety leadership to pinpoint where to allocate resources:

- **Focus on dominant clusters.** The Pareto module shows which operational clusters account for the highest report volume and highest combined Risk × Severity priority.
- **Target high-severity ratios.** Clusters with a high *Incident-to-Near-Miss* conversion rate (e.g., AI Monitoring & Decision-Support Errors in 2024) are vulnerabilities where current defenses are frequently failing — those should be the top investment areas.
- **Proactive monitoring.** The timeline module surfaces emerging risks (e.g., a 2024 spike in IT/AI-related exposures) before they become systemic hazards.
- **Grounded triage at the desk.** The Generative AI Safety Analyst tab gives any investigator a structured, corpus-grounded triage of a free-text incident in seconds — including the top-10 most similar historical events for cross-reference.

---

## 9. Legacy Vertex AI Stack

The original v1 implementation (Vertex AI Vector Search + Gemini 2.5 Flash) is preserved under [`legacy_rag_engine/`](./legacy_rag_engine/) for reference. See [`legacy_rag_engine/README.md`](./legacy_rag_engine/README.md) for:

- A side-by-side comparison of v1 vs v2.
- Instructions for reactivating the Vertex AI stack if you have a GCP project.
- Why the folder is excluded from Cloud Run deploys via `.gcloudignore`.

> **Security note:** `gcp-key.json` (in either location) is a **template only**. Real service-account keys must never be committed — both `.gitignore` and `.gcloudignore` enforce this.

---

## License

See [`LICENSE`](./LICENSE).
