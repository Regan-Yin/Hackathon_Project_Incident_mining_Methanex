# Legacy RAG Engine (Vertex AI / Gemini)

> **STATUS: ARCHIVED — for historical reference only. NOT used by the live MVP.**
>
> This folder preserves the **first-generation** Retrieval-Augmented Generation
> stack that powered the Methanex EPSSC dashboard during the initial UBC MBAn
> Hackathon submission. It has been superseded by the lighter-weight, free-tier
> stack documented in the project root (`safety_analyst.py` + Ollama Cloud +
> DSPy/GEPA-tuned prompts). The files here are kept in version control purely
> so the original architecture can be inspected / re-run, and so the README can
> faithfully document the project's evolution.

## Why was it replaced?

| Concern | Old (this folder) | New (project root) |
|---|---|---|
| Retrieval | Vertex AI Matching Engine (Vector Search) | Local TF-IDF cosine similarity over `data/events_clean.csv` |
| Embeddings | `text-embedding-004` (Vertex AI) | None — TF-IDF on the corpus directly |
| LLM | `gemini-2.5-flash` via `langchain_google_vertexai` | Free-tier Ollama Cloud models (`gpt-oss:20b-cloud`, `gemini-3-flash-preview:cloud`, …) with cascading fallback |
| Prompt quality | Hand-written prompt | DSPy + GEPA-optimized instruction (`dspy_gepa_best_config.json`) |
| Cost | Pay-per-call (embeddings + endpoint hosting + LLM) | $0 (free Ollama Cloud tier) |
| Cold-start | ~30–45 min one-time GCP index build + endpoint deploy | < 5 s — TF-IDF is built in-memory at startup |
| Cloud deploy | Vertex AI endpoints (long-lived) | Cloud Run + Docker (scale-to-zero) |
| Failure mode | Hard dependency on GCP project / quota / billing | Deterministic corpus-only fallback if the LLM is unreachable |

The new stack is intentionally Methanex-portable: anyone can clone the repo,
add a free Ollama Cloud key, and run the dashboard locally **without** a GCP
project, billing account, or service-account key.

## What's in this folder

| File | Purpose |
|---|---|
| `rag_engine.py` | The original `analyze_new_event()` — embedded the user's hypothesis with `text-embedding-004`, called `MatchingEngineIndexEndpoint.find_neighbors()`, and prompted Gemini for the structured response. |
| `setup_cloud.py` | One-time bootstrap script that built the Vertex AI Vector Search index from `data/events_clean.csv`, uploaded JSONL embeddings to GCS, and deployed the index to a public endpoint. |
| `data_processing.py` | A small helper module (KPI calc + matplotlib/plotly graphs) that pre-dated the Dash app's inline implementation. No longer imported. |
| `gcp-key.json` | **Template only** — the placeholder service-account key the user used to copy/paste their real credentials over. Real credentials are never committed. |

## Deployment exclusion

The repository's `.gcloudignore` excludes this entire `legacy_rag_engine/`
folder so it is **never uploaded to Cloud Run** during a `gcloud run deploy`.
This keeps the production image small and prevents accidental Vertex AI
imports leaking into the live container. If you ever want to redeploy the
legacy stack, copy the files back into the project root and remove the
`legacy_rag_engine/` exclusion from `.gcloudignore`.

## How to run the legacy stack (only if you really want to)

> Requires an active GCP project with billing enabled, the Vertex AI API
> enabled, and a service-account JSON key.

1. Copy `legacy_rag_engine/rag_engine.py` and `legacy_rag_engine/setup_cloud.py`
   back into the project root.
2. Replace `gcp-key.json` placeholder values with your real service-account key,
   move it into the project root, and add the corresponding `GCP_*` and
   `VERTEX_*` variables to your `.env` (see the historical entries at the
   bottom of `.env.example`).
3. `python setup_cloud.py` — runs once, takes ~30–45 minutes, prints the
   endpoint and index IDs you must paste into `.env`.
4. Edit `app.py` and replace `from safety_analyst import analyze_new_event`
   with `from rag_engine import analyze_new_event`.
5. `python app.py` — the dashboard will now query the Vertex AI endpoint.

For the production / hackathon-MVP experience, **stay on the new stack** —
it is faster, cheaper, and works offline if Ollama Cloud is unreachable.
