# ⚛️ Methanex Safety Intelligence: Incident & Near-Miss Pattern Mining

Welcome to the repository for **Regan's (Team 9)** solution to the Methanex Challenge #2: **Safety Incident & Near-Miss Pattern Mining**, developed during the UBC MBAn Hackathon 2026.

🌐 **Live MVP Dashboard:** [ilovemethanex.ca](http://ilovemethanex.ca)

---

## 📖 Background & Problem Statement

Methanex collects vast amounts of historical safety records, including incidents, near-misses, and root-cause analyses. While incredibly valuable, this unstructured text data is often reviewed case-by-case, making it difficult to see the bigger picture.

Our challenge was to bridge the gap between these raw, localized safety narratives and systemic business intelligence. We needed to:
1. Identify patterns and clusters of similar events (e.g., AI system failures, HR privacy exposures).
2. Understand the factors driving higher severity (actual incidents) versus high-potential warnings (near-misses).
3. Provide data-driven recommendations on where Methanex should focus its prevention efforts and training.

---

## 🛠️ Step-by-Step Solution Methodology

To move beyond static slides and local notebooks, we engineered an end-to-end data pipeline culminating in a production-ready, full-stack web application.

### Step 1: Data Ingestion & Preprocessing
We began with raw, messy text narratives and structured fields. We cleaned and standardized the datasets, extracting key temporal and categorical variables (like `year` and `category_type`) to build a solid foundation for our models.

### Step 2: NLP Pattern Mining & Clustering
Using the power of **Google Cloud Platform (Vertex AI & Gemini)**, we analyzed the unstructured text descriptions. We applied text embeddings and clustering techniques to group disparate safety events into distinct, actionable "Risk Scenarios" (e.g., *AI Monitoring & Decision-Support Errors* vs. *Chemical Transfer Leaks*). This mapping was saved into our core `case_cluster_map.csv`.

### Step 3: Severity Driver Analysis
We mapped our newly defined clusters against the ratio of **Incidents** (realized harm) to **Near-Misses** (free lessons). This allowed us to statistically identify which operational areas are "high-potential" risks that bypass current safety barriers, versus areas where existing safeguards are working.

### Step 4: MVP Web Deployment
Drawing on full-stack development experience, we elevated our localized Python scripts into a live MVP. We utilized **Dash** and **Plotly** to build an interactive front-end, styled with a custom corporate CSS theme (`style.css`), and deployed it to the web so stakeholders could dynamically explore trends and drill down into the data themselves.

---

## 💡 Key Findings & Recommendations

Our dashboard enables business leaders to easily pinpoint where to allocate resources:
* **Focus on Dominant Clusters:** By utilizing the Pareto analysis module, management can immediately see which operational clusters account for the highest volume of reports.
* **Target High-Severity Ratios:** We recommend prioritizing prevention efforts (training, new safeguards) on clusters that exhibit a high ratio of *Incidents* to *Near-Misses*, as these represent vulnerabilities where current defenses are frequently failing.
* **Proactive Monitoring:** Utilizing the timeline trend analysis, Methanex can spot emerging risks (e.g., a spike in specific IT/AI-related exposures in 2024) before they become systemic hazards.

---

## 📂 Repository File Structure

Below is the directory structure for this project. To reproduce the environment, you only need the cleaned `.csv` files provided in the `data/` directory.

```text
methanex-safety-intelligence/
│
├── data/
│   ├── events_clean.csv           # Cleaned core event data
│   ├── actions_clean.csv          # Cleaned associated actions
│   └── case_cluster_map.csv       # NLP generated cluster mappings
│
├── assets/
│   └── style.css                  # Custom Methanex corporate CSS styling
│
├── app.py                         # Main Dash application & deployment script
├── rag_engine.py                  # Vertex AI / Gemini integration module
├── setup_cloud.py                 # One-time GCP Vector Search setup script
├── data_processing.py             # Data loading & visualization helpers
├── requirements.txt               # Python package dependencies
├── .env.example                   # Template for required environment variables
├── .gitignore                     # Protects credentials from accidental commits
├── gcp-key.json                   # TEMPLATE — add your own GCP service account key
└── README.md                      # Project documentation
```
---
## Getting Started: Local Setup & GCP Configuration

To run this application smoothly on your local machine or your own server, you must configure your Google Cloud Platform (GCP) environment. The application relies heavily on Vertex AI and Vector Search to process and analyze the safety intelligence data.

### 1. Google Cloud Platform (GCP) Prerequisites

Before running the code, ensure you have set up the following in your GCP Console:

- **Create a GCP Project:** Initialize a new project and enable billing.
- **Enable APIs:** Enable the Vertex AI API to utilize Gemini for the text analysis and RAG functionalities.
- **Configure Vertex AI Vector Search:** You will need to set up a Vector Search index and deploy it to an endpoint to handle the similarity search for the clustered safety events.

### 2. Service Account & Credentials (`gcp-key.json`)

You must authenticate your application with GCP:

1. Navigate to **IAM & Admin > Service Accounts** in your GCP console.
2. Create a new service account and grant it the necessary roles (e.g., `Vertex AI User`).
3. Generate a new JSON key for this service account and download it.
4. Rename the downloaded file to `gcp-key.json` and place it in the root directory of this repository.
5. The included `gcp-key.json` is a **template with placeholder values** — replace all `YOUR_*` fields with your actual credentials.

> **Security:** `gcp-key.json` and `.env` are listed in `.gitignore` to prevent accidental exposure. Never commit real credentials to version control.

### 3. Environment Variables (`.env` Setup)

All sensitive configuration is managed through environment variables. No API keys or project IDs are hardcoded in the source code.

```bash
# Copy the example file to create your own .env
cp .env.example .env
```

Then open `.env` and fill in your values:

| Variable | Description | Example |
|---|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID | `my-project-123456` |
| `GCP_LOCATION` | GCP region for Vertex AI | `us-west1` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to your service account key | `./gcp-key.json` |
| `VERTEX_ENDPOINT_ID` | Matching Engine Endpoint ID (from `setup_cloud.py` output) | `1234567890123456789` |
| `VERTEX_DEPLOYED_INDEX_ID` | Deployed Index ID (from `setup_cloud.py` output) | `my_deployed_index` |
| `GCS_BUCKET_NAME` | GCS bucket for embeddings (from `setup_cloud.py` output) | `my-project-rag-data` |

### 4. Running the Application
Once your data is in the `data/` folder, your `.env` is configured, and your GCP credentials are authenticated:

```bash
# Install dependencies
pip install -r requirements.txt

# (First time only) Set up Vertex AI Vector Search index
python setup_cloud.py

# Run the Dash server
python app.py
```

Navigate to http://127.0.0.1:8050/ in your browser to view the dashboard.