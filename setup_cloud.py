import os
import pandas as pd
import json
import vertexai
from google.cloud import storage, aiplatform
from vertexai.language_models import TextEmbeddingModel
import time
from dotenv import load_dotenv

load_dotenv()

# ADD YOUR OWN VALUES: Set these in your .env file (see .env.example)
PROJECT_ID = os.getenv("GCP_PROJECT_ID")          # Your GCP project ID
LOCATION = os.getenv("GCP_LOCATION", "us-west1")   # Your GCP region
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{PROJECT_ID}-rag-data")  # Your GCS bucket name

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# 1. Preprocess Data
print("Loading and formatting data...")
events_df = pd.read_csv("data/events_clean.csv")
events_df['rag_content'] = (
    "Title: " + events_df['title'].fillna('') + 
    " | Category: " + events_df['category_type'].fillna('') +
    " | What Happened: " + events_df['what_happened'].fillna('') + 
    " | Root Causes: " + events_df['root_causes'].fillna('')
)

# 2. Generate Embeddings (in batches to avoid token limits)
print("Generating embeddings in batches...")
model = TextEmbeddingModel.from_pretrained("text-embedding-004")

texts = events_df['rag_content'].tolist()
embeddings = []
batch_size = 25  # Small batch size to stay safely under the 20,000 token limit

for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    
    # Get embeddings for this specific batch
    batch_results = model.get_embeddings(batch)
    
    # Extract the numerical vectors and add them to our main list
    embeddings.extend([emb.values for emb in batch_results])
    print(f"Embedded {len(embeddings)} / {len(texts)} cases...")
    
    # Pause for a split second to respect API rate limits
    time.sleep(0.5)

# 3. Format as JSONL (but save with a .json extension!)
print("Formatting as JSONL...")
with open("methanex_corpus.json", "w") as f:
    for i, row in events_df.iterrows():
        item = {
            "id": str(row['event_id']),
            "embedding": embeddings[i] 
        }
        f.write(json.dumps(item) + "\n")

# 4. Upload to Cloud Storage (into a clean subfolder)
print("Uploading to Cloud Storage...")
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME) # Use .bucket() since it was already created in the last run

# Notice we are putting it in an 'index_data/' folder so Vertex AI ignores the old .jsonl file
blob = bucket.blob("index_data/methanex_corpus.json")
blob.upload_from_filename("methanex_corpus.json")

# 5. Create Vector Search Index
print("Creating Index (This takes ~30-45 minutes)...")
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="methanex_safety_index",
    # Point the URI strictly to the new subfolder
    contents_delta_uri=f"gs://{BUCKET_NAME}/index_data", 
    dimensions=768,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE"
)

# 6. Deploy to Endpoint
print("Deploying Endpoint (This takes ~30 minutes)...")
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="methanex_safety_endpoint",
    public_endpoint_enabled=True
)
my_index_endpoint.deploy_index(
    index=my_index,
    deployed_index_id="methanex_deployed_index"
)

print(f"SAVE THIS ENDPOINT ID: {my_index_endpoint.name}")
print(f"SAVE THIS INDEX ID: {my_index.name}")