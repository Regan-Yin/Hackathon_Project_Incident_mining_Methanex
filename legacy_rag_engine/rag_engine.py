import os
import pandas as pd
import vertexai
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()

# ADD YOUR OWN VALUES: Set these in your .env file (see .env.example)
PROJECT_ID = os.getenv("GCP_PROJECT_ID")        # Your GCP project ID
LOCATION = os.getenv("GCP_LOCATION", "us-west1") # Your GCP region

# ADD YOUR OWN VALUES: These are output after running setup_cloud.py
ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")           # Your Matching Engine Index Endpoint ID
DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID") # Your Deployed Index ID

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def analyze_new_event(hypothesis_text, events_df):
    """Finds Top 10 similar events via GCP Vector Search and generates analysis."""
    
    # 1. Embed user query using native Vertex AI (bypassing LangChain validation)
    embeddings_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    # We pass the text in a list, and extract the .values from the first result
    query_vector = embeddings_model.get_embeddings([hypothesis_text])[0].values
    
    # 2. Search GCP Endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_ID)
    response = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=10
    )
    
    # Extract IDs of the top 10 matching historical events
    top_10_ids = [neighbor.id for neighbor in response[0]]
    top_10_events = events_df[events_df['event_id'].isin(top_10_ids)]
    
    # 3. Construct Context for Gemini - Include all columns
    context = "TOP 10 HISTORICAL METHANEX EVENTS:\n\n"
    for idx, (_, row) in enumerate(top_10_events.iterrows(), 1):
        context += f"Event {idx}:\n"
        for col in top_10_events.columns:
            value = row[col]
            # Skip NaN values
            if pd.notna(value):
                context += f"  {col}: {value}\n"
        context += "\n"
        
    prompt = f"""
    You are an expert Process Safety Engineer for Methanex EPSSC. 
    Analyze the following hypothetical incident/near-miss report based strictly on the historical events provided.

    Hypothetical Event Input:
    {hypothesis_text}
    
    {context}
    
    Provide your analysis STRICTLY in the following format refer to the TOP 10 HISTORICAL METHANEX EVENTS examples, Do NOT deviate from this structure:
    
    ### Predicted Risk Level & Severity
    - Risk Level (MUST be one of: "low", "medium", "high")
    - Severity (MUST be one of: "minor", "potentially significant", "near miss", "serious", "major")
    
    ### Potential Root Cause
    
    ### Suggested Actions
    """
    
    # 4. Generate AI Response
    # Note: Ensure the model name is valid for your region (gemini-1.5-flash is standard)
    llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0.2)
    ai_response = llm.invoke(prompt)
    
    return ai_response.content, top_10_events