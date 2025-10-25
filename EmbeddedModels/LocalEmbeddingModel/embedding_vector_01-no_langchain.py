import os
from sentence_transformers import SentenceTransformer

# Enable offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Path to your locally downloaded model
LOCAL_MODEL_PATH = r"D:\Study\AI\GenAI_EmbeddingModel\models\all-MiniLM-L6-v2"

# Load the model offline
model = SentenceTransformer(LOCAL_MODEL_PATH)

# Example query
QUERY = "Delhi is the capital of India."

# Get embeddings
embedding = model.encode(QUERY)
print(f"Embedding for '{QUERY}':\n{embedding}")
print(f"Length of vector: {len(embedding)}")

