from sentence_transformers import SentenceTransformer
import os

# Model ID on Hugging Face
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Local folder where the model will be saved
LOCAL_MODEL_PATH = r"D:\Study\AI\GenAI_EmbeddingModel\models\all-MiniLM-L6-v2"

# Create folder if it does not exist
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

print("Downloading the Sentence-Transformers model (online)...")
# This downloads the proper Sentence-Transformers structure
model = SentenceTransformer(MODEL_ID, cache_folder=LOCAL_MODEL_PATH)

print(f"Model downloaded and saved at {LOCAL_MODEL_PATH}")
