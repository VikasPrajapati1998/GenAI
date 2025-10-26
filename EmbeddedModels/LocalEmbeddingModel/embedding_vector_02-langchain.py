import os
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Enable offline mode
# -------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"

# -------------------------------
# Local model path
# -------------------------------
LOCAL_MODEL_PATH = r"D:\Study\AI\GenAI_EmbeddingModel\models\all-MiniLM-L6-v2"

# -------------------------------
# Load SentenceTransformer model (optional - for direct usage)
# -------------------------------
sbert_model = SentenceTransformer(LOCAL_MODEL_PATH)

# -------------------------------
# LangChain HuggingFaceEmbeddings wrapper
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------------------
# Example query
# -------------------------------
QUERY = "Hello world!"

# Get embedding vector
vector = embeddings.embed_query(QUERY)

print(f"Embedding for query: '{QUERY}'")
print(f"Vector: {vector}")
print(f"Vector dimension: {len(vector)}")
print(f"First 10 values: {vector[:10]}")
print("\n")

# Optional: Embed multiple documents
documents = [
    "Hello world!",
    "How are you?",
    "This is a test document."
]

doc_vectors = embeddings.embed_documents(documents)
print(f"\nEmbedded {len(doc_vectors)} documents")
print(f"Each vector has dimension: {len(doc_vectors[0])}")


