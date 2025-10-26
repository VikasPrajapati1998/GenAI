import os

# -------------------------------
# Offline mode
# -------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"

# -------------------------------
# Use the updated LangChain-HuggingFace import
# -------------------------------
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Local Jina Embeddings v2 model path
# -------------------------------
LOCAL_MODEL_PATH = r"D:\Study\AI\GenAI_EmbeddingModel\models\jina-embeddings-v2-base-en"

# -------------------------------
# Initialize embeddings
# -------------------------------
emb = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,
    model_kwargs={"device": "cpu"},       # Use "cuda" if GPU is available
    encode_kwargs={"normalize_embeddings": True}  # Optional normalization
)

# -------------------------------
# Embed a single query
# -------------------------------
query = "Hello world!"
vector = emb.embed_query(query)

print(f"Vector dimension: {len(vector)}")
print(f"Vector: {vector}")

# -------------------------------
# Embed multiple documents
# -------------------------------
documents = [
    "Hello world!",
    "How are you?",
    "This is a test document."
]

doc_vectors = emb.embed_documents(documents)
print(f"\nEmbedded {len(doc_vectors)} documents")
print(f"Each vector has dimension: {len(doc_vectors[0])}")
