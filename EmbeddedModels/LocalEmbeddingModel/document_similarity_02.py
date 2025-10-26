# document_similarity_cached.py
import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Enable offline mode
# -------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"

# -------------------------------
# Local model path
# -------------------------------
LOCAL_MODEL_PATH = r"D:\Study\AI\GenAI_EmbeddingModel\models\all-MiniLM-L6-v2"

# -------------------------------
# Initialize embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------------------
# Sample documents
# -------------------------------
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style, consistency, and strong leadership as captain of the Indian cricket team.",
    "MS Dhoni is a former Indian cricket captain, celebrated for his calm demeanor, sharp wicketkeeping, and exceptional finishing skills in limited-overs cricket.",
    "Sachin Tendulkar is a legendary Indian batsman, often regarded as one of the greatest cricketers of all time for his records and contribution to Indian cricket.",
    "Rohit Sharma is an Indian opening batsman known for his elegant stroke play and record of scoring multiple double centuries in One Day Internationals.",
    "Kapil Dev is a former Indian all-rounder who led India to its first Cricket World Cup victory in 1983 and is remembered for his powerful batting and fast bowling.",
]

# -------------------------------
# Vector cache file
# -------------------------------
VECTOR_CACHE_FILE = "doc_vectors.npy"

# -------------------------------
# Load or compute document embeddings
# -------------------------------
if os.path.exists(VECTOR_CACHE_FILE):
    print("Loading cached document embeddings...")
    doc_vectors = np.load(VECTOR_CACHE_FILE)
else:
    print("Computing document embeddings...")
    doc_vectors = embeddings.embed_documents(documents)
    np.save(VECTOR_CACHE_FILE, doc_vectors)
    print(f"Saved embeddings to {VECTOR_CACHE_FILE}")

print(f"Embedded {len(doc_vectors)} documents")
print(f"Each vector has dimension: {len(doc_vectors[0])}\n")

# -------------------------------
# Interactive query loop
# -------------------------------
while True:
    query = input("Your: ")
    if query.lower() in ["exit", "quit", "finish"]:
        print("Bye Bye ...!\n")
        break

    query_vector = embeddings.embed_query(query)
    similarity_matrix = cosine_similarity(doc_vectors, [query_vector])

    index, score = sorted(list(enumerate(similarity_matrix)), key=lambda x: x[1])[-1]

    print(f"Model({score[0]:.4f}): {documents[index]}\n")
