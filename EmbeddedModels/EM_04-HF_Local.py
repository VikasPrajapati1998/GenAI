import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

model_path = "./Model"
os.makedirs(model_path, exist_ok=True)

# -------------------------------------------------
# 1. Authenticate (your HF_TOKEN + accept license)
# -------------------------------------------------
load_dotenv(find_dotenv())
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("Set HUGGINGFACEHUB_ACCESS_TOKEN in .env")

login(token=hf_token)

# -------------------------------------------------
# 2. Load EmbeddingGemma (768-dim, bi-directional)
# -------------------------------------------------
model_name = "google/embeddinggemma-300m"  # Google's SOTA lightweight embedder

embeddings = HuggingFaceEmbeddings(
    cache_folder=model_path,
    model_name=model_name,
    model_kwargs={
        "device": "cpu",  # "cuda" for GPU acceleration
        "trust_remote_code": True  # Required for Gemma variants
    },
    encode_kwargs={
        "normalize_embeddings": True,  # L2 norm for cosine similarity
        "batch_size": 32  # Tune for throughput
    }
)

text = "Delhi is the capital of India."

output = embeddings.embed_query(text)
print(output)

