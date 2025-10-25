from langchain_huggingface import HuggingFaceEmbeddings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model=MODEL)

query =  "Delhi is the capital of India"

vector = embedding.embed_query(query)

print(str(vector))

