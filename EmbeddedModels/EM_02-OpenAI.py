from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", # different name of model you can visit the website of openaiembeddings
    dimensions = 32  # dimenstion of vector which will it return.
    )

docs = [
    "Delhi is the capital of India.",
    "Kolkata is capital of West Bengal",
    "Paris is the capitla of France"
]

result = embedding.embed_documents(docs)

print(str(result))
