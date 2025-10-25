from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", # different name of model you can visit the website of openaiembeddings
    dimensions = 32  # dimenstion of vector which will it return.
    )

query = "Delhi is the capital of India."
result = embedding.embed_query(query)

print(str(result))

