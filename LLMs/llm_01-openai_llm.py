from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = OpenAI(model="gpt-3.5-turbo-instruct")

response = llm.invoke("What is the capital of India.")
print(response)


