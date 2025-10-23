from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize the Gemini model using the API key from your environment variables
# The model "gemini-2.5-flash" is available on the free tier
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

# Create a chain combining the prompt and the model
chain = prompt | llm

# Query
query = "What is the capital of India?"

# Invoke the chain with a question
response = chain.invoke({"input": query})

print(response.content)

