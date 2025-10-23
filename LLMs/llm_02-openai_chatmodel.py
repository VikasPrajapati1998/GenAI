from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-5" if available in your environment

# Query
ques = "What is the capital of India?"

# Invoke LLM
ans = llm.invoke(ques)

print(ans.content)

