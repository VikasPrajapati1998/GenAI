from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# Initialize ChatModel
chatmodel = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-5" if available in your environment

# Invoke ChatModel
ques = "What is the capital of India?"
ans = chatmodel.invoke(ques)

print("---------------------------------------------")
print(ans)
print("--------------------------------------------\n")

print(ans.content)

