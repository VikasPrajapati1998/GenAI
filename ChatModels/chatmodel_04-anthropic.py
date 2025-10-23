from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv(find_dotenv())

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

question = "what is the capital of India?"
response = model.invoke(question)

print(response.content)
