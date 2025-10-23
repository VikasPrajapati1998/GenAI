import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    huggingfacehub_api_token=api_token,
    temperature=0.7  # Adjust for less randomness
)

# Initialize ChatHuggingFace for conversational tasks
chat_model = ChatHuggingFace(llm=llm)


# Query the model
query = "What is the capital of India?"
response = chat_model.invoke([HumanMessage(content=query)])
print(response.content)
