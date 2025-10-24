# run_tinyllama_chat.py
"""
Offline TinyLlama-Chat with LangChain using HuggingFacePipeline
(no HuggingFaceChatPipeline required)
"""

import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# ---- CONFIG ----
LOCAL_MODEL_PATH = r"D:\\Study\\AI\\GenAI_LocalModel\\models\\TinyLlama-Chat"
DEVICE = -1  # -1 = CPU ; set 0 if you have GPU with CUDA
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.95

# ---- Load tokenizer + model ----
print("Loading tokenizer from:", LOCAL_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

print("Loading model from:", LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

if DEVICE != -1 and torch.cuda.is_available():
    print("Using GPU")
    model = model.to(f"cuda:{DEVICE}")
else:
    print("Using CPU")

# ---- HF text generation pipeline ----
chat_pipeline = pipeline(
    "text-generation",  # use "chat" if your transformers version supports it
    model=model,
    tokenizer=tokenizer,
    device=0 if (DEVICE != -1 and torch.cuda.is_available()) else -1,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=TEMPERATURE,
    top_p=TOP_P
)

# ---- LangChain wrapper ----
llm = HuggingFacePipeline(pipeline=chat_pipeline)

# ---- Prompt template ----
template = """You are a helpful assistant.
User: {user_input}
Assistant:"""
prompt = ChatPromptTemplate.from_template(template)

# ---- Runnable chain ----
chain = RunnableSequence(prompt | llm)

# ---- Helper ----
def run_query(query: str) -> str:
    prompt_text = f"Answer the following question clearly and concisely:\nQuestion: {query}\nAnswer:"
    result_list = chat_pipeline(
        prompt_text,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        num_return_sequences=1
    )
    raw_output = result_list[0]["generated_text"]
    # strip prompt
    if raw_output.startswith(prompt_text):
        answer = raw_output[len(prompt_text):].strip()
    else:
        answer = raw_output.strip()
    return answer


# ---- Interactive chat loop ----
if __name__ == "__main__":
    print("Offline TinyLlama-Chat LangChain (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = run_query(user_input)
        print("\nAssistant:\n")
        print(textwrap.fill(response, width=100))
