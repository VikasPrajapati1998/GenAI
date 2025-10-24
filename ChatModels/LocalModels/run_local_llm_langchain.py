# run_local_llm_langchain.py
"""
LangChain v1.x + local Hugging Face model (offline compatible).
Uses HuggingFacePipeline from langchain-huggingface and RunnableSequence for chaining.
"""

import os
import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---- CONFIG ----
LOCAL_MODEL_PATH = r"D:\\Study\\AI\GenAI_LocalModel\\models\\microsoft\\phi-2"
TASK = "text-generation"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.95
DEVICE = -1   # -1 = CPU ; set 0 if you have CUDA GPU

# ---- Load model + tokenizer ----
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
text_gen_pipeline = pipeline(
    TASK,
    model=model,
    tokenizer=tokenizer,
    device=0 if (DEVICE != -1 and torch.cuda.is_available()) else -1,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    top_p=TOP_P,
    temperature=TEMPERATURE,
)

# ---- LangChain v1.x integration ----
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Wrap the HF pipeline as a LangChain LLM
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# ---- Define prompt template ----
template = """You are a helpful assistant.
User: {user_input}
Assistant:"""
prompt = ChatPromptTemplate.from_template(template)

# ---- Build Runnable chain (Prompt -> Model) ----
chain = RunnableSequence(prompt | llm)

# ---- Run Query ----
def run_query(query: str) -> str:
    result = chain.invoke({"user_input": query})
    # Some pipeline models return structured output; ensure plain text
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    return str(result)

if __name__ == "__main__":
    query = "What is machine learning?"
    print("\nUser query:\n", query)
    print("\nGenerating...\n")
    output = run_query(query)
    print("\n--- LangChain v1 model output ---\n")
    print(textwrap.fill(output, width=100))
