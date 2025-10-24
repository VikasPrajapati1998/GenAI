# run_local_llm.py
"""
Offline local HF model runner (no langchain chains). 
Uses local model folder and HF pipeline directly to generate text.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import textwrap

# ---- CONFIG ----
# LOCAL_MODEL_PATH = r"D:\\Study\\AI\\GenAI_LocalModel\\models\\gpt2" 
# LOCAL_MODEL_PATH = r"D:\\Study\\AI\\GenAI_LocalModel\\models\\TinyLlama-Chat"
LOCAL_MODEL_PATH = r"D:\\Study\\AI\\GenAI_LocalModel\\models\\microsoft\\phi-2"
TASK = "text-generation"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.95
DEVICE = -1   # -1 = CPU ; set 0 if you have CUDA GPU and torch supports it

# ---- Load tokenizer and model from local folder ----
print("Loading tokenizer from:", LOCAL_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

print("Loading model from:", LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

if DEVICE != -1 and torch.cuda.is_available():
    print("Using GPU")
    model = model.to(f"cuda:{DEVICE}")
else:
    print("Device set to use CPU")

# ---- HF pipeline using local model ----
text_gen_pipeline = pipeline(
    TASK,
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    top_p=TOP_P,
    temperature=TEMPERATURE,
)

# ---- Helper: simple prompt formatting + generate ----
SYSTEM_PROMPT = "You are a helpful assistant."
def build_prompt(user_input: str) -> str:
    # Simple template similar to the LangChain example
    return f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"

def generate(user_input: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    prompt = build_prompt(user_input)
    # pipeline returns a list of dicts with 'generated_text'
    outputs = text_gen_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        num_return_sequences=1,
    )
    # generated_text contains the prompt + completion; strip prompt to return completion only
    raw = outputs[0]["generated_text"]
    if raw.startswith(prompt):
        completion = raw[len(prompt):].strip()
    else:
        # fallback: attempt to remove the prompt heuristically
        completion = raw.replace(prompt, "").strip()
    return completion

if __name__ == "__main__":
    query = "What is programming?"
    print("\nUser query:\n", query)
    print("\nGenerating...\n")
    out = generate(query)
    print("--- Model output ---\n")
    # pretty print with wrapping
    print(textwrap.fill(out, width=100))
