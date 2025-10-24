# Offline TinyLlama Chat with LangChain

This project allows you to run a **local Hugging Face chat model** (TinyLlama-Chat) **fully offline** with LangChain on Windows 11. It uses Python, Transformers, and LangChain to generate responses from a local model without any internet connection.

---

## Step 1: Create Python Virtual Environment

1.1 Open a terminal or PowerShell and run:

```powershell
# Create virtual environment
python -m venv C:\envs\hf_offline

# Activate the environment
C:\envs\hf_offline\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

1.2 Install required Python packages (CPU version shown; see notes below for CUDA GPU):

```powershell
# Install CPU PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade

# Install Hugging Face, LangChain, and helpers
pip install transformers huggingface_hub sentencepiece tokenizers safetensors
pip install langchain langchain-huggingface
```

> **Note:** If you have a CUDA-compatible GPU, install PyTorch with CUDA instead:
>
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step 2: Download the Model (Online Machine)

Use the included `download_model.py` script to download your model locally:

```python
from huggingface_hub import snapshot_download

# Download TinyLlama-Chat
local_dir = snapshot_download(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="./models/TinyLlama-Chat"
)

print("Saved model to:", local_dir)
```

* The downloaded folder (`./models/TinyLlama-Chat`) will include:

  * `pytorch_model.bin` or `*.safetensors`
  * `config.json`
  * tokenizer files

---

## Step 3: Prepare Offline Environment

If your offline machine cannot access the internet, download all required wheels on your online machine:

```powershell
# Create a folder for offline wheels
mkdir wheelhouse

# Download PyTorch CPU wheel
pip download --dest wheelhouse torch --index-url https://download.pytorch.org/whl/cpu

# Download Transformers, LangChain, and helpers
pip download --dest wheelhouse transformers huggingface_hub sentencepiece tokenizers safetensors langchain langchain-huggingface
```

Copy the `wheelhouse` folder and the `models/TinyLlama-Chat` folder to the offline machine.

---

## Step 4: Set Environment Variables for Offline Usage

On the offline machine, configure Transformers and HF cache:

```powershell
# Prevent Transformers and datasets from trying to access the internet
setx TRANSFORMERS_OFFLINE "1"
setx HF_DATASETS_OFFLINE "1"

# Optional: set a custom HF cache root (where you put model folder)
setx HF_HOME "C:\hf_cache"
```

---

## Step 5: Run the Offline Chat Script

Use the provided `run_tinyllama_chat_offline.py` script to chat with the model:

```powershell
python run_tinyllama_chat_offline.py
```

* Type your questions in the console.
* The assistant will respond with clean answers (no repeated prompts).
* Type `exit` or `quit` to end the session.

---

## Tips for Better Output

1. Make sure `LOCAL_MODEL_PATH` in the script points to the local model folder:

```python
LOCAL_MODEL_PATH = r"C:\path\to\models\TinyLlama-Chat"
```

2. To get **short, concise answers**, the script strips the prompt from the model output.

3. Adjust generation parameters in the script:

```python
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.95
```

* Lower `TEMPERATURE` → more deterministic answers
* Increase `MAX_NEW_TOKENS` → longer answers

---

## References

* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* [LangChain Documentation](https://docs.langchain.com/)
* [TinyLlama-Chat on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

## Summary

This setup allows you to:

* Run a **local chat LLM** offline
* Avoid repeated prompts or extra text in outputs
* Use LangChain to manage prompts and pipelines
* Work fully on a Windows machine with CPU or GPU
