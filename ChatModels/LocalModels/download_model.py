# download_model.py  (run while online)
from huggingface_hub import snapshot_download

# set this to model id you want, e.g. "gpt2" or "gpt2-medium" or "bigmodel/whatever"
# MODEL_ID = "gpt2"
MODEL_ID = "microsoft/phi-2"
local_dir = snapshot_download(repo_id=MODEL_ID, 
                              revision="main", 
                              cache_dir="./models_cache", 
                              local_dir=f"./models/{MODEL_ID}", 
                              allow_patterns=None)

# download TinyLlama 
# local_dir = snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir="./models/TinyLlama-Chat")


print("Saved model to:", local_dir)


# setx HF_HOME "D:\\Study\\AI\\GenAI_LocalModel\\hf_cache"

