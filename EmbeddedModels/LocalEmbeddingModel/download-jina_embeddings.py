# download_models.py
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_OFFLINE"] = "0"

# Model id on Hugging Face
MODEL_ID = "jinaai/jina-embeddings-v2-base-en"

# Local folder where the model will be saved (change if you want)
LOCAL_MODEL_PATH = r"D:\Study\AI\GenAI_EmbeddingModel\models\jina-embeddings-v2-base-en"

if __name__ == "__main__":
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {LOCAL_MODEL_PATH} (this may take a while)...")

    # snapshot_download will save the model files into LOCAL_MODEL_PATH (as a HF repo)
    # It may decide to symlink into the HF cache system; on Windows you may see symlink warnings.
    snapshot_download(repo_id=MODEL_ID, cache_dir=LOCAL_MODEL_PATH)

    print("Download finished.")
    print("Contents of the folder (top-level):")
    for p in sorted(os.listdir(LOCAL_MODEL_PATH)):
        print("  ", p)
    print("\nNow you can run the embedding script in OFFLINE mode.")
