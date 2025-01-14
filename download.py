import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download


def download_ckpt():
    local_dir = "ckpt/"
    if os.path.exists(local_dir):
        return local_dir
    os.makedirs(local_dir)
    snapshot_download("VITA-MLLM/VITA-1.5", local_dir=local_dir)
    return local_dir


if __name__ == "__main__":
    download_ckpt()
