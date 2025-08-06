import os
import json
import hashlib

CACHE_DIR = "cache"

def get_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def get_cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, get_url_hash(url))

def is_cached_url(url: str) -> bool:
    return os.path.exists(os.path.join(get_cache_path(url), "index"))

def save_to_cache(url: str, text: str, metadata: dict, vector_store):
    path = get_cache_path(url)
    os.makedirs(path, exist_ok=True)
    
    # Save FAISS index
    vector_store.save_local(os.path.join(path, "index"))

    # Save raw text
    with open(os.path.join(path, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    # Save metadata
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def load_from_cache(url: str, embeddings):
    from langchain_community.vectorstores import FAISS
    path = os.path.join(get_cache_path(url), "index")
    return FAISS.load_local(path, embeddings)
