import os
import json
import hashlib
import os
from ..Utils.downloader import DocumentDownloader
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # => HackRx3/
CACHE_DIR = os.path.join(BASE_DIR, "cache")

def get_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def get_cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, get_url_hash(url))

def is_cached_url(url: str) -> bool:
    try:
        # Download document temporarily
        downloader = DocumentDownloader()
        temp_path, _ = downloader.download_from_url(url)
        
        # Hash its contents
        file_hash = get_file_hash(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)

        # Check if cache exists for this hash
        return os.path.exists(os.path.join(CACHE_DIR, file_hash, "index"))
    except Exception as e:
        return False

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

def get_file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def is_cached_file(filepath: str) -> bool:
    file_hash = get_file_hash(filepath)
    return os.path.exists(os.path.join(CACHE_DIR, file_hash, "index"))

def get_cache_path_for_file(filepath: str) -> str:
    return os.path.join(CACHE_DIR, get_file_hash(filepath))

def get_file_hash_from_url(url: str, downloaded_path: str = None) -> str:
    """
    Always use content hash if file path is available; fallback to URL hash only if file isn't accessible.
    """
    if downloaded_path and os.path.exists(downloaded_path):
        try:
            with open(downloaded_path, "rb") as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            pass  # fallback if file can't be read

    # fallback: hash the URL (non-deterministic if content changes)
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

