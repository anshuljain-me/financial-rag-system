import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 cryptographic hash of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
