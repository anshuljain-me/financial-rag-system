import hashlib
from pathlib import Path

def compute_file_sha256(file_path: str | Path) -> str:
    """
    Computes a cryptographic SHA-256 hash of a file.
    Used as a content-addressable ID to prevent duplicate chunking & embedding costs.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536): # 64KB buffer
            sha256.update(chunk)
    return sha256.hexdigest()