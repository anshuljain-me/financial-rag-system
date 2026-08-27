import hashlib
from pathlib import Path
from typing import Union

def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes the SHA-256 cryptographic hash of a file for exact deduplication.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hashing: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_sha256(file_path: Union[str, Path]) -> str:
    return compute_file_hash(file_path)

def hash_file(file_path: Union[str, Path]) -> str:
    return compute_file_hash(file_path)

class FileHasher:
    @staticmethod
    def compute_hash(file_path: Union[str, Path]) -> str:
        return compute_file_hash(file_path)
    
    @staticmethod
    def compute_file_hash(file_path: Union[str, Path]) -> str:
        return compute_file_hash(file_path)
