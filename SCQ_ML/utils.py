import torch
from torch_geometric.data import Data
from pathlib import Path

def write_chunk(chunk: list[Data], fn: Path):
    """
    Saves a list of Data objects to a file.
    
    Args:
        chunk: A list of PyG Data objects.
        fn: The file path to save the chunk to.
    """
    torch.save(chunk, fn) 