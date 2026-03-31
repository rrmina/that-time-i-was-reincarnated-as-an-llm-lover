import pickle
from pathlib import Path

def load_pickle(
    path: Path
) -> object:
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(
    obj, 
    path: Path
) -> None:
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)