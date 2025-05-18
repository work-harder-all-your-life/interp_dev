from pathlib import Path
import os
import shutil

import chromadb
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_audio_path(audio_dir):
    audio_dir = Path(audio_dir)
    return list(audio_dir.glob('**/*.wav')) + list(audio_dir.glob('**/*.mp3'))


def prepare_chunks(train_files, chunk_size, random_state=42):
    """
    Splits the data into stratified chunks.
    """
    n_chunks = max(2, len(train_files) // chunk_size)
    file_paths = np.array(train_files)
    file_labels = np.array([Path(f).parent.name for f in train_files])

    skf = StratifiedKFold(n_splits=n_chunks, shuffle=True,
                          random_state=random_state)
    return skf, file_paths, file_labels


def save_to_npy(embeddings, save_dir):
    """
    Saves embeddings in .npy format.
    """
    numpy_embs = np.array(embeddings)
    np.save(os.path.join(save_dir, "34_ft.npy"), numpy_embs)


def save_to_chromadb(embeddings, db_path, split):
    """
    Stores embeddings in ChromaDB
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="gender_embeddings")

    collection.add(
        ids=[f"{split}_{i}" for i in range(len(embeddings))],
        embeddings=[item['embedding'] for item in embeddings],
        metadatas=[{
            "file_path": item['file_path'], "label": item['label'],
            "split": split
        }
            for item in embeddings]
    )


def save_tmp(data, file_name):
    folder = Path("tmp")
    folder.mkdir(exist_ok=True)
    torch.save(data, folder / file_name)


def delete_tmp():
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
