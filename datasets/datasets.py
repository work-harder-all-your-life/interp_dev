from base_dataset import BaseDataset
import os

import chromadb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch


class EmbeddingsDataset(Dataset):
    """
    Dataset class for loading embeddings
    """
    def __init__(
            self,
            source_path,
            split,
            source_type,
            collection_name="gender_embeddings"):
        self.lb = LabelEncoder()

        if source_type == "npy":
            self.embeddings, self.labels = self.get_npy_embeddings(
                source_path, split)
        elif source_type == "chromadb":
            self.embeddings, self.labels = self.get_chroma_embeddings(
                source_path, split, collection_name)
        else:
            raise ValueError(
                f"Invalid source type: {source_type}. "
                "Choose 'npy' or 'chromadb'."
            )

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def get_npy_embeddings(self, source_path, split):
        """
        Reads embddings from a .npy file
        """
        source = np.load(os.path.join(
            source_path, "numpy_embs.npy"), allow_pickle=True)
        source = source[0]

        if split == "train":
            embeddings = np.array([item['embedding']
                                  for item in source['train']])
            labels = [item['label'] for item in source['train']]
        elif split == "test":
            embeddings = np.array([item['embedding']
                                  for item in source['test']])
            labels = [item['label'] for item in source['test']]
        else:
            raise ValueError(
                f"Invalid split. Expected 'test' or 'train', got {split}")
        labels = self.lb.fit_transform(labels)
        return embeddings, labels

    def get_chroma_embeddings(
            self,
            source_path,
            split,
            collection_name="gender_embeddings"):
        """
        Reads embeddings from ChromaDB
        """
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={"split": split}, include=[
            "embeddings", "metadatas"])
        embeddings = np.array(results['embeddings'], dtype=np.float32)
        labels = [item['label'] for item in results['metadatas']]

        labels = self.lb.fit_transform(labels)
        return embeddings, labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)


class ActivationDataset(BaseDataset):
    def __init__(self, activations, labels):
        self.audio_data = self.prepare_data(activations)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def prepare_data(self, activations):
        activations = [act.clone() for act in activations]
        max_len = max(act.shape[-1] for act in activations)

        for i in range(len(activations)):
            pad_size = max_len - activations[i].shape[-1]
            activations[i] = torch.nn.functional.pad(
                activations[i], (0, pad_size), value=0.0)
            if len(activations[i].shape) != 2:
                activations[i] = activations[i].view(
                    activations[i].size(0), -1)

        return torch.stack(activations).squeeze(1)
