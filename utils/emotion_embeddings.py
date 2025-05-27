import os
import json
from datasets import EmotionEmbeddingsDataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch


def get_emotion_loaders(source_path, source_type):
    """
    Creates dataloaders for train and test files.
    """
    train_dataset = EmotionEmbeddingsDataset(
        source_path, split="train", source_type=source_type)
    test_dataset = EmotionEmbeddingsDataset(
        source_path, split="test", source_type=source_type)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (
        train_loader,
        test_loader,
        test_dataset,
        train_dataset.audio_data.shape[1]
    )


def assign_emotion_labels(embeddings, data_dir, split):
    """
    Assigns emotion labels from .jsonl file to classes.
    """
    metadata_file_name = f"{split}_metadata.jsonl"
    metadata_file_path = os.path.join(data_dir, metadata_file_name)

    if not os.path.exists(metadata_file_path):
        raise FileNotFoundError(f"File {metadata_file_path} does not exists.")
    
    metadata = {}
    with open(metadata_file_path, "r") as file:
        for line in file:
            entry = json.loads(line)
            audio_path = entry["audio_path"]
            file_name = os.path.basename(audio_path)
            emotion = entry["annotator_emo"]
            metadata[file_name] = emotion

    for emb in embeddings:
        current_file_name = os.path.basename(emb["file_path"])
        if current_file_name in metadata:
            emb["label"] = current_file_name
        else:
            raise ValueError(f"File {current_file_name} not found in {metadata_file_path}")


def save_emotion_emb_visualization(model, vectors, labels, save_path, device):
    """
    Saves embedding visualization in .png files.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1, predicted = model(vectors)

    reducer = TSNE(n_components=5, random_state=42)
    x1_reduced = reducer.fit_transform(x1.detach().cpu().numpy())

    unique_labels = list(set(labels))

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            x1_reduced[indices, 0],
            x1_reduced[indices, 1],
            label=f"Label: {label}",
            alpha=0.6
        )

    plt.title("Visualization of embeddings after first layer")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_emotion_emb_metrics(metrics, save_path):
    """
    Saves computed metrics in .txt file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
