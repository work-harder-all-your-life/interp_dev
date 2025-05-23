import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import torch
from tqdm import tqdm


def evaluate_emb_model(model, test_loader, device):
    """
    Evaluates a model on a test dataset. Calculates accuracy,
    precision, recall and f1-score
    """
    model.eval()
    total_samples_test = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for embeddings_batch, labels_batch in tqdm(
                test_loader, desc="Evaluation Progress"):
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.long()
            x1, outputs = model(embeddings_batch)

            total_samples_test += 1

            _, predicted = torch.max(outputs.cpu(), 1)
            true_labels.extend(labels_batch.numpy())
            pred_labels.extend(predicted.numpy())

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1_score": f1_score(true_labels, pred_labels)
    }

    return metrics


def evaluate_probing(layer, y_pred, y_true):
    y_pred_labels = (y_pred >= 0.5).astype(int).squeeze(1)
    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    return (layer, {"accuracy": acc, "f1_score": f1})


def read_metrics(file_path):
    metrics_list = []
    current_layer = None
    current_metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_layer, current_metrics = None, {}
                continue
            if ':' not in line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_metrics = {}
                current_layer = line
            else:
                key, value = line.split(':', 1)
                current_metrics[key.strip()] = float(value.strip())
    if current_layer and current_metrics:
        metrics_list.append((current_layer, current_metrics))
    return metrics_list


def plot_metrics(metrics_list, save_path):
    layers = [m[0] for m in metrics_list]
    accuracies = [m[1]["accuracy"] for m in metrics_list]
    f1_scores = [m[1]["f1_score"] for m in metrics_list]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(layers, accuracies, color='b', label="Accuracy")
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across layers")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(layers, f1_scores, color='g', label="F1-score")
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("Layers")
    plt.ylabel("F1-score")
    plt.title("F1-score across layers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def save_metrics(metrics_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a') as f:
        for layer, metrics in metrics_list:
            f.write(f"{layer}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


def save_to_csv(chunk_rows, layer, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_new = pd.DataFrame(chunk_rows)
    prediction_col = f"prediction_{layer}"

    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)

        if prediction_col not in df_existing.columns:
            df_existing[prediction_col] = np.nan

        for _, row in df_new.iterrows():
            filename = row["filename"]
            true_label = row["true_label"]
            prediction = row[prediction_col]

            if filename in df_existing["filename"].values:
                idx = df_existing[df_existing["filename"] == filename].index[0]

                if pd.isna(df_existing.at[idx, "true_label"]):
                    df_existing.at[idx, "true_label"] = true_label

                if pd.isna(df_existing.at[idx, prediction_col]):
                    df_existing.at[idx, prediction_col] = prediction
            else:
                new_row = {col: np.nan for col in df_existing.columns}
                new_row["filename"] = filename
                new_row["true_label"] = true_label
                new_row[prediction_col] = prediction
                df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)

        df_existing.to_csv(save_path, index=False)
    else:
        df_new.to_csv(save_path, index=False)
