import argparse
import os
from utils.file_ops import get_audio_path, prepare_chunks, save_tmp, delete_tmp
from utils.layers import (
    GetActivations,
    get_activations,
    get_layers,
    resume_test_layer
)
from utils.metrics import (
    evaluate_probing,
    read_metrics,
    plot_metrics,
    save_metrics,
    save_to_csv
)
from datasets.activations_dataset import ActivationDataset
from models.probing_model.model import ProbingCls
from models.train_models import train_probing_model
from pathlib import Path
import json

import numpy as np
from torch.utils.data import DataLoader
import torch
import wespeaker


def test_probing_model(layer, dataset, loader, device):
    model_path = f"./probing_models/{layer}.pth"
    probing_model = ProbingCls(dataset.audio_data.shape[-1]).to(device)
    probing_model.load_state_dict(torch.load(model_path, weights_only=True))
    probing_model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = probing_model(X_batch).cpu()
            y_pred.extend(outputs.numpy())
            y_true.extend(y_batch.numpy())

    return y_pred, y_true


def check_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder {path} does not exists.")


def get_remaining_layers(layers, resume_layer, done_message, on_done=None):
    if resume_layer is None:
        return layers
    if resume_layer == layers[-1]:
        print(done_message)
        if on_done:
            on_done()
        return []
    try:
        layers = layers[layers.index(resume_layer) + 1:]
    except ValueError:
        raise ValueError(
            f"Resume layer {resume_layer} not found in model.")
    return layers


def load_or_extract_acts(i, chunk, acts_model, device, layer, mode):
    acts_path = f"tmp/tmp_acts_{i}.pt"
    labels_path = f"tmp/tmp_labels_{i}.pt"

    if not os.path.exists(acts_path):
        activations, labels = get_activations(
            acts_model, chunk, device, i, layer)
    else:
        acts_list = torch.load(acts_path)
        labels = torch.load(labels_path)
        activations = []
        for num, act in enumerate(acts_list):
            with torch.no_grad():
                act = act.to(device)
                layer_acts, _ = acts_model(
                    act, layer, True, identity_file=f"identity_{i}_{num}.pt")
                activations.append(layer_acts[layer])

    dataset = ActivationDataset(activations, labels)

    return dataset, activations, labels


def train(model, acts_model, train_files, device, resume_file, args):
    layers = get_layers(model)
    message = "Last training layer already completed." \
        "Skipping training and going to test."

    if Path(resume_file).exists():
        with open(resume_file, "r") as f:
            resume_layer = json.load(f).get("last_layer")
    else:
        resume_layer = None
    layers = get_remaining_layers(layers, resume_layer,
                                  done_message=message)
    if not layers:
        return

    skf, file_paths, file_labels = prepare_chunks(
        train_files, args.chunk_size)

    for layer in layers:
        print(f"Processing layer: {layer}")
        probing_model = None

        for i, (_, chunk_idx) in enumerate(
                skf.split(file_paths, file_labels)
        ):
            chunk = file_paths[chunk_idx]
            dataset, activations, labels = load_or_extract_acts(
                i, chunk, acts_model, device, layer, mode="train")
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            probing_model = train_probing_model(
                loader,
                input_dim=dataset.audio_data.shape[-1],
                device=device,
                existing_model=probing_model
            )
            save_tmp(activations, f"tmp_acts_{i}.pt")
            save_tmp(labels, f"tmp_labels_{i}.pt")

        del activations, labels, dataset, loader
        torch.save(probing_model.state_dict(), f"./probing_models/{layer}.pth")
        with open(resume_file, "w") as f:
            json.dump({"last_layer": layer}, f)

    acts_model.delete_identity()
    delete_tmp()


def test(model, acts_model, test_files, device, args):
    resume_test_file = args.text_save_path
    message = "Last test layer already evaluated. Exiting and visualizing."
    layers = get_layers(model)
    resume_layer = resume_test_layer(resume_test_file)
    layers = get_remaining_layers(layers, resume_layer,
                                  done_message=message,
                                  on_done=lambda: plot_metrics(
                                      read_metrics(args.text_save_path),
                                      args.visual_save_path))
    if not layers:
        return

    skf, file_paths, file_labels = prepare_chunks(test_files, args.chunk_size)

    for layer in layers:
        print(f"Processing layer: {layer}")
        all_preds, all_labels = [], []
        all_filenames = []
        chunk_rows = []

        for i, (_, chunk_idx) in enumerate(skf.split(file_paths, file_labels)):
            chunk = file_paths[chunk_idx]
            dataset, activations, labels = load_or_extract_acts(
                i, chunk, acts_model, device, layer, mode="test")
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            save_tmp(activations, f"tmp_acts_{i}.pt")
            save_tmp(labels, f"tmp_labels_{i}.pt")

            y_pred_chunk, y_true_chunk = test_probing_model(
                layer, dataset, loader, device)

            for filepath, true_label, pred in zip(chunk, y_true_chunk,
                                                  y_pred_chunk):
                chunk_rows.append({
                    "filename": os.path.relpath(filepath, args.test_dir),
                    "true_label": true_label,
                    f"prediction_{layer}": int(pred > 0.5)
                })

            all_preds.extend(y_pred_chunk)
            all_labels.extend(y_true_chunk)
            all_filenames.extend(chunk)

            del activations, labels, dataset, loader
            torch.cuda.empty_cache()

        metrics = evaluate_probing(
            layer, np.array(all_preds), np.array(all_labels))

        save_metrics([metrics], args.text_save_path)
        save_to_csv(chunk_rows, layer, args.csv_save_path)

    acts_model.delete_identity()
    delete_tmp()
    plot_metrics(read_metrics(args.text_save_path), args.visual_save_path)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./train_audio",
        help="Path to train audio files."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_audio",
        help="Path to test audio files."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of file chunks per epoch."
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./result/probing.txt",
        help="Save path for text result."
    )
    parser.add_argument(
        "--csv_save_path",
        type=str,
        default="./result/probing.csv",
        help="Save path for csv result."
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./result/probing.png",
        help="Save path for visual result."
    )
    args = parser.parse_args()

    check_paths(args.pretrain_dir, args.train_dir, args.test_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)
    acts_model = GetActivations(model)

    train_files = get_audio_path(args.train_dir)
    test_files = get_audio_path(args.test_dir)

    resume_file = "last_layer.json"

    train(model, acts_model, train_files, device, resume_file, args)
    print("Testing")
    test(model, acts_model, test_files, device, args)


if __name__ == '__main__':
    main()
