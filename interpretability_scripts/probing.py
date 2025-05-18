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
from datasets.datasets import ActivationDataset
from models.models import ProbingCls
from models.train_models import train_probing_model
from pathlib import Path
import json

import numpy as np
from torch.utils.data import DataLoader
import torch
import wespeaker


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

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    acts_model = GetActivations(model)

    train_files = get_audio_path(args.train_dir)
    test_files = get_audio_path(args.test_dir)

    resume_file = "last_layer.json"
    resume_test_file = args.text_save_path

    layers = get_layers(model)

    if Path(resume_file).exists():
        with open(resume_file, "r") as f:
            resume_layer = json.load(f).get("last_layer")
    else:
        resume_layer = None

    if resume_layer is not None and resume_layer == layers[-1]:
        print(
            "Last training layer already completed. "
            "Skipping training and going to test."
        )
        layers = []
    elif resume_layer is not None:
        try:
            resume_index = layers.index(resume_layer)
            layers = layers[resume_index + 1:]
        except ValueError:
            raise ValueError(
                f"Resume layer {resume_layer} not found in model.")

    if layers:
        skf, file_paths, file_labels = prepare_chunks(
            train_files, args.chunk_size)

        for layer in layers:
            print(f"Processing layer: {layer}")
            probing_model = None
            for i, (_, chunk_idx) in enumerate(
                skf.split(file_paths, file_labels)
            ):
                chunk = file_paths[chunk_idx]
                if not os.path.exists(f"tmp/tmp_acts_{i}.pt"):
                    train_acts, train_labels = get_activations(
                        acts_model, chunk, device, i, layer)
                    train_dataset = ActivationDataset(train_acts, train_labels)
                    train_loader = DataLoader(
                        train_dataset, batch_size=32, shuffle=True)
                else:
                    train_acts = []
                    acts_list = torch.load(f"tmp/tmp_acts_{i}.pt")
                    train_labels = torch.load(f"tmp/tmp_labels_{i}.pt")
                    for num, act in enumerate(acts_list):
                        with torch.no_grad():
                            act = act.to(device)
                            acts, _ = acts_model(
                                act, layer, True,
                                identity_file=f"identity_{i}_{num}.pt")
                            train_acts.append(acts[layer])
                    train_dataset = ActivationDataset(train_acts, train_labels)
                    train_loader = DataLoader(
                        train_dataset, batch_size=32, shuffle=True)

                if probing_model is None:
                    probing_model = train_probing_model(
                        train_loader,
                        input_dim=train_dataset.audio_data.shape[-1],
                        device=device
                    )
                else:
                    probing_model = train_probing_model(
                        train_loader,
                        input_dim=train_dataset.audio_data.shape[-1],
                        device=device,
                        existing_model=probing_model
                    )
                save_tmp(train_acts, f"tmp_acts_{i}.pt")
                save_tmp(train_labels, f"tmp_labels_{i}.pt")
            del train_acts, train_labels, train_dataset, train_loader
            torch.save(probing_model.state_dict(), f"./models/{layer}.pth")
            with open(resume_file, "w") as f:
                json.dump({"last_layer": layer}, f)

    acts_model.delete_identity()
    delete_tmp()

    print("Testing")
    layers = get_layers(model)
    resume_test = resume_test_layer(resume_test_file)
    if resume_test is not None:
        if resume_test == layers[-1]:
            print(
                "Last test layer already evaluated. Exiting and visualizing."
            )
            plot_metrics(read_metrics(args.text_save_path),
                         args.visual_save_path)
            return
        try:
            resume_index = layers.index(resume_test)
            layers = layers[resume_index + 1:]
        except ValueError:
            raise ValueError(f"Resume test layer {resume_test} not found.")

    skf, file_paths, file_labels = prepare_chunks(test_files, args.chunk_size)

    for layer in layers:
        print(f"Processing layer: {layer}")
        all_preds = []
        all_labels = []

        for i, (_, chunk_idx) in enumerate(skf.split(file_paths, file_labels)):
            chunk = file_paths[chunk_idx]
            if not os.path.exists(f"tmp/tmp_acts_{i}.pt"):
                test_acts, test_labels = get_activations(
                    acts_model, chunk, device, i, layer)
                dataset = ActivationDataset(test_acts, test_labels)
                loader = DataLoader(dataset, batch_size=32, shuffle=False)
            else:
                test_acts = []
                acts_list = torch.load(f"tmp/tmp_acts_{i}.pt")
                test_labels = torch.load(f"tmp/tmp_labels_{i}.pt")
                for num, act in enumerate(acts_list):
                    with torch.no_grad():
                        act = act.to(device)
                        acts, _ = acts_model(
                            act, layer, True,
                            identity_file=f"identity_{i}_{num}.pt"
                        )
                        test_acts.append(acts[layer])

                dataset = ActivationDataset(test_acts, test_labels)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

            save_tmp(test_acts, f"tmp_acts_{i}.pt")
            save_tmp(test_labels, f"tmp_labels_{i}.pt")
            probing_model = ProbingCls(dataset.audio_data.shape[-1]).to(device)
            probing_model.load_state_dict(torch.load(
                f"./models/{layer}.pth", weights_only=True))
            probing_model.eval()

            y_pred_chunk, y_true_chunk = [], []
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(device)
                    outputs = probing_model(X_batch).cpu()
                    y_pred_chunk.extend(outputs.numpy())
                    y_true_chunk.extend(y_batch.numpy())

            all_preds.extend(y_pred_chunk)
            all_labels.extend(y_true_chunk)

            del test_acts, test_labels, dataset, loader, probing_model
            torch.cuda.empty_cache()

        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
        metrics = evaluate_probing(layer, y_pred, y_true)

        save_metrics([metrics], args.text_save_path)
        save_to_csv([metrics], args.csv_save_path)

    acts_model.delete_identity()
    delete_tmp()

    plot_metrics(read_metrics(args.text_save_path), args.visual_save_path)


if __name__ == '__main__':
    main()
