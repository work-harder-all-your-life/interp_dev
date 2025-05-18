from extract_features import extract_features
import os
from pathlib import Path
import shutil


from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from tqdm import tqdm


class GetActivations(nn.Module):
    """
    Class for getting activations from a model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.saved_out = None

    def save_identity(self, file_name):
        folder = Path("tmp_identity")
        folder.mkdir(exist_ok=True)

        file_path = folder / file_name
        torch.save(self.saved_out, file_path)

    def delete_identity(self):
        if os.path.exists("tmp_identity"):
            shutil.rmtree("tmp_identity")

    def forward(
            self, x, target_layer, from_activation=False, identity_file=None
    ):
        activations = {}
        model_front = self.model.model.front
        out = x
        if not from_activation:
            out = x.permute(0, 2, 1).unsqueeze(dim=1)
            out = model_front.relu(model_front.bn1(model_front.conv1(out)))
            self.saved_out = out.clone()
            if identity_file:
                self.save_identity(identity_file)
            if "first relu" == target_layer:
                activations["first relu"] = out
                return activations, out
        elif from_activation:
            if identity_file and os.path.exists(
                "tmp_identity/" + identity_file
            ):
                self.saved_out = torch.load(
                    "tmp_identity/" + identity_file, map_location=x.device)
            out = x

        for name, layer in model_front.named_children():
            c_sim = 0
            c_relu = 0

            for block_idx, block in layer.named_children():
                identity = self.saved_out

                c_relu += 1
                if f"{name} relu {c_relu}" == target_layer:
                    out = block.relu(block.bn1(block.conv1(out)))
                    activations[f"{name} relu {c_relu}"] = out
                    return activations, out

                c_sim += 1
                if f"{name} SimAM {c_sim}" == target_layer:
                    out = block.bn2(block.conv2(out))
                    out = block.SimAM(out)
                    activations[f"{name} SimAM {c_sim}"] = out
                    return activations, out

                c_relu += 1
                if f"{name} relu {c_relu}" == target_layer:
                    if block.downsample is not None:
                        identity = block.downsample(identity)
                    out += identity
                    out = block.relu(out)
                    self.saved_out = out.clone()
                    if identity_file:
                        self.save_identity(identity_file)
                    activations[f"{name} relu {c_relu}"] = out
                    return activations, out

        if "pooling" == target_layer:
            out = self.model.model.pooling(out)
            activations["pooling"] = out
            return activations, out

        if self.model.model.drop:
            out = self.model.model.drop(out)

        return activations, out


def get_layers(model):
    """
    Returns SimAM ResNet's layers
    """
    layers = []

    model_front = model.model.front
    layers.append("first relu")

    for name, layer in model_front.named_children():
        c_relu = 0
        c_sim = 0
        if name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for sec_name, sec_layer in layer.named_children():
                c_relu += 1
                layers.append(f"{name} relu {c_relu}")
                c_sim += 1
                layers.append(f"{name} SimAM {c_sim}")
                c_relu += 1
                layers.append(f"{name} relu {c_relu}")
    layers.append("pooling")

    return layers


def get_activations(model, audio_files, device, chunk_num, layer):
    """
    Gets model activations for a specified layer.
    """
    label_encoder = LabelEncoder()
    labels = [Path(f).parent.name for f in audio_files]
    labels = label_encoder.fit_transform(labels)

    activations = []
    with torch.no_grad():
        for i, audio_path in enumerate(tqdm(
            audio_files, desc="Extracting activations"
        )):
            feats = extract_features(audio_path).to(device)
            acts, _ = model(
                feats, layer,
                identity_file=f"identity_{chunk_num}_{i}.pt")
            activations.append(acts[layer].cpu())
    return activations, labels


def resume_test_layer(metrics_path):
    """
    Defines last test layer
    """
    if not Path(metrics_path).exists():
        return None
    with open(metrics_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in reversed(lines):
        if ':' not in line:
            return line
    return None
