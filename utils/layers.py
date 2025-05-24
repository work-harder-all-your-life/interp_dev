from .extract_features import extract_features
import os
from pathlib import Path
import shutil

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from tqdm import tqdm


class GetActivations(nn.Module):
    """
    A class for obtaining activations of ResNet model intermediate layers.

    Provides preservation of the intermediate tensor (identity) for use in
    residual links and retrieves outputs from a given layer.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.saved_out = None
        self.identity_dir = Path("tmp_identity")
        self.identity_dir.mkdir(exist_ok=True)

    def save_identity(self, file_name):
        """
        Saves the current saved_out (identity-tensor) to a file.
        """
        file_path = self.identity_dir / file_name
        torch.save(self.saved_out, file_path)

    def delete_identity(self):
        """
        Deletes the temporary identity files directory, if it exists.
        Used to clean up after activation extraction is complete.
        """
        if os.path.exists("tmp_identity"):
            shutil.rmtree("tmp_identity")

    def _process_first_relu(self, x, model_front):
        """
        Applies the initial transformations (Conv + BN + ReLU) to the input
        and saves the result to self.saved_out.
        """
        out = x.permute(0, 2, 1).unsqueeze(dim=1)
        out = model_front.relu(model_front.bn1(model_front.conv1(out)))
        self.saved_out = out.clone()
        return out

    def _load_identity(self, identity_file, device):
        """
        Loads the saved identity-tensor from disk
        and saves it to self.saved_out.
        """
        path = self.identity_dir / identity_file
        if path.exists():
            self.saved_out = torch.load(path, map_location=device)

    def forward(
            self, x, target_layer, from_activation=False, identity_file=None
    ):
        """
        A forward pass through the model for the specified layer,
        with identity-tensors saved or loaded.
        """
        activations = {}
        model_front = self.model.model.front
        out = x

        if not from_activation:
            out = self._process_first_relu(x, model_front)

            if identity_file:
                self.save_identity(identity_file)
            if target_layer == "first relu":
                activations["first relu"] = out
                return activations, out
        else:
            if identity_file:
                self._load_identity(identity_file, x.device)
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
    Returns a list of names of all extracted SimAM ResNet model layers.
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


def get_activations(model, audio_files, device, chunk_num, layer, mode):
    """
    Retrieves activations of the specified layer for a set of audio files.

    Uses a pre-trained model and the GetActivations class.
    For each audio file, an identity-tensor is stored or loaded (if required).
    The resulting activations are saved to a list.
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
                identity_file=f"{mode}_identity_{chunk_num}_{i}.pt")
            activations.append(acts[layer].cpu())
    return activations, labels
