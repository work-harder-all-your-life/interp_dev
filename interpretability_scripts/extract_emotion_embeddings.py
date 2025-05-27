import argparse
import os
from utils import (
    get_audio_path,
    assign_emotion_labels,
    save_to_npy,
    save_to_chromadb
)

import torch
from tqdm import tqdm
import wespeaker


def extract_embeddings(audio_files, device, pretrain_dir):
    """
    Extracts embeddings from audio files using the WeSpeaker model
    """
    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device(device)

    embeddings = []

    for file_path in tqdm(
        audio_files, desc="Embeddings computing process"
    ):
        embedding = model.extract_embedding(file_path)

        embedding = embedding.cpu().numpy()
        embeddings.append({
            "file_path": str(file_path),
            "embedding": embedding
        })

    return embeddings


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./emotion_classification/dataset/train",
                        help="Path to train audio files.")
    parser.add_argument("--test_dir", type=str, default="./emotion_classificaction/dataset/test",
                        help="Path to test audio files.")
    parser.add_argument("--pretrain_dir", type=str, default="./emotion_classification/pretrain_dir",
                        help="Path to wespeaker model pretrain_dir.")
    parser.add_argument("--output", type=str, required=True,
                        choices=["npy", "chromadb"],
                        help="Embeddings saving format: npy or chromadb.")
    parser.add_argument("--save_path", type=str, default="./emotion_classification",
                        help="Save path for calculated embeddings")
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_audio_files = get_audio_path(args.train_dir)
    test_audio_files = get_audio_path(args.test_dir)

    train_embeddings = extract_embeddings(train_audio_files, device,
                                          args.pretrain_dir)
    test_embeddings = extract_embeddings(test_audio_files, device,
                                         args.pretrain_dir)

    assign_emotion_labels(train_embeddings, args.train_dir, "train")
    assign_emotion_labels(test_embeddings, args.test_dir, "test")

    if args.output == "npy":
        os.makedirs(args.save_path, exist_ok=True)
        embeddings = [{"train": train_embeddings, "test": test_embeddings}]
        save_to_npy(embeddings, args.save_path)
    elif args.output == "chromadb":
        save_to_chromadb(train_embeddings, args.save_path, split="train")
        save_to_chromadb(test_embeddings, args.save_path, split="test")


if __name__ == '__main__':
    main()
