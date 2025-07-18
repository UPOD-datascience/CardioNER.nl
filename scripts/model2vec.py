# this is a simple script to learn static models with model2vec
import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tokenlearn.train import train_model
from tokenlearn.utils import collect_means_and_texts
from tokenlearn.featurize import featurize
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train static models with model2vec")
    parser.add_argument("--dataset", required=True, help="Dataset name to load")
    parser.add_argument("--model", required=True, help="SentenceTransformer model name")
    parser.add_argument("--device", required=False, choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    my_corpus = load_dataset(args.dataset, split="train", streaming=True)
    model = SentenceTransformer(args.model)
    output_dir = "output"

    featurize(
        dataset=my_corpus,
        model=model,
        output_dir=output_dir,
        max_means=2_000_000,
        batch_size=32,
        text_key="text"
    )

    data_dir = output_dir

    # Collect paths for training data
    paths = sorted(Path(data_dir).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths)

    model = train_model(
        args.model,
        train_txt,
        train_vec,
        device=args.device,
        pca_dims=512
    )

if __name__ == "__main__":
    main()
