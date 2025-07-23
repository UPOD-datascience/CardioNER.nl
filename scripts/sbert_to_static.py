# this is a simple script to learn static models with model2vec
import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tokenlearn.train import train_model
from tokenlearn.utils import collect_means_and_texts
from tokenlearn.featurize import featurize

from model2vec.distill import distill

from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv('.env')
TOKEN = os.environ['HF_TOKEN']

def main():
    parser = argparse.ArgumentParser(description="Train static models with model2vec")
    parser.add_argument("--dataset", required=True, help="Dataset name to load")
    parser.add_argument("--model", required=True, help="SentenceTransformer model name")
    parser.add_argument("--split", required=False, help="Which split of the dataset to be used for the featurization", default="train")
    parser.add_argument("--device", required=False, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--preptype", choices=["train", "distill"], default="distill")
    parser.add_argument("--pca_dims", type=int, default=256)
    args = parser.parse_args()


    model = SentenceTransformer(args.model)
    output_dir = "output"

    if args.preptype=='train':
        my_corpus = load_dataset(args.dataset, split="train", streaming=True, token=TOKEN)
        featurize(
            dataset=my_corpus,
            model=model,
            output_dir=output_dir,
            max_means=2_000_000,
            batch_size=128,
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
            pca_dims=args.pca_dims
        )
    elif args.preptype=='distill':
        model = distill(model_name=args.model, pca_dims=args.pca_dims, trust_remote_code=False)
    else:
        raise ValueError("Invalid preptype")

    # save model
    model.save_pretrained("output/static_model")

if __name__ == "__main__":
    main()
