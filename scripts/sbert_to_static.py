# this is a simple script to learn static models with model2vec
import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tokenlearn.train import train_model
from tokenlearn.utils import collect_means_and_texts
from tokenlearn.featurize import featurize
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from model2vec.distill import distill
from model2vec import StaticModel

from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv('.env')
TOKEN = os.environ['HF_TOKEN']

def main():
    parser = argparse.ArgumentParser(description="Train static models with model2vec")
    parser.add_argument("--dataset", required=False, type=str, help="Dataset name to load")
    parser.add_argument("--model", required=True, help="SentenceTransformer model name")
    parser.add_argument("--split", required=False, help="Which split of the dataset to be used for the featurization", default="train")
    parser.add_argument("--device", required=False, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--preptype", choices=["train", "distill"], default="distill")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--featurize", action="store_true")
    parser.add_argument("--pca_dims", type=int, default=384)
    args = parser.parse_args()

    if args.test:
        model = StaticModel.from_pretrained(args.model)

        phrases_test = ['De patient heeft gele koorts',
                            'De patient heeft koorts',
                            'De patient heeft geelzucht',
                            'De voetbreuk is geheeld',
                            'De voetbreuk is niet hersteld']

        embeddings = model.encode(phrases_test)

        distances = cosine_similarity(embeddings)
        print(tabulate(distances, headers=phrases_test, tablefmt='psql', floatfmt='.4f'))


    else:
        # asset args.dataset exists
        assert(args.dataset), "Dataset cannot be empty"
        st_model = SentenceTransformer(args.model)
        output_dir = "output"
        if args.preptype=='train':
            if args.featurize:
                my_corpus = load_dataset(args.dataset, split="train", streaming=True, token=TOKEN)
                featurize(
                    dataset=my_corpus,
                    model=st_model,
                    output_dir=output_dir,
                    max_means=2_000_000,
                    batch_size=128,
                    text_key="text"
                )

            data_dir = output_dir

            # Collect paths for training data
            paths = sorted(Path(data_dir).glob("*.json"))
            train_txt, train_vec = collect_means_and_texts(paths)

            static_model_init = distill(model_name=args.model, pca_dims=args.pca_dims, trust_remote_code=False)

            model = train_model(
                static_model_init,
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
