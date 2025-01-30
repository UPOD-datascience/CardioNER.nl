"""
Push model to Huggingface

inputs
- model_card
- model_directory
- hf_data_organisation
- language
"""

from datasets import DatasetDict, Dataset
import argparse
from huggingface_hub import HfApi, DatasetCard, errors, ModelCard
from requests.exceptions import HTTPError
from huggingface_hub import add_collection_item, get_collection
import os
import dotenv
from typing import Optional


from dotenv import load_dotenv
import hf_config

load_dotenv('.env')

TOKEN = os.environ['HF_TOKEN']

# Example usage:
# python hf_dataset.py --organization "DT4H" --name "Example dataset"\
# --dataset_path data --name example_api --description 'This is an example dataset'\
#  --language es --license mit --token YOUR_TOKEN

def create_dataset_card(name, description, language, license, tags):
    """
    Gets main information and creates a dataset card using the template in config.py
    """
    text = hf_config.description_text_data(name, description, language, license, tags)
    # Using the Template
    card = DatasetCard(content=text)

    return card

def create_model_card(name, data_organisation, description, 
                      data_description, language, license, tags, mod_type, 
                      mod_target, base_model):
    """
    Gets main information and creates a dataset card using the template in config.py
    """
    if mod_target == 'sap':
        text = hf_config.description_text_model_sap(name, data_organisation, description,
                                                        data_description, language,
                                                         license, tags, mod_type)
    elif mod_target == 'ner':
        text = hf_config.description_text_model_ner(name, data_organisation, description,
                                                        data_description, language,
                                                         license, tags, mod_type, base_model)

    # Using the Template
    card = ModelCard(content=text)

    return card

def push_to_huggingface(repo_id, dataset_path, card, private):
    api = HfApi(token=TOKEN)
    print(f"Attempting to push to Repository {repo_id}. \nRepo type {hf_config.repo_type}\n Token {TOKEN}")
    try:
        # Check if repository exists by trying to fetch its info
        api.repo_info(repo_id=repo_id, repo_type=hf_config.repo_type)  # You can adjust repo_type if it's a dataset or space
        print(f"Repository '{repo_id}' already exists.")
    except HTTPError as e:
        if e.response.status_code == 404:
            # If repository does not exist, create it
            print(f"Repository '{repo_id}' does not exist. Creating...")
            api.create_repo(token=TOKEN, repo_id=repo_id, private=private, repo_type=hf_config.repo_type)
        else:
            if e.response.status_code == 409:
                print(f"Repository '{repo_id}' already exists. Continuing")
            else:
                raise e

    # Upload dataset files
    if dataset_path.endswith(".jsonl") | dataset_path.endswith(".json"):
        file_path = dataset_path
        dataset_path = os.path.dirname(dataset_path)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.relpath(file_path, dataset_path),
            repo_id=repo_id,
            repo_type=hf_config.repo_type,
            token=TOKEN,
        )
    else:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.relpath(file_path, dataset_path),
                    repo_id=repo_id,
                    repo_type=hf_config.repo_type,
                    token=TOKEN,
                )

    # Push dataset card
    card.push_to_hub(
                        repo_id,
                        token=TOKEN,
                        repo_type=hf_config.repo_type,
        )

# class HuggingFaceDatasetManager:
#     def __init__(self, dataset: Dataset):
#         self.dataset= dataset

#     def save_to_disk(self, path):
#         self.dataset.save_to_disk(path)

#     def push_to_hub(self, repo_name, token):
#         self.dataset.push_to_hub(repo_name, token=TOKEN)

def main():

    parser = argparse.ArgumentParser(description="Push dataset and dataset card to Hugging Face")
    parser.add_argument("--data_organization", default="DT4H", help="Organization to push the dataset to")
    parser.add_argument("--collection_organization", default="DT4H", help="Organization that owns the collection")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    # parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset files")
    parser.add_argument("--name", required=True, help="Name of the dataset")
    parser.add_argument("--description", required=True, help="Description of the model")
    parser.add_argument("--data_description", required=True, help="Description of the dataset")
    parser.add_argument("--language", required=True, help="Language of the dataset")
    parser.add_argument("--license", default="mit", choices=hf_config.licenses, help="License of the dataset")
    parser.add_argument("--mod_type", choices=["mean", "cls", "multilabel", "multiclass"], required=True)
    parser.add_argument("--mod_target", choices=["sap", "ner"], required=True)
    parser.add_argument("--base_model", type=str, help="Base model used for the finetuning")
    parser.add_argument("--tags", nargs="+", default=[], help="Tags for the dataset")

    args = parser.parse_args()

    assert ((args.mod_target=='ner') & (args.mod_type in ["multilabel", "multiclass"]) |\
            (args.mod_target=='sap') & (args.mod_type in ["mean", "cls"])), \
        "If target is NER then the mod_type MUST be multilabel/multiclass, if SAP then mean/cls"

    repo_id = args.name.replace(" ", "_").lower()
    repo_id = f"{args.data_organization}/{repo_id}" if args.data_organization else repo_id

    # Create dataset card
    card = create_model_card(args.name, args.data_organization, args.description, args.data_description,
                              args.language, args.license, args.tags, 
                              args.mod_type, args.mod_target, args.base_model)

    # Push dataset and card to Hugging Face
    push_to_huggingface(repo_id, args.dataset_path, card, private=args.private)

    # Add dataset to collection
    collection_id = f"{args.collection_organization}/{hf_config.collections[args.language]}"
    print(f"Adding to collection {collection_id}")
    add_collection_item(collection_id, item_id=repo_id, item_type=hf_config.repo_type, exists_ok=True)

    if hf_config.repo_type == "model":
        repo_url = f"https://huggingface.co/models/{repo_id}"
        coll_url = f"https://huggingface.co/models/{collection_id}"

        print(f"Model and card successfully pushed to {repo_url}")
        print(f"Model successfully added to {coll_url}")


if __name__ == "__main__":
    main()
