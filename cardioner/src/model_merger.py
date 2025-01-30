"""
 Model merger

 Given a list of folder with transformer models

 Accumulate the model weights and average


[Example](https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008)
```
Example:

modelA = nn.Linear(1, 1)
modelB = nn.Linear(1, 1)

sdA = modelA.state_dict()
sdB = modelB.state_dict()

# Average all parameters
for key in sdA:
    sdB[key] = (sdB[key] + sdA[key]) / 2.

# Recreate model and load averaged state_dict (or use modelA/B)
model = nn.Linear(1, 1)
model.load_state_dict(sdB)
```

Input:
- main_model_folder with
 - model.bin's
 - model folders
- output_dir

Output:
- averaged_model
"""

from os import environ
import os
import json
import argparse
import torch
import transformers

def model_averager(model_locations):
    # Load the first model to get initial weights
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_locations[0])
    averaged_weights = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    num_models = len(model_locations)

    # Iterate through each model, accumulate weights
    for location in model_locations:
        print(f"Processing model at {location}")
        model = transformers.AutoModelForTokenClassification.from_pretrained(location)
        for name, param in model.named_parameters():
            averaged_weights[name] += param.data

    # Perform averaging based on the number of models
    for name, param in averaged_weights.items():
        averaged_weights[name] /= num_models

    # Load a new model (or reuse the first model) and update weights
    model_averaged = transformers.AutoModelForTokenClassification.from_pretrained(model_locations[0])
    model_averaged.load_state_dict(averaged_weights, strict=False)
    
    # Handle the heads separately if needed
    head_averaged_weights = {}
    for location in model_locations:
        model = transformers.AutoModelForTokenClassification.from_pretrained(location)
        for name, param in model.classifier.named_parameters():
            if name not in head_averaged_weights:
                head_averaged_weights[name] = torch.zeros_like(param)
            head_averaged_weights[name] += param.data

    for name, param in head_averaged_weights.items():
        head_averaged_weights[name] /= num_models

    # Update the head weights
    model_averaged.classifier.load_state_dict(head_averaged_weights, 
      strict=False)
    
    return model_averaged

def path_parser(models_dir):
    """
    Parses a given directory to find all model paths.

    Args:
      models_dir: The directory containing the model files or folders.

    Returns:
      A list of paths to all the models found within the directory.
    """
    model_paths = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if (('model' in file) and (file.endswith('.bin') or file.endswith('.safetensors'))) or os.path.isdir(os.path.join(root, file)):
                model_paths.append(os.path.join(root))
                continue
    return model_paths

argparser = argparse.ArgumentParser()

argparser.add_argument('--models_dir', type=str, required=True)
argparser.add_argument('--output_dir', type=str, required=True)
argparser.add_argument('--averaging', type=str, choices=['arithmetic', 'harmonic'], default='arithmetic')

args = argparser.parse_args()

list_of_model_locations = path_parser(args.models_dir)
print("Models to merge..")
print(list_of_model_locations)

model_averaged = model_averager(list_of_model_locations)

os.makedirs(args.output_dir, exist_ok=True)
model_averaged.save_pretrained(args.output_dir)
