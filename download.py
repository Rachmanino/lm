"""
    This script is the same as `download.ipynb`.
"""

from transformers import AutoModelForCausalLM
from datasets import load_dataset

# download the base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# download the dataset
dataset = load_dataset("pookie3000/trump-interviews", num_proc=4)

# create a dataset in the format required by LlamaFactory
import json
dataset_path = './LLaMA-Factory/data/trump.json'
data = [{'messages': msg} for msg in dataset['train']['conversations']]
with open(dataset_path, 'w') as f:
    json.dump(data, f)