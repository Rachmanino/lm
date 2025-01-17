#!/bin/sh
# prepare the env
conda install -f environment.yml

# download the LLaMA-Factory, which is used to finetune the model
git clone https://github.com/hiyouga/LLaMA-Factory.git  
pip install -r LLaMA-Factory/requirements.txt

# download the model and data, then process the data
python download.py
cp -f dataset_info.json LLaMA-Factory/data/dataset_info.json

# finetune the model in GUI
cd LLaMA-Factory
llamafactory-cli webui