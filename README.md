# 大模型原理 期末作业
## Prepare
```sh
bash prepare.sh
```
- Environment
```sh
conda install -f environment.yml
```
## Content
### Part 1 : Finetuning Llama with LLaMA-Factory
- We finetuned the base model (Llama-2-7b-chat)[https://huggingface.co/meta-llama/Llama-2-7b-chat-hf] on the [dataset](https://huggingface.co/datasets/pookie3000/trump-interviews) containing Trump's interviews. 
- Model is available at https://huggingface.co/rachmanino/Llama-2-7B-chat-Trump-v1

![demo1](assets/812b16fe24a449baaea685adab321f3.png812b16fe24a449baaea685adab321f3.png)

### Part 2: WebUI Chatbot built by Gradio
Usage:
```sh
python app.py
```
The Chatbot supports single-round `QA` and conversation with memory, i.e. `chat`.

### Part 3: Extending Llama-2-7B-chat's knowledge with RAG
Based on the WebUI implemented in Part 2, we further build RAG database to extend the model's knowledge.
