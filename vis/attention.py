import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse 
import numpy as np
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
args = argparser.parse_args()


model_name_or_path = args.model_name_or_path
dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='cpu',
    token = 'xxx'
)


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


word_to_attention_list = {}

htmls = []
color = 'red'

def attention(text):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    tokens_text = tokenizer.convert_ids_to_tokens(token_ids[0])
    # get_attention for next token
    outputs = model(
        input_ids = token_ids,
        output_attentions = True
    )
    attentions = outputs.attentions
    # last layer attention for each token_word:
    # Get the attention from the last layer (we assume the last attention layer is at index -1)
    last_layer_attention = attentions[-1]  # shape: (batch_size, num_heads, seq_len, seq_len)


    attention_scores = last_layer_attention[0]  # shape: (num_heads, seq_len, seq_len)
    num_heads = attention_scores.shape[0]
    # softmax the attention scores
    attention_scores = attention_scores.norm(dim=0)

    # divide by sqrt(num_heads)
    attention_scores = attention_scores / np.sqrt(num_heads)

    attention_scores = attention_scores.cpu().detach().numpy()



    return attention_scores, tokens_text


    