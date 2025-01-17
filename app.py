from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

LOCAL = True

def load_llama2():
    if LOCAL:
        return (
            AutoModelForCausalLM.from_pretrained("/root/autodl-fs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
                                                torch_dtype=torch.float16, 
                                                device_map="cuda"),
            AutoTokenizer.from_pretrained("/root/autodl-fs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590")
        )

    # REMOTE
    return (
        AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              torch_dtype=torch.float16,
                                              device_map="cuda"),
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    )

def load_llama2_trump():
    if LOCAL:
        return (
            AutoModelForCausalLM.from_pretrained("/root/autodl-fs/llama2-trump",
                                                torch_dtype=torch.float16, 
                                                device_map="cuda"),
            AutoTokenizer.from_pretrained("/root/autodl-fs/llama2-trump")
        )

    # REMOTE
    return (
        AutoModelForCausalLM.from_pretrained("rachmanino/Llama-2-7B-chat-Trump-v1",
                                             torch_dtype=torch.float16, 
                                             device_map="cuda"),
        AutoTokenizer.from_pretrained("rachmanino/Llama-2-7B-chat-Trump-v1")
    )

MODEL2FN = {
    # "GPT-2": load_gpt2,
    "Llama-2-chat": load_llama2,
    "Llama-2-Trump-chat": load_llama2_trump
}

caches = [
    'No KVCache',
    'Enable KVCache',
    'H2O KVCache'
]
current_model = None
model = None
tokenizer = None
enable_rag: bool = False
db = None
k = 5

def load(model_name):
    global current_model, model, tokenizer
    if current_model != model_name:
        if current_model is not None:
            del model
        current_model = model_name
        
        model, tokenizer = MODEL2FN[model_name]()
        gr.Info(f"Successfully loaded {model_name}!")

def augment_prompt(prompt: str) -> str:
    global db, k
    if not enable_rag:
        raise gr.Error("RAG is not enabled!")
    entries = db.similarity_search(prompt, top_k=k) # 
    augmented_prompt = 'Suppose you are a helpful assistant. You are given the following data:\n\n'
    for entry in entries:
        augmented_prompt += entry.page_content + '\n\n'
    augmented_prompt += f'Based on the data, answer following prompt: {prompt}'
    return augmented_prompt
    

def qa(model_name: str, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, cache: str) -> str:
    global current_model, model, tokenizer, enable_rag, augmented_prompt    
    load(model_name)
            
    if enable_rag:
        prompt = augment_prompt(prompt)
    chat = [{"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    outputs = model.generate(tokenized_chat, 
                             max_new_tokens=max_new_tokens, 
                             do_sample=do_sample, 
                             temperature=temperature, 
                             use_cache=True if cache=="Enable KVCache" else False)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ans.split(" [/INST] ")[-1]

itf1 = gr.Interface(
    fn=qa, 
    inputs=[
        gr.Dropdown(MODEL2FN.keys(), label="Model"),
        gr.Textbox(lines=3, placeholder="Your input...",label="Prompt"),
        gr.Number(label="Max New Tokens"),
        gr.Checkbox(label="Sampling"),
        gr.Slider(0.01, 2, step=0.01, label="Temperature(requires sampling enabled)"),
        gr.Dropdown(caches, label="KVCache")
    ],
    outputs=gr.Textbox(label="Response"))

def chat(message, history, model_name):
    global current_model, model, tokenizer, enable_rag, augmented_prompt    
    load(model_name)

    chat = []
    if enable_rag:
        message = augment_prompt(message)
    for qa in history:
        assert len(qa) == 2
        chat.append({"role": "user", "content": qa[0]})
        chat.append({"role": "assistant", "content": qa[1]})
    chat.append({"role": "user", "content": message})
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    outputs = model.generate(tokenized_chat)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(ans)
    return ans.split(" [/INST] ")[-1]

with gr.Blocks() as itf2:
    model = gr.Dropdown(MODEL2FN.keys(), label="Model")
    gr.ChatInterface(chat, additional_inputs=model)


def rag(use_rag: bool, path: str, chunk_size: int, chunk_overlap: int, topk: int) -> str:
    global current_model, model, tokenizer, enable_rag, db, k  
    print(use_rag, path, chunk_size, chunk_overlap)
    print(enable_rag)
    if enable_rag != use_rag:
        enable_rag = use_rag
        if enable_rag:
            if chunk_size <= chunk_overlap:
                raise gr.Error("Chunk size must be greater than chunk overlap!")
            k = topk
            loader = TextLoader(path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                           chunk_overlap=chunk_overlap) 
            docs = text_splitter.split_documents(pages)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")
            db = FAISS.from_documents(docs, embeddings)
            return f"Created {len(docs)} entries...RAG is enabled!"
        else:
            db = None
            return "RAG is disabled"
    
itf3 = gr.Interface(
    fn=rag, 
    inputs=[
        gr.Checkbox(label="Enable RAG"),
        gr.File(label="File for RAG(requires RAG enabled)"),
        gr.Slider(label="Chunk Size(requires RAG enabled)", minimum=2, maximum=1000, step=1),
        gr.Slider(label="Chunk Overlap(requires RAG enabled)", minimum=1, maximum=1000, step=1),
        gr.Slider(label="Top K(requires RAG enabled)", minimum=1, maximum=10, step=1)
    ],
    outputs=gr.Textbox(label="Result"),
    submit_btn='Set'
)


chatbot = gr.TabbedInterface([itf1, itf2, itf3], tab_names=["QA", "Chat", "Build RAG"], 
                          title="Mini Chatbot", 
                          theme='soft') # can switch

if __name__ == "__main__":
    chatbot.launch()
