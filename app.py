from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# def load_gpt2():
#     return (
#         AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="cuda"),
#         AutoTokenizer.from_pretrained("openai-community/gpt2")
#     )

def load_llama2():
    return (
        AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              torch_dtype=torch.float16,
                                              device_map="cuda"),
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    )

def load_llama2_trump():

    # LOCAL
    return (
        AutoModelForCausalLM.from_pretrained("/root/autodl-fs/llama2-trump",
                                             torch_dtype=torch.float16, 
                                             device_map="cuda"),
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    )

    # REMOTE
    '''
    # return (
    #     AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-Trump-chat",
    #                                          torch_dtype=torch.float16, 
    #                                          device_map="cuda"),
    #     AutoTokenizer.from_pretrained("meta-llama/Llama-2-Trump-chat")
    # )
    '''

MODEL2FN = {
    # "GPT-2": load_gpt2,
    "Llama-2-chat": load_llama2,
    "Llama-2-Trump-chat": load_llama2_trump
}

caches = [
    'No KVCache',
    'Enable KVCache'
]
current_model = None
model = None
tokenizer = None

def load(model_name):
    global current_model, model, tokenizer
    if current_model != model_name:
        if current_model is not None:
            del model
        current_model = model_name
        model, tokenizer = MODEL2FN[model_name]()
        gr.Info(f"Successfully loaded {model_name}!")

def qa(model_name: str, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, cache: str) -> str:
    global current_model, model, tokenizer
    load(model_name)
            
    chat = [{"role": "user", "content": prompt},]
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    outputs = model.generate(tokenized_chat, 
                             max_new_tokens=max_new_tokens, 
                             do_sample=do_sample, 
                             temperature=temperature, 
                             use_cache=True if cache=="Enable KVCache" else False)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ans.split(" [/INST] ")[-1]

def chat(message, history, model_name):
    global current_model, model, tokenizer
    load(model_name)
    chat = []
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

itf1 = gr.Interface(
    fn=qa, 
    inputs=[
        gr.Dropdown(MODEL2FN.keys(), label="Model"),
        gr.Textbox(lines=3, placeholder="Your input...",label="Prompt"),
        gr.Number(label="Max New Tokens"),
        gr.Checkbox(label="Do sample"),
        gr.Slider(0.01, 2, step=0.01, label="Temperature(need do sample)"),
        gr.Dropdown(caches, label="KVCache")
    ],
    outputs=gr.Textbox(label="Response"))

itf2 = gr.ChatInterface(chat, 
                        additional_inputs=gr.Dropdown(MODEL2FN.keys(), label="Model"))

itf3 = NotImplemented #TODO: Implement QA with RAG

chatbot = gr.TabbedInterface([itf1, itf2, itf3], tab_names=["QA", "Chat", "QA with RAG"], 
                          title="Mini Chatbot", 
                          theme='soft') # can switch

if __name__ == "__main__":
    chatbot.launch()
