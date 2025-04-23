import transformers
import torch
import time
import shutil
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from headinfer.cache import OffloadedCache
from headinfer.mp import mp_headinfer, mp_simulate_decode

# Load the 
# Qwen support: ckpt = "Qwen/Qwen2.5-7B-Instruct"
ckpt = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2").to("cuda")

generation_config = GenerationConfig.from_pretrained(ckpt)
eos_token_ids = generation_config.eos_token_id
if not isinstance(eos_token_ids, list):
    eos_token_ids = [eos_token_ids]
eos_token_ids += tokenizer.encode("</user>", add_special_tokens=False)
eos_token_ids += tokenizer.encode("</s>", add_special_tokens=False)
eos_token_ids += tokenizer.encode("</", add_special_tokens=False)


with torch.inference_mode():

    # patch the model
    mp_headinfer(model)


    ### Simulate Prefill
    start_length = 10240
    chunks = 4
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, start_length)).to('cuda')
    for epoch in range(10):
        past_key_values = OffloadedCache()
        past_key_values.max_cache_len = 0
        past_key_values._seen_tokens = 0
        past_key_values._next_seen_tokens = 0

        for i in range(chunks):
            
            torch.cuda.synchronize()
            start_time = time.time()

            model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)

            torch.cuda.synchronize()
            epoch_time = time.time() - start_time
            print(f"[Epoch {epoch+1}, Chunk {i+1}]   {epoch_time:.2f} seconds")

        del past_key_values
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    ### Simulate Decode
    start_length = 1024000
    past_key_values = mp_simulate_decode(model, start_length=start_length)


    max_new_tokens = 10
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 1)).to('cuda')

    for epoch in range(10):
        for i in range(max_new_tokens):
            torch.cuda.synchronize()
            start_time = time.time()
            
            model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)
            
            torch.cuda.synchronize()
            epoch_time = time.time() - start_time
            print(f"Epoch {i+1}  {epoch_time:.2f} seconds")
        start_length = start_length + 5120




