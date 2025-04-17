# HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading  

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)  

[[paper](https://arxiv.org/abs/2502.12574)]

## Overview  

**HeadInfer** is a memory-efficient inference framework for large language models (LLMs) that significantly reduces GPU memory consumption by leveraging a **head-wise offloading** strategy. Unlike traditional layer-wise KV cache offloading, **HeadInfer** dynamically manages attention heads, maintaining only a subset of the KV cache on the GPU while offloading the rest to CPU memory.  

With **HeadInfer**, an **8B model can process up to 4 million tokens on a single consumer-grade GPU** (e.g., RTX 4090 with 24GB VRAM), **reducing GPU KV cache memory from 128GB to just 1GB** without approximation.  

## Features  

- ✅ **Head-wise KV cache offloading**: Fine-grained memory optimization for long-context inference.  
- ✅ **Supports million-token inference**: Achieves up to **4M context length** on consumer GPUs.  
- ✅ **Asynchronous data transfer**: Overlaps computation with offloading to minimize bottlenecks.  
- ✅ **Compatible with major LLMs**: Works with LLaMA, Mistral, Qwen, and more.  
- ✅ **Minimal changes to existing inference frameworks**: Easy integration with Hugging Face models.  

## Installation  

#### Training and Evaluation Environment

```bash
conda create -yn headinfer python=3.10
conda activate headinfer

conda install -y git
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers==4.45.2 accelerate 
pip install flash-attn --no-build-isolation
```


## Usage  

One-click-run with HeadInfer

```bash
python main.py
```

Running Inference with HeadInfer
```python

from transformers import AutoModelForCausalLM, AutoTokenizer


from headinfer.cache import OffloadedCache
from headinfer.mp import mp_headinfer, mp_simulate_decode

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap the model with HeadInfer
headinfer_model = HeadInferModel(model)

# Generate text with long context
input_text = "Once upon a time in a galaxy far, far away..."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids


with torch.inference_mode():

    # patch the model
    mp_headinfer(model)
    past_key_values = OffloadedCache()

    model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)

```

## Citation
If you find HeadInfer useful for your research, please cite:

```bibtex
@article{luo2025headinfer,
  title={HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading},
  author={Luo, Cheng and Cai, Zefan and Sun, Hanshi and Xiao, Jinqi and Yuan, Bo and Xiao, Wen and Hu, Junjie and Zhao, Jiawei and Chen, Beidi and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2502.12574},
  year={2025}
}
```
