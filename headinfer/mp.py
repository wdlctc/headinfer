import transformers
from transformers.models.llama.modeling_llama import (
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaSdpaAttention,
    LlamaFlashAttention2,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2FlashAttention2
)
import torch

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache, QuantizedCacheConfig, HQQQuantizedCache
from transformers.modeling_flash_attention_utils  import _flash_attention_forward


def LlamaAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) :
    output_attentions = False

    bsz, q_len, hd = hidden_states.size()
    chunk_size = hd // self.num_key_value_heads
    num_heads = self.num_heads // self.num_key_value_heads

    if not hasattr(self, 'q_proj_list'):
        self.q_proj_list = list((self.q_proj.weight.split(self.head_dim * num_heads, dim=0)))
        # self.q_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'k_proj_list'):
        self.k_proj_list = list((self.k_proj.weight.split(self.head_dim, dim=0)))
        # self.k_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'v_proj_list'):
        self.v_proj_list = list((self.v_proj.weight.split(self.head_dim, dim=0)))
        # self.v_proj.weight.data.storage().resize_(0)


    attn_output_list = [None for _ in range((self.num_key_value_heads))]
    
    for i in range(self.num_key_value_heads):
        bsz, q_len, hd = hidden_states.size()

        self.q_proj.weight.data = self.q_proj_list[i].data
        self.k_proj.weight.data = self.k_proj_list[i].data
        self.v_proj.weight.data = self.v_proj_list[i].data

        # print(hidden_states.shape, self.q_proj.weight.shape)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
    
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx + i, cache_kwargs)
    
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
    
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output_list[i] = attn_output
        
    attn_output = torch.cat(attn_output_list, dim=-1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def Qwen2Attention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) :
    output_attentions = False

    bsz, q_len, hd = hidden_states.size()
    chunk_size = hd // self.num_key_value_heads
    num_heads = self.num_heads // self.num_key_value_heads

    if not hasattr(self, 'q_proj_list'):
        self.q_proj_list = list((self.q_proj.weight.split(self.head_dim * num_heads, dim=0)))
        self.q_proj_bias_list = list((self.q_proj.bias.split(self.head_dim * num_heads, dim=0)))
        # self.q_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'k_proj_list'):
        self.k_proj_list = list((self.k_proj.weight.split(self.head_dim, dim=0)))
        self.k_proj_bias_list = list((self.k_proj.bias.split(self.head_dim, dim=0)))
        # self.k_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'v_proj_list'):
        self.v_proj_list = list((self.v_proj.weight.split(self.head_dim, dim=0)))
        self.v_proj_bias_list = list((self.v_proj.bias.split(self.head_dim, dim=0)))
        # self.v_proj.weight.data.storage().resize_(0)


    attn_output_list = [None for _ in range((self.num_key_value_heads))]
    
    for i in range(self.num_key_value_heads):
        bsz, q_len, hd = hidden_states.size()

        self.q_proj.weight.data = self.q_proj_list[i].data
        self.q_proj.bias.data = self.q_proj_bias_list[i].data
        self.k_proj.weight.data = self.k_proj_list[i].data
        self.k_proj.bias.data = self.k_proj_bias_list[i].data
        self.v_proj.weight.data = self.v_proj_list[i].data
        self.v_proj.bias.data = self.v_proj_bias_list[i].data

        # print(hidden_states.shape, self.q_proj.weight.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
    
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx + i, cache_kwargs)
    
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
    
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output_list[i] = attn_output
        
    attn_output = torch.cat(attn_output_list, dim=-1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def mp_headinfer(model):

    if isinstance(model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        
        layer_idx = 0
        for idx, layer in enumerate(model.model.layers):
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            module = layer.self_attn
            module.layer_idx = layer_idx
            layer_idx += module.num_key_value_heads
            module.head_group = 8
    elif isinstance(model, transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM):
        Qwen2FlashAttention2.forward = Qwen2Attention_fast_forward
        
        layer_idx = 0
        for idx, layer in enumerate(model.model.layers):
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            module = layer.self_attn
            module.layer_idx = layer_idx
            layer_idx += module.num_key_value_heads
            module.head_group = 4

    print("[OK] HeadInfer Patched Successfully.")


def mp_simulate_decode(model, start_length=1024000):
    past_key_values = OffloadedCache()
    config = model.config

    for idx, layer in enumerate(model.model.layers):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        module = layer.self_attn

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        # head_dim = config.hidden_size
        cache_shape = (1, 1, start_length, module.head_dim)
        key_states = torch.zeros(cache_shape, dtype=dtype, device=device)
        value_states = torch.zeros(cache_shape, dtype=dtype, device=device)

        for i in range(config.num_key_value_heads):
            cache_kwargs = {'full_head': None}
            key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx + i, cache_kwargs)

    print("[OK] Simulation of Decoding Ready to start.")
    return past_key_values