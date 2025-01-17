import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

__all__ = ['LlamaAttention_heavy_hitter_realdrop', 'convert_kvcache_llama_heavy_realdrop']

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
class LlamaAttention_heavy_hitter_realdrop(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        #self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        # 4.46.1 change
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        # 4.46.1 change
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _reset_masks(self):
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _update_previous_scores(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2] #1吧
        if self.previous_scores is None:
            # attn_score_cache (BS, heads, q-tokens, k-tokens) 16, 15, 1, 16
           self.previous_scores = attn_score_cache.sum(0).sum(1)  # (heads, saved-k-tokens)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.previous_scores
            self.previous_scores = attn_score_cache
            
    # 请补全此函数，实现 H2O算法的 real drop版本
    def KV_update(self, 
        past_key_value: Optional[Cache] = None, #change Cache
        attn_score_cache: Optional[torch.Tensor] = None,
        layer_idx: int,
        ) -> Tuple[torch.Tensor]:  

        self._update_previous_scores(attn_score_cache) 
        new_key_states, new_value_states = None, None

        if past_key_value is None: #不使用cache
            return None
        #seq_len = past_key_value[0].size(2) # old
        seq_len = past_key_value.key_cache[layer_idx].shape[-2]
        if seq_len <= self.cache_budget:
            return past_key_value

        bsz, num_heads, _, head_dim = past_key_value.key_cache[layer_idx].shape

        select_hh_scores = self.previous_scores[:, :seq_len - self.recent_budget] #取出从滑动窗口外的token
        _, keep_topk = torch.topk(select_hh_scores, self.heavy_budget, dim=-1)#选得分最高的k个
        keep_topk = keep_topk.sort().values

        keep_recent = torch.arange(seq_len - self.recent_budget, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1) 
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1) #两部分拼起来得到保留id

        mask = torch.zeros(self.previous_scores.shape, dtype=torch.bool).to(past_key_value.key_cache[layer_idx].device)
        mask = mask.scatter(-1, keep_idx, 1) #对-1轴，如果在idx里，mask为1，否则为0

        new_key_states = past_key_value.key_cache[layer_idx].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        new_value_states = past_key_value.value_cache[layer_idx].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        self.previous_scores= self.previous_scores[mask].view(num_heads, self.cache_budget) #过滤掉驱逐token的注意力分数累计
        return (new_key_states, new_value_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        #past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,

        # 4.46.1 change
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        #old
        # bsz, q_len, _ = hidden_states.size()
        # query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 4.46.1 change
        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1: #其实默认是1
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        #

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value.key_cache[layer_idx].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2] * self.num_key_value_groups
        if past_key_value is not None:
            kv_seq_len += past_key_value.key_cache[layer_idx].shape[-2] * self.num_key_value_groups
        
        # 由于压缩 KV cache，相对位置发生变化。这里会按照压缩前的原 position进行位置编码
        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1
        cos, sin = self.rotary_emb(value_states, seq_len=position_length)                
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # if past_key_value is not None:
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 4.46.1 change
       if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        #

        if past_key_value is None:
            self.heavy_budget = int(self.heavy_budget_ratio * kv_seq_len)
            self.recent_budget = int(self.recent_budget_ratio * kv_seq_len)
            self.cache_budget = self.heavy_budget + self.recent_budget

        #past_key_value = (key_states, value_states) if use_cache else None
        past_key_value.key_cache[layer_idx],past_key_value.value_cache[layer_idx] = key_states, value_states if use_cache else None #for Cache class

        key_states = repeat_kv(key_states, self.num_key_value_groups)  # 暂时 num_key_value_groups=1 ，先不管了
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"*myADD Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"*myADD Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 对 KV cache进行压缩。
        #past_key_value = self.KV_update(past_key_value, attn_weights.detach().clone())  #old
        past_key_value.key_cache[layer_idx],past_key_value.value_cache[layer_idx] = self.KV_update(past_key_value, attn_weights.detach().clone())
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        #attn_output = self.o_proj(attn_output)
        # 4，46，1
        if self.config.pretraining_tp > 1:  #其实默认也是1
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        #

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0) 
    sin = sin.squeeze(1).squeeze(0) 
    cos = cos[position_ids].unsqueeze(1)  
    sin = sin[position_ids].unsqueeze(1)  
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def convert_kvcache_llama_heavy_realdrop(model, config):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_realdrop(module, config)
        if isinstance(module, LlamaAttention):
            model._modules[name] = LlamaAttention_heavy_hitter_realdrop(config)

    return model