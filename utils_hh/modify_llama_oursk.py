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

# Simulated implementation without CUDA

__all__ = ['LlamaAttention_heavy_hitter_our_sketch']

class LlamaAttention_heavy_hitter_our_sketch(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        
        self.epsilon_times = config.epsilon_times
        self.epsilon = torch.finfo(torch.float16).tiny * self.epsilon_times

        self.layer_idx = config.layer_idx
        self.heavy_budget_ratio = config.heavy_ratio_layers[self.layer_idx]
        self.recent_budget_ratio = config.recent_ratio_layers[self.layer_idx]
        self.distance_weight = config.pyramidinfer_distance_weight
        self.sink_len = config.pyramidinfer_sink_len

        self.similarity_threshold = config.sketch_similarity_threshold
        self.prefill_similarity_threshold = config.sketch_prefill_similarity_threshold
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_recent_scores = None
        self.cluster_cnt = None


    def _reset_masks(self):
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_recent_scores = None
        self.cluster_cnt = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def KV_update(self, 
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_weights: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor]:
        past_key_value = (past_key_value[0].float(), past_key_value[1].float())
        
        # update cnt & scores
        is_prefill = True
        num_new_tokens = attn_weights.shape[2]
        if self.previous_recent_scores is None: # prefill
            self.previous_recent_scores = attn_weights[0, :, -self.recent_budget:, :].detach().clone()
            add_cluster_cnt = torch.ones((attn_weights.size(1), attn_weights.size(3)), device=attn_weights.device, dtype=torch.float32)
        else:                                   # decoding
            is_prefill = False
            assert num_new_tokens == 1
            self.previous_recent_scores = torch.cat([self.previous_recent_scores , torch.zeros((self.previous_recent_scores.shape[0], self.previous_recent_scores.shape[1], 1), 
                                                                                        device=attn_weights.device, dtype=torch.float32)], dim=-1)
            self.previous_recent_scores = torch.cat([self.previous_recent_scores, attn_weights[0]], dim=-2)                    
            self.previous_recent_scores = self.previous_recent_scores[:, -self.recent_budget:, :]
            add_cluster_cnt = torch.zeros((attn_weights.size(1), attn_weights.size(3)), device=attn_weights.device, dtype=torch.float32)
            add_cluster_cnt[:, -num_new_tokens] = 1.0
            add_cluster_cnt[:, :-num_new_tokens] += self.cluster_cnt
        self.cluster_cnt = add_cluster_cnt

        if past_key_value is None:
            return None

        seq_len = past_key_value[0].size(2)
        assert seq_len == self.previous_recent_scores.size(-1)

        if seq_len <= self.cache_budget:
            return past_key_value
        
        bsz, num_heads, _, head_dim = past_key_value[0].shape

        weight_previous_recent_scores = self.previous_recent_scores * torch.linspace(1.0, self.distance_weight, self.previous_recent_scores.shape[-2],
                                                                                        device=attn_weights.device, dtype=torch.float32)[None, :, None]
        selected_scores_sum = (weight_previous_recent_scores.sum(1))[:, self.sink_len:-self.recent_budget]
        
        _, keep_topk = torch.topk(selected_scores_sum, self.heavy_budget-self.sink_len, dim=-1)
        keep_topk = keep_topk.sort().values + self.sink_len

        keep_recent = torch.arange(seq_len - self.recent_budget, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_sink = torch.arange(0, self.sink_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_sink, keep_topk, keep_recent], dim=-1)

        mask = torch.zeros((self.previous_recent_scores.shape[0], self.previous_recent_scores.shape[-1]), dtype=torch.bool).to(past_key_value[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        new_key_states = past_key_value[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        new_value_states = past_key_value[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        new_cluster_cnt = self.cluster_cnt[mask].view(num_heads, self.cache_budget)

        discard_idx = torch.nonzero(~mask, as_tuple=False)[:, 1].view(num_heads, -1).unsqueeze(0)
        discarded_key_states = past_key_value[0].gather(2, discard_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        discarded_value_states = past_key_value[1].gather(2, discard_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        discard_recent_score = self.previous_recent_scores.gather(2, 
                                    discard_idx.squeeze(0).unsqueeze(1).expand(-1, self.previous_recent_scores.shape[1], -1))  
        discard_cluster_cnt = self.cluster_cnt.gather(1, discard_idx.squeeze(0))

        expanded_mask = mask.unsqueeze(1).repeat(1, self.previous_recent_scores.shape[1], 1)
        self.previous_recent_scores = self.previous_recent_scores[expanded_mask].view(num_heads, -1, self.cache_budget)
        
        discarded_key_states_norm = F.normalize(discarded_key_states, p=2, dim=-1).half() 
        new_key_states_norm = F.normalize(new_key_states, p=2, dim=-1).half()
        cosine_similarity = torch.matmul(discarded_key_states_norm, new_key_states_norm.transpose(-1, -2))[0]

        # merge
        max_sim, max_sim_idx = torch.max(cosine_similarity, dim=-1)
        max_sim_mask = max_sim >= self.similarity_threshold

        W_keep_esti = self.previous_recent_scores.mean(1)
        W_keep_esti[:, -self.recent_budget:] *= (self.recent_budget / torch.arange(self.recent_budget, 0, -1, dtype=torch.float32, 
                                device=W_keep_esti.device).unsqueeze(0).repeat(self.num_heads, 1))
        W_keep_esti += self.epsilon
        W_keep = new_cluster_cnt * W_keep_esti          
        
        new_key_states[0] *= W_keep.unsqueeze(-1)       
        new_value_states[0] *= W_keep.unsqueeze(-1)     
        
        W_discard_esit = discard_recent_score.mean(1) + self.epsilon
        discard_cluster_cnt = torch.where(max_sim_mask, discard_cluster_cnt, torch.tensor(0.0, device=discard_cluster_cnt.device, dtype=torch.float32))
        W_discard = discard_cluster_cnt * W_discard_esit     
        discarded_key_states[0] *= W_discard.unsqueeze(-1)          
        discarded_value_states[0] *= W_discard.unsqueeze(-1) 
        
        expanded_max_sim_idx = max_sim_idx.unsqueeze(-1).expand(-1, -1, head_dim)  
        new_key_states[0].scatter_add_(1, expanded_max_sim_idx, discarded_key_states[0])        
        new_value_states[0].scatter_add_(1, expanded_max_sim_idx, discarded_value_states[0])    
        
        W_total = torch.scatter_add(W_keep, 1, max_sim_idx, W_discard) 
        new_value_states[0] /= W_total.unsqueeze(-1)                   
        new_key_states[0] /= W_total.unsqueeze(-1)                  

        cnt_total = torch.scatter_add(new_cluster_cnt, 1, max_sim_idx, discard_cluster_cnt)
        self.cluster_cnt = cnt_total

        new_recent_scores = self.previous_recent_scores * new_cluster_cnt.unsqueeze(1)
        discard_recent_score = discard_recent_score * discard_cluster_cnt.unsqueeze(1)
        new_recent_scores.scatter_add_(2, max_sim_idx.unsqueeze(1).expand(-1, new_recent_scores.size(1), -1), discard_recent_score)
        self.previous_recent_scores = new_recent_scores / cnt_total.unsqueeze(1)  
        tri_mask = torch.tril(torch.ones(self.previous_recent_scores.size(-2), self.previous_recent_scores.size(-1), device=self.previous_recent_scores.device, dtype=torch.float32), 
                                diagonal=self.previous_recent_scores.size(-1)-self.previous_recent_scores.size(-2))
        self.previous_recent_scores *= tri_mask.unsqueeze(0)

        new_key_states = new_key_states.half()
        new_value_states = new_value_states.half()
        return (new_key_states, new_value_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size() 

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        position_length = kv_seq_len
        if not position_ids.nelement() > 1: 
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1
        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
                
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if past_key_value is None:
            self.heavy_budget = math.ceil(self.heavy_budget_ratio * kv_seq_len)
            self.recent_budget = math.ceil(self.recent_budget_ratio * kv_seq_len)
            if self.recent_budget <= 0:
                self.heavy_budget -= (1 - self.recent_budget)
                self.recent_budget = 1
            assert self.recent_budget > 0 and self.heavy_budget > 0
            self.cache_budget = self.heavy_budget + self.recent_budget

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"*Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.to(torch.float32)
        exp_attn_weights = torch.exp(attn_weights)

        attn_weights_wcnt = attn_weights
        if self.cluster_cnt is not None:
            attn_weights_wcnt[:, :, :, :-1] += torch.log(self.cluster_cnt).unsqueeze(0).unsqueeze(2)
        attn_weights_wcnt = nn.functional.softmax(attn_weights_wcnt, dim=-1, dtype=torch.float32).to(query_states.dtype)

        past_key_value = self.KV_update(past_key_value, exp_attn_weights) 

        attn_output = torch.matmul(attn_weights_wcnt, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights_wcnt = None

        return attn_output, attn_weights_wcnt, past_key_value


def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
