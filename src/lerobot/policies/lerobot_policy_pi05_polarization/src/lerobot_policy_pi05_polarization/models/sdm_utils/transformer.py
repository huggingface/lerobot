"""
(CVPR2023) Scalable, Detailed and Mask-free Universal Photometric Stereo Network 
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, attention_dropout = 0.1, dim_feedforward = 512,  
                 q_bucket_size = 1024, k_bucket_size = 2048, attn_mode = 'Normal'):
        super().__init__()
        if attn_mode == 'Efficient':
            self.q_bucket_size = q_bucket_size
            self.k_bucket_size = k_bucket_size
        self.attn_mode = attn_mode
        self.dim_V = dim_out
        self.dim_Q = dim_in
        self.dim_K = dim_in
        self.num_heads = num_heads
        self.fc_q = nn.Linear(self.dim_Q, self.dim_V, bias=False) # dimin -> dimhidden
        self.fc_k = nn.Linear(self.dim_K, self.dim_V, bias=False) # dimin -> dimhidden
        self.fc_v = nn.Linear(self.dim_K, self.dim_V, bias=False) # dimhidden -> dim
        if ln:
            self.ln0 = nn.LayerNorm(self.dim_Q)
            self.ln1 = nn.LayerNorm(self.dim_V)

        self.dropout_attn = nn.Dropout(attention_dropout)
        self.fc_o1 = nn.Linear(self.dim_V, dim_feedforward, bias=False)
        self.fc_o2 = nn.Linear(dim_feedforward, self.dim_V, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout)
        self.dropout2 = nn.Dropout(attention_dropout)

        # memory efficient attention related parameters
        # can be overriden on forward
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    # memory efficient attention
    def summarize_qkv_chunk(self, q, k, v):
        weight = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        weight_max = weight.amax(dim = -1, keepdim = True).detach()
        weight = weight - weight_max

        exp_weight = self.dropout_attn(weight.exp()) # attention_dropout
        weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v)
        return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

    def memory_efficient_attention(
        self,
        q, k, v,
        q_bucket_size = 512,
        k_bucket_size = 1024,
        eps = 1e-8,
    ):
        scale = q.shape[-1] ** -0.5
        q = q * scale

        summarize_qkv_fn = self.summarize_qkv_chunk

        # chunk all the inputs
        q_chunks = q.split(q_bucket_size, dim = -2)
        k_chunks = k.split(k_bucket_size, dim = -2)
        v_chunks = v.split(k_bucket_size, dim = -2)

        # loop through all chunks and accumulate
        values = []
        weights = []
        for q_chunk in q_chunks:
            exp_weights = []
            weighted_values = []
            weight_maxes = []
   

            for (k_chunk, v_chunk) in zip(k_chunks, v_chunks):
                exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                    q_chunk,
                    k_chunk,
                    v_chunk
                )

                exp_weights.append(exp_weight_chunk)
                weighted_values.append(weighted_value_chunk)
                weight_maxes.append(weight_max_chunk)

            weight_maxes = torch.stack(weight_maxes, dim = -1)

            weighted_values = torch.stack(weighted_values, dim = -1)
            exp_weights = torch.stack(exp_weights, dim = -1)

            global_max = weight_maxes.amax(dim = -1, keepdim = True)
            renorm_factor = (weight_maxes - global_max).exp().detach()

            exp_weights = exp_weights * renorm_factor
            weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')         

            all_values = weighted_values.sum(dim = -1)
            all_weights = exp_weights.sum(dim = -1)
            values.append(all_values)
            weights.append(all_weights)    
        values = torch.cat(values, dim=2)
        weights = torch.cat(weights, dim=2)
        # (rearrange(weights, '... -> ... 1') 
        normalized_values = values / (rearrange(weights, '... -> ... 1')  + eps)
        return normalized_values


    def forward(
        self,
        x,y,
    ):
        x = x if getattr(self, 'ln0', None) is None else self.ln0(x) # pre-normalization
        Q = self.fc_q(x) # input_dim -> embed dim       
        K, V = self.fc_k(y), self.fc_v(y) # input_dim -> embed dim
        dim_split = self.dim_V // self.num_heads # multi-head attention
        if self.attn_mode == 'Efficient':
            q_bucket_size = self.q_bucket_size
            k_bucket_size = self.k_bucket_size
            Q_ = torch.stack(Q.split(int(dim_split), 2), 1)
            K_ = torch.stack(K.split(int(dim_split), 2), 1)
            V_ = torch.stack(V.split(int(dim_split), 2), 1) 
            A = self.memory_efficient_attention(Q_, K_, V_, q_bucket_size = q_bucket_size, k_bucket_size = k_bucket_size)
            A = A.reshape(-1, A.shape[2], A.shape[3])
            Q_ = Q_.reshape(-1, Q_.shape[2], Q_.shape[3])
            O = torch.cat((Q_ + A).split(Q.size(0), 0), 2)
        else: # Basic
            Q_ = torch.cat(Q.split(int(dim_split), 2), 0)
            K_ = torch.cat(K.split(int(dim_split), 2), 0)
            V_ = torch.cat(V.split(int(dim_split), 2), 0)
            A = self.dropout_attn(torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)) # this may not be correct due to mult-head attention
            A =  A.bmm(V_) # A(Q, K, V) attention_output
            O = torch.cat((Q_ + A).split(Q.size(0), 0), 2)

        
        O_ = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O + self.dropout2(self.fc_o2(self.dropout1(F.gelu(self.fc_o1(O_))))) 
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, ln=False, attention_dropout = 0.1, dim_feedforward = 512, attn_mode = 'Normal'):
        super(SAB, self).__init__()
        self.mab = MultiHeadSelfAttentionBlock(dim_in, dim_out, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward, attn_mode=attn_mode)
    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, attn_mode='Normal'):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim)) # (B, output=1, dim)
        self.mab = MultiHeadSelfAttentionBlock(dim, dim, num_heads, ln=ln, attn_mode=attn_mode)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class CommunicationBlock(nn.Module):
    def __init__(self, dim_input, num_enc_sab=3, dim_hidden=384, dim_feedforward=1024, num_heads=8, ln=False, attention_dropout=0.1, use_efficient_attention=False):
        super(CommunicationBlock, self).__init__()

        if use_efficient_attention:
            attn_mode = 'Efficient'
        else:
            attn_mode = 'Normal'
        self.dim_hidden = dim_hidden
        modules_enc = []
        modules_enc.append(SAB(dim_input, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward, attn_mode=attn_mode))
        for k in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward, attn_mode=attn_mode))
        self.enc = nn.Sequential(*modules_enc)

    def forward(self, x):
        x = self.enc(x)
        return x

class AggregationBlock(nn.Module):
    def __init__(self, dim_input, num_enc_sab = 3, num_outputs = 1, dim_hidden=384, dim_feedforward = 1024, num_heads=8, ln=False, attention_dropout=0.1, use_efficient_attention=False):
        super(AggregationBlock, self).__init__()

        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden

        if use_efficient_attention:
            attn_mode = 'Efficient'
        else:
            attn_mode = 'Normal'

        modules_enc = []
        modules_enc.append(SAB(dim_input, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward, attn_mode=attn_mode))
        for k in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward, attn_mode=attn_mode))
        self.enc = nn.Sequential(*modules_enc)
        modules_dec = []
        modules_dec.append(PMA(dim_hidden, num_heads, num_outputs, attn_mode=attn_mode)) # after the PMA we should not put drop out
        self.dec = nn.Sequential(*modules_dec)
        
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        x = x.view(-1, self.num_outputs * self.dim_hidden)
        return x


