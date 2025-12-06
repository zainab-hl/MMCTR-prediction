### all components needed : embeddings layer, sequential feature learning module,DCN, prediction layer, loss function
## params of each component can be adjusted as needed
import torch.nn.functional as F
import polars as pl
import torch, torch.nn as nn
import pandas as pd
import numpy as np

class CTRPredictionLayer(nn.Module):
    """
    Prediction layer (Section 2.2.4).
    2-layer perceptron with hidden units [64, 32] and sigmoid output.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.0):
        super().__init__()
        assert len(hidden_dims) == 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            #nn.Dropout(dropout_rate), ### paper did not use dropout, but we can use it if needed
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            #nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1)  # Output layer
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, f_o):
        logits = self.mlp(f_o)
        y_hat= torch.sigmoid(logits)
        return y_hat


class CTRLoss(nn.Module):
    """
    Binary cross-entropy loss (Section 2.2.5).
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_hat, y):
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        loss = F.binary_cross_entropy(y_hat, y, reduction=self.reduction)
        return loss


#### deep cross network
import torch
import torch.nn as nn

class DCNv2FeatureInteraction(nn.Module):
    """
    DCNv2 feature interaction module (Section 2.2.3, Equations 7-8).

    Inputs:
        e_target : [B, D_t]   # Target item embedding
        e_side   : [B, D_s]   # Side feature embeddings
        S_o      : [B, D_sqo] # Sequential features from transformer

    Output:
        f_o : [B, input_dim + deep_output_dim] = [c_o, d_o]
    """
    def __init__(
        self,
        input_dim,                  # D = D_t + D_s + D_sqo
        num_cross_layers=3,
        deep_hidden_dims=[1024, 512, 256], # Table 1
        deep_output_dim=256,        # d_o dimension
        dropout_rate=0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_cross_layers = num_cross_layers

        # Cross Network (c_{l+1} = f_i ⊙ (W_l c_l + b_l) + c_l)
        self.cross_W = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False)
                                      for _ in range(num_cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim))
                                         for _ in range(num_cross_layers)])

        # Deep Network: MLP_f(f_i)
        deep_layers = []
        dim = input_dim
        for h in deep_hidden_dims:
            deep_layers.append(nn.Linear(dim, h))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            dim = h
        deep_layers.append(nn.Linear(dim, deep_output_dim))
        self.deep_mlp = nn.Sequential(*deep_layers)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, e_target, e_side, S_o):
        # Initial concatenation: f_i = [e_target || e_side || S_o]
        f_i = torch.cat([e_target, e_side, S_o], dim=-1)

        # Cross Network
        c_l = f_i
        for l in range(self.num_cross_layers):
            linear_out = self.cross_W[l](c_l) + self.cross_b[l]
            c_l = f_i * linear_out + c_l
            c_l = self.dropout(c_l)
        c_o = c_l

        # Deep Network
        d_o = self.deep_mlp(self.dropout(f_i))

        # Parallel output
        f_o = torch.cat([c_o, d_o], dim=-1)
        return f_o

## side features embeddings
# Simplified version 
class SimpleSideFeatureEmbedding(nn.Module):
    """
    Simplified version assuming like_level and view_level are categorical.
    """

    def __init__(self, like_vocab_size=11, view_vocab_size=11, emb_dim=16):
        super().__init__()

        self.like_emb = nn.Embedding(like_vocab_size, emb_dim, padding_idx=0)
        self.view_emb = nn.Embedding(view_vocab_size, emb_dim, padding_idx=0)

        self.output_dim = emb_dim * 2

        nn.init.xavier_uniform_(self.like_emb.weight.data[1:])
        nn.init.xavier_uniform_(self.view_emb.weight.data[1:])

        with torch.no_grad():
            self.like_emb.weight.data[0].zero_()
            self.view_emb.weight.data[0].zero_()

    def forward(self, likes_level, views_level):
        e_like = self.like_emb(likes_level)  # [B, emb_dim]
        e_view = self.view_emb(views_level)  # [B, emb_dim]

        e_side = torch.cat([e_like, e_view], dim=-1)  # [B, output_dim]
        return e_side
    
#### embeddings layer
class ItemEmbeddingLayer(nn.Module):
    """
    e_item = concat( e_id(16d), e_tag1(32d), ..., e_tagT(32d), frozen_emb(128d) )
    Accepts frozen_mm (tensor) at init and uses it as lookup (not learnable).
    """
    def __init__(self, num_items, num_tag_ids, frozen_mm, tag_count_per_item=5):
        super().__init__()
        self.tag_count_per_item = tag_count_per_item
        self.frozen_mm = nn.Parameter(frozen_mm, requires_grad=False)  # [num_items, 128]
        self.id_emb = nn.Embedding(num_items, 16, padding_idx=0)      # 16-d
        self.tag_emb = nn.Embedding(num_tag_ids, 32, padding_idx=0)   # 32-d per tag

        # init (skip padding idx)
        if num_items > 1:
            nn.init.xavier_uniform_(self.id_emb.weight.data[1:])
        if num_tag_ids > 1:
            nn.init.xavier_uniform_(self.tag_emb.weight.data[1:])

        # ensure zero at padding
        with torch.no_grad():
            self.id_emb.weight.data[0].zero_()
            self.tag_emb.weight.data[0].zero_()

    def forward(self, item_ids, item_tags):
        """
        item_ids:  [B, N] or [N] or [B]             -- integer ids
        item_tags: [B, N, T] or [N, T] or [T]      -- tag ids per item (T tags)
        returns:   embeddings with shape [..., 16 + T*32 + 128]
        """
        # id embedding => [..., 16]
        id_vec = self.id_emb(item_ids)

        tag_vecs = self.tag_emb(item_tags)   # tag ids of 0 -> zero vector
        if tag_vecs.dim() == 2:  # [T, 32] or [B, T]? handle generically
            # If tag_vecs is [T,32] (single item case), make it [..., T, 32]
            tag_vecs = tag_vecs.unsqueeze(0)
        tag_flat = tag_vecs.view(*tag_vecs.shape[:-2], -1)  # [..., T*32]

        # frozen multimodal: lookup by item_ids 
        frozen_vec = self.frozen_mm[item_ids]  
        # concat final
        final = torch.cat([id_vec, tag_flat, frozen_vec], dim=-1)
        return final
#### sequential feature learning module: for outputing the user vector
class SequentialFeatureLearning(nn.Module):
    """
    Implements Section 2.2.2 with corrections:
      - concatenates e_item_i || e_target per item (broadcasted)
      - Transformer with src_key_padding_mask
      - masked max-pool to ignore padded positions
      - handles k > N by zero-padding the short-term vector
    Inputs:
      history_ids  : [B, N]
      history_tags : [B, N, T]
      target_id    : [B]
      target_tags  : [B, T]
    Output:
      S_o shape = [B, k*dt + dt]  (stable even if k > N)
    """
    def __init__(self,
                 item_embedding_layer: ItemEmbeddingLayer,
                 embed_dim=304,   # item embedding dim
                 dt=128,
                 num_layers=2,
                 num_heads=4,
                 k=16,
                 dropout=0.2):
        super().__init__()
        self.item_embedding_layer = item_embedding_layer
        self.embed_dim = embed_dim            # 304
        self.concat_dim = embed_dim * 2       # e_item || e_target
        self.dt = dt
        self.k = k

        self.input_proj = nn.Linear(self.concat_dim, dt)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dt,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dt * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, history_ids, history_tags, target_id, target_tags):
        """
        history_ids:  [B, N]
        history_tags: [B, N, T]
        target_id:    [B]
        target_tags:  [B, T]
        """
        B, N = history_ids.shape
        device = history_ids.device

        hist_emb = self.item_embedding_layer(history_ids, history_tags)   # [..., 304]

        targ_emb = self.item_embedding_layer(target_id, target_tags)     # [B, 304]
        targ_exp = targ_emb.unsqueeze(1).expand(-1, N, -1)               # [B, N, 304]

        seq = torch.cat([hist_emb, targ_exp], dim=-1)

        # 4) project to dt dim: [B, N, dt]
        seq = self.input_proj(seq)

        padding_mask = (history_ids == 0)   

        S = self.transformer(seq, src_key_padding_mask=padding_mask)   # [B, N, dt]

        # 7) Short-term: last k outputs.
        # If k > N, we use available outputs and left-pad with zeros so output shape is stable: [B, k*dt]
        k_eff = min(self.k, N)
        if k_eff > 0:
            last_k = S[:, -k_eff:, :]  
            last_k_flat = last_k.reshape(B, -1)  
            if k_eff < self.k:
                # Need to pad left with zeros to reach k*dt
                pad_elems = self.k - k_eff
                pad = torch.zeros(B, pad_elems * self.dt, device=device, dtype=last_k_flat.dtype)
                short_term = torch.cat([pad, last_k_flat], dim=-1)  # [B, k*dt]
            else:
                short_term = last_k_flat  # [B, k*dt]
        else:
            short_term = torch.zeros(B, self.k * self.dt, device=device, dtype=S.dtype)

        # 8) Long-term: masked max-pooling over time.
        # For masked positions set to a very small value before max.
        neg_inf = torch.tensor(-1e9, device=device, dtype=S.dtype)
        # expand padding mask to [B, N, 1] for broadcasting
        pm_expand = padding_mask.unsqueeze(-1)  # True where padding
        S_masked = torch.where(pm_expand, neg_inf, S) 
        long_term = torch.max(S_masked, dim=1).values  # [B, dt]

        long_term = torch.where(torch.isfinite(long_term), long_term, torch.zeros_like(long_term))

        # Final S_o: concat short_term (k*dt) and long_term (dt) => [B, k*dt + dt]
        S_o = torch.cat([short_term, long_term], dim=-1)
        return S_o
