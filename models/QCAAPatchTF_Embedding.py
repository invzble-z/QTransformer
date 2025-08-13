import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding
import warnings

warnings.filterwarnings('ignore')

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class EmbeddingHead(nn.Module):
    """Enhanced head that handles both numerical and categorical features"""
    def __init__(self, d_model, target_window, categorical_dims=None, head_dropout=0):
        super().__init__()
        self.d_model = d_model
        self.target_window = target_window
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        
        if categorical_dims:
            for cat_name, cat_dim in categorical_dims.items():
                embed_dim = min(50, (cat_dim + 1) // 2)  # Rule of thumb
                self.embeddings[cat_name] = nn.Embedding(cat_dim, embed_dim)
                total_embed_dim += embed_dim
        
        # Projection layer to combine embeddings with main features
        self.feature_projection = nn.Linear(d_model + total_embed_dim, d_model)
        
        # Final prediction head
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(d_model, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, categorical_features=None):
        # x: [bs x nvars x d_model x patch_num]
        batch_size, nvars, d_model, patch_num = x.shape
        
        if categorical_features and self.embeddings:
            # Process categorical features
            embedded_cats = []
            for cat_name, cat_indices in categorical_features.items():
                if cat_name in self.embeddings:
                    # cat_indices: [batch_size, seq_len]
                    # Take the last value for prediction
                    last_cat = cat_indices[:, -1]  # [batch_size]
                    embedded = self.embeddings[cat_name](last_cat)  # [batch_size, embed_dim]
                    embedded_cats.append(embedded)
            
            if embedded_cats:
                cat_embedded = torch.cat(embedded_cats, dim=-1)  # [batch_size, total_embed_dim]
                # Expand to match patch dimensions
                cat_embedded = cat_embedded.unsqueeze(1).unsqueeze(-1).expand(-1, nvars, -1, patch_num)
                
                # Combine with main features
                x_expanded = torch.cat([x, cat_embedded], dim=2)  # [bs, nvars, d_model + embed_dim, patch_num]
                x = self.feature_projection(x_expanded.transpose(-2, -1)).transpose(-2, -1)
        
        # Apply prediction head
        x = self.flatten(x)  # [bs, nvars * d_model * patch_num]
        x = self.linear(x)   # [bs, target_window]
        x = self.dropout(x)
        return x

def compute_patch_len(seq_len, num_patches=None, method="evaluate", d_model=None):
    if method == "evaluate":
        if num_patches is None:
            num_patches = 6
        patch_len = seq_len // num_patches
    else:
        # For other methods, maintain original logic
        patch_len = seq_len // num_patches if num_patches else seq_len
    return patch_len

class QCAAPatchTF_Embedding(nn.Module):
    """
    QCAAPatchTF model enhanced with embedding support for categorical features
    """
    def __init__(self, configs, method="evaluate"):
        super(QCAAPatchTF_Embedding, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Get categorical dimensions from configs (will be set dynamically)
        self.categorical_dims = getattr(configs, 'categorical_dims', {})
        
        # Compute patch parameters
        self.patch_len = compute_patch_len(self.seq_len, method=method, d_model=configs.d_model)
        self.patch_num = self.seq_len // self.patch_len
        
        # Model parameters
        self.enc_in = configs.enc_in
        self.channel_independence = configs.channel_independence
        
        # Patch embedding
        self.W_P = nn.Linear(self.patch_len, configs.d_model)
        
        # Embedding layers for categorical features (initialized as empty, will be populated dynamically)
        self.embeddings = nn.ModuleDict()
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Prediction head
        if self.channel_independence == 1:
            self.head = FlattenHead(
                1, 
                self.patch_num * configs.d_model, 
                self.pred_len,
                head_dropout=configs.dropout
            )
        else:
            self.head = FlattenHead(
                configs.enc_in,
                self.patch_num * configs.d_model, 
                self.pred_len, 
                head_dropout=configs.dropout
            )
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, categorical_features=None, mask=None):
        # Input: x_enc [batch_size, seq_len, enc_in]
        B, L, M = x_enc.shape
        
        # Channel independence
        if self.channel_independence == 1:
            # Process each channel independently
            x_enc = x_enc.permute(0, 2, 1)  # [B, M, L]
            x_enc = x_enc.reshape(-1, L)     # [B*M, L]
            
            # Patching: [B*M, L] -> [B*M, patch_num, patch_len]
            x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
            
            # Patch embedding: [B*M, patch_num, patch_len] -> [B*M, patch_num, d_model]
            enc_out = self.W_P(x_enc)
            
            # Transformer encoding
            enc_out, _ = self.encoder(enc_out)
            
            # Reshape for head: [B*M, patch_num, d_model] -> [B, M, d_model, patch_num]
            enc_out = enc_out.reshape(B, M, self.patch_num, -1).permute(0, 1, 3, 2)
            
            # Only use the first channel for prediction in MS mode
            if M > 1:
                enc_out = enc_out[:, 0:1, :, :]  # [B, 1, d_model, patch_num]
            
        else:
            # Process all channels together
            # Patching for each channel
            patches = []
            for i in range(M):
                channel_data = x_enc[:, :, i]  # [B, L]
                channel_patches = channel_data.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, patch_num, patch_len]
                patches.append(channel_patches)
            
            x_enc = torch.stack(patches, dim=1)  # [B, M, patch_num, patch_len]
            
            # Reshape for patch embedding
            B, M, P, PL = x_enc.shape
            x_enc = x_enc.reshape(B, M * P, PL)  # [B, M*patch_num, patch_len]
            
            # Patch embedding
            enc_out = self.W_P(x_enc)  # [B, M*patch_num, d_model]
            
            # Transformer encoding
            enc_out, _ = self.encoder(enc_out)
            
            # Reshape for head: [B, M*patch_num, d_model] -> [B, M, d_model, patch_num]
            enc_out = enc_out.reshape(B, M, P, -1).permute(0, 1, 3, 2)
        
        # Apply prediction head
        dec_out = self.head(enc_out)
        
        # Reshape output: [B, pred_len] -> [B, pred_len, 1]
        if len(dec_out.shape) == 2:
            dec_out = dec_out.unsqueeze(-1)
            
        return dec_out

# Alias for compatibility
class Model(QCAAPatchTF_Embedding):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
