#Advanced Patch Embedding Transformer with Quantum Attention            

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, QuantumAttention
from layers.Embed import PatchEmbedding

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

def compute_patch_len(seq_len, num_patches=None, method="evaluate", d_model=None):
    if method == "evaluate":
        if num_patches is None:
            num_patches = 6
        patch_len = seq_len // num_patches
        print('Patch length:')
        print(patch_len)
        return max(1, patch_len)
    else:
        raise ValueError("Invalid method or missing required parameters.")
        
class Model(nn.Module):

    def __init__(self, configs, method="evaluate"):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Embedding layers cho categorical features
        self.embeddings = nn.ModuleDict()
        if hasattr(configs, 'categorical_dims'):
            for cat_name, cat_dim in configs.categorical_dims.items():
                embed_dim = min(50, (cat_dim + 1) // 2)  # Rule of thumb for embedding dimension
                self.embeddings[cat_name] = nn.Embedding(cat_dim, embed_dim)
        
        # Tính toán input dimension sau khi embed
        embed_total_dim = sum([min(50, (dim + 1) // 2) for dim in configs.categorical_dims.values()])
        numerical_dim = configs.enc_in - len(configs.categorical_dims)  # Số numerical features
        total_input_dim = embed_total_dim + numerical_dim
        
        # Projection layer to match d_model
        self.input_projection = nn.Linear(total_input_dim, configs.d_model)
        self.patch_len = compute_patch_len(configs.seq_len, method=method, d_model=configs.d_model)
        stride = self.patch_len // 2 
        print(stride)
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, stride, padding, configs.dropout)


        # Encoder
        self.use_quantum_attention = getattr(configs, "use_quantum_attention", False)  # Check if QuantumAttention is enabled
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        QuantumAttention(
                            scale=1.0 / np.sqrt(configs.d_model),
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            num_heads=configs.n_heads,
                        ) if i % 2 == 0 and self.use_quantum_attention  # Use QuantumAttention in even layers
                        else FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),  # Use FullAttention in odd layers
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for i in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - self.patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        _, _, N = x_enc.shape

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def anomaly_detection(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

   	 # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
    
        return dec_out


    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
              torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Flatten and classification head
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, categorical_features=None):
        if categorical_features is not None:
            # Process categorical features through embeddings
            embedded_cats = []
            for cat_name, cat_indices in categorical_features.items():
                embedded = self.embeddings[cat_name](cat_indices)
                embedded_cats.append(embedded)
            
            # Concatenate embeddings with numerical features
            if embedded_cats:
                cat_embedded = torch.cat(embedded_cats, dim=-1)
                x_enc = torch.cat([x_enc, cat_embedded], dim=-1)
            
            # Project to d_model dimension
            x_enc = self.input_projection(x_enc)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


