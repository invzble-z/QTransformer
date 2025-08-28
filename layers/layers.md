# Layers - Các Thành phần Kiến trúc Model

## Tổng quan
Folder `layers` chứa các building blocks cơ bản để xây dựng kiến trúc QTransformer. Đây là nơi định nghĩa các layer components: embedding layers, attention mechanisms, và encoder-decoder structures. Các thành phần này được thiết kế modular để tái sử dụng và kết hợp linh hoạt.

## Cấu trúc Thành phần

### 📊 **Embed.py** - Hệ thống Embedding

#### 🎯 **Mục đích chính:**
Chuyển đổi raw input data thành representation vectors phù hợp cho Transformer architecture, bao gồm value embeddings, positional encodings, và temporal features.

#### 🧩 **Các Class chính:**

##### **1. TokenEmbedding**
```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
```
**Chức năng:** Chuyển đổi input features thành token embeddings sử dụng 1D convolution
**Ứng dụng:** Embedding cho numerical features trong time series data

##### **2. PositionalEmbedding**
```python
class PositionalEmbedding(nn.Module):
    def forward(self, x):
        return self.pe[:, :x.size(1)]  # Positional encoding sinusoid
```
**Chức năng:** Thêm thông tin vị trí cho sequence tokens
**Ý nghĩa:** Giúp model hiểu thứ tự temporal trong time series

##### **3. TemporalEmbedding**
```python
class TemporalEmbedding(nn.Module):
    def forward(self, x):
        hour_x = self.hour_embed(x[:, :, 3])      # Hour of day
        weekday_x = self.weekday_embed(x[:, :, 2])  # Day of week  
        day_x = self.day_embed(x[:, :, 1])        # Day of month
        month_x = self.month_embed(x[:, :, 0])    # Month
        return hour_x + weekday_x + day_x + month_x
```
**Chức năng:** Embedding cho temporal features (giờ, ngày, tuần, tháng)
**Lợi ích:** Capture seasonality patterns trong supply chain data

##### **4. DataEmbedding (Composite Class)**
```python
class DataEmbedding(nn.Module):
    def forward(self, x, x_mark):
        x = self.value_embedding(x) + \
            self.temporal_embedding(x_mark) + \
            self.position_embedding(x)
        return self.dropout(x)
```
**Chức năng:** Kết hợp tất cả embedding types thành final representation
**Input:** Raw features (x) + time features (x_mark)
**Output:** Rich embedding vectors cho Transformer layers

##### **5. PatchEmbedding**
```python
class PatchEmbedding(nn.Module):
    def forward(self, x):
        # Patching process
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
```
**Chức năng:** Chia time series thành patches nhỏ, tương tự Vision Transformer
**Ưu điểm:** Giảm computational complexity, tăng locality awareness

#### 🎨 **Embedding Variants:**

- **DataEmbedding_inverted**: Cho inverted architecture (feature-wise attention)
- **DataEmbedding_wo_pos**: Không sử dụng positional embedding
- **TimeFeatureEmbedding**: Linear embedding cho continuous time features

### ⚡ **SelfAttention_Family.py** - Họ Attention Mechanisms

#### 🎯 **Mục đích chính:**
Cung cấp các loại attention mechanisms khác nhau để capture dependencies trong time series data, từ standard attention đến quantum-enhanced attention.

#### 🧠 **Các Attention Types:**

##### **1. FullAttention (Standard)**
```python
class FullAttention(nn.Module):
    def forward(self, queries, keys, values, attn_mask):
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
```
**Đặc điểm:** Classic scaled dot-product attention
**Sử dụng:** Baseline attention cho standard Transformer

##### **2. DSAttention (De-stationary)**
```python
class DSAttention(nn.Module):
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
```
**Đặc điểm:** Attention với learnable de-stationary factors (tau, delta)
**Ứng dụng:** Xử lý non-stationary patterns trong supply chain data
**Lợi ích:** Adaptive cho seasonal và trend changes

##### **3. ProbAttention (Sparse)**
```python
class ProbAttention(nn.Module):
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Probability-based key sampling
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
```
**Đặc điểm:** Chỉ compute attention cho top-k most relevant pairs
**Ưu điểm:** Giảm complexity từ O(L²) xuống O(L log L)
**Sử dụng:** Long sequence forecasting

##### **4. QuantumAttention (Experimental)** 🚀
```python
class QuantumAttention(nn.Module):
    def variational_circuit(self, params):
        # Quantum circuit với VQE
        for i in range(self.num_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))
```
**Đặc điểm:** Sử dụng Variational Quantum Eigensolver cho attention computation
**Innovation:** Quantum entanglement để capture complex dependencies
**Status:** Research experiment cho Phase 2

##### **5. AttentionLayer (Wrapper)**
```python
class AttentionLayer(nn.Module):
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Multi-head projections
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # Apply attention mechanism
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        return self.out_projection(out.view(B, L, -1)), attn
```
**Chức năng:** Wrapper để implement multi-head attention với bất kỳ attention type nào

#### 🔄 **Specialized Layers:**

- **TwoStageAttentionLayer**: Two-stage attention cho hierarchical patterns
- **TimeAttention**: Time-aware attention với temporal weighting

### 🏗️ **Transformer_EncDec.py** - Encoder-Decoder Architecture

#### 🎯 **Mục đích chính:**
Định nghĩa các building blocks cho Transformer encoder-decoder architecture, bao gồm các layer components và stacking logic.

#### 🧩 **Core Components:**

##### **1. EncoderLayer**
```python
class EncoderLayer(nn.Module):
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)  # Residual connection
        
        # Feed-forward network
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn  # Layer normalization
```
**Cấu trúc:** Self-attention + Feed-forward + Residual connections + Layer normalization
**Chức năng:** Encode input sequence thành rich representations

##### **2. Encoder (Stack)**
```python
class Encoder(nn.Module):
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns
```
**Chức năng:** Stack nhiều EncoderLayers để tạo deep representations
**Flexibility:** Support cả convolutional layers cho downsampling

##### **3. DecoderLayer**
```python
class DecoderLayer(nn.Module):
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Self-attention trên decoder sequence
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        
        # Cross-attention với encoder output
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        
        # Feed-forward network
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)
```
**Cấu trúc:** Self-attention + Cross-attention + Feed-forward
**Chức năng:** Generate output sequence dựa trên encoder representations

##### **4. Decoder (Stack)**
```python
class Decoder(nn.Module):
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        
        if self.projection is not None:
            x = self.projection(x)  # Final output projection
        return x
```
**Chức năng:** Stack nhiều DecoderLayers + final projection
**Output:** Final forecasting results

##### **5. ConvLayer (Auxiliary)**
```python
class ConvLayer(nn.Module):
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))    # 1D convolution
        x = self.norm(x)                         # Batch normalization
        x = self.activation(x)                   # ELU activation
        x = self.maxPool(x)                      # Max pooling
        return x.transpose(1, 2)
```
**Chức năng:** Downsampling layer cho hierarchical attention
**Ứng dụng:** Giảm sequence length trong deep encoders

## Tích hợp với QTransformer

### 🔗 **Component Integration:**

#### **Phase 1 Architecture Stack:**
```python
# 1. Data Embedding
embedding = DataEmbedding(c_in=features, d_model=512)

# 2. Encoder Stack
encoder_layers = [
    EncoderLayer(
        attention=AttentionLayer(DSAttention(), d_model=512, n_heads=8),
        d_model=512, d_ff=2048
    ) for _ in range(6)
]
encoder = Encoder(encoder_layers)

# 3. Decoder Stack  
decoder_layers = [
    DecoderLayer(
        self_attention=AttentionLayer(FullAttention(), d_model=512, n_heads=8),
        cross_attention=AttentionLayer(FullAttention(), d_model=512, n_heads=8),
        d_model=512, d_ff=2048
    ) for _ in range(3)
]
decoder = Decoder(decoder_layers, projection=nn.Linear(512, c_out))
```

#### **Embedding cho Supply Chain:**
```python
# Supply chain specific embedding
data_embedding = DataEmbedding(
    c_in=21,           # 21 features từ processed data
    d_model=512,       # Hidden dimension
    embed_type='fixed', # Fixed temporal embedding
    freq='d',          # Daily frequency
    dropout=0.1
)

# Input: [batch_size, seq_len, 21], [batch_size, seq_len, 4]
# Output: [batch_size, seq_len, 512]
```

### 🎯 **Attention Strategy cho Multi-Market:**

#### **DSAttention cho Non-stationary Supply Chain:**
- **tau parameter**: Học seasonal scaling factors
- **delta parameter**: Học trend adjustment factors  
- **Multi-head**: 8 heads để capture different temporal patterns

#### **QuantumAttention Research Direction:**
- **Entanglement**: Model cross-market dependencies
- **VQE optimization**: Learnable quantum parameters
- **Hybrid approach**: Classical + Quantum attention combination

## Performance Characteristics

### ⚡ **Computational Complexity:**

| Attention Type | Time Complexity | Space Complexity | Use Case |
|---------------|----------------|------------------|----------|
| FullAttention | O(L²) | O(L²) | Standard sequences |
| ProbAttention | O(L log L) | O(L log L) | Long sequences |
| DSAttention | O(L²) | O(L²) | Non-stationary data |
| QuantumAttention | O(L²) | O(L²) | Research experiments |

### 📊 **Phase 1 Usage:**
- **Primary**: DSAttention với TemporalEmbedding
- **Embedding**: DataEmbedding với 21 features + 4 time features
- **Architecture**: 6 encoder layers + 3 decoder layers
- **Results**: 98.71% MSE cải thiện với attention mechanisms này

## Hướng phát triển Phase 2

### 🚀 **Planned Enhancements:**

#### 1. **Advanced Embedding:**
- **Categorical Embeddings**: Tích hợp Market, Category embeddings  
- **External Features**: Weather, holiday, economic indicators
- **Dynamic Embedding**: Adaptive embedding dimensions

#### 2. **Attention Innovations:**
- **Multi-scale Attention**: Capture patterns ở multiple time scales
- **Cross-market Attention**: Explicit cross-region dependencies
- **Adaptive Sparsity**: Dynamic attention sparsity patterns

#### 3. **Architecture Optimization:**
- **Efficient Transformers**: Linformer, Performer variants
- **Memory Efficient**: Gradient checkpointing, mixed precision
- **Model Parallelism**: Distributed training support

### 🔬 **Research Directions:**
- **Quantum Integration**: Expand QuantumAttention với real quantum hardware
- **Neuromorphic Attention**: Brain-inspired attention mechanisms
- **Causal Discovery**: Attention-based causal inference

## Hướng dẫn Sử dụng

### 🛠️ **Custom Attention Implementation:**
```python
# Tạo custom attention type
from layers.SelfAttention_Family import AttentionLayer

# Initialize attention
custom_attention = AttentionLayer(
    attention=DSAttention(mask_flag=True, scale=None, attention_dropout=0.1),
    d_model=512,
    n_heads=8
)

# Sử dụng trong EncoderLayer
encoder_layer = EncoderLayer(
    attention=custom_attention,
    d_model=512,
    d_ff=2048,
    dropout=0.1
)
```

### 🔧 **Debugging Attention:**
```python
# Kiểm tra attention weights
_, attention_weights = attention_layer(x, x, x, output_attention=True)
print(f"Attention shape: {attention_weights.shape}")  # [B, H, L, S]

# Visualize attention patterns
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0, 0].detach().cpu(), cmap='Blues')
plt.title('Attention Pattern Head 0')
```

### ⚠️ **Common Issues:**
- **Memory errors**: Giảm sequence length hoặc sử dụng ProbAttention
- **NaN values**: Kiểm tra attention mask và scaling factor
- **Slow training**: Sử dụng mixed precision và efficient attention variants

Layers folder này cung cấp foundation building blocks đã được tối ưu cho supply chain forecasting, với flexibility cao để mở rộng cho Phase 2. Architecture modular giúp dễ dàng experiment với different attention mechanisms và embedding strategies.
