# Models - Kiến trúc Mô hình QTransformer

## Tổng quan
Folder `models` chứa các kiến trúc mô hình chính của dự án QTransformer. Đây là nơi định nghĩa các neural network architectures được thiết kế đặc biệt cho supply chain forecasting, từ baseline model đến advanced embedding-enhanced variants. Tất cả models đều dựa trên **Patch-based Transformer** approach với quantum attention capabilities.

## Kiến trúc Mô hình

### 🏗️ **QCAAPatchTF.py** - Mô hình Cơ bản (Baseline)

#### 🎯 **Mục đích chính:**
Mô hình Transformer baseline sử dụng patch embedding technique kết hợp với quantum attention mechanisms cho time series forecasting.

#### 🧩 **Thành phần chính:**

##### **1. Patch Embedding Strategy:**
```python
def compute_patch_len(seq_len, num_patches=None, method="evaluate"):
    if method == "evaluate":
        num_patches = 6 if num_patches is None else num_patches
        patch_len = seq_len // num_patches
        return max(1, patch_len)
```
**Chức năng:** Chia time series thành patches để giảm computational complexity
**Lợi ích:** Giảm từ O(L²) xuống O(P²) với P << L

##### **2. Hybrid Attention Architecture:**
```python
# Encoder với mixed attention
AttentionLayer(
    QuantumAttention() if i % 2 == 0 and self.use_quantum_attention
    else FullAttention(),  # Alternating layers
    d_model, n_heads
)
```
**Đặc điểm:** 
- **Even layers**: QuantumAttention để capture complex dependencies
- **Odd layers**: FullAttention để maintain stability
- **Hybrid approach**: Cân bằng innovation và reliability

##### **3. Multi-task Support:**
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, categorical_features=None):
    if self.task_name == 'long_term_forecast':
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    elif self.task_name == 'anomaly_detection':
        return self.anomaly_detection(x_enc, x_mark_enc)
    elif self.task_name == 'classification':
        return self.classification(x_enc, x_mark_enc)
```
**Flexibility:** Một model cho multiple tasks

##### **4. Normalization Strategy:**
```python
# Non-stationary normalization
means = x_enc.mean(1, keepdim=True).detach()
x_enc = x_enc - means
stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
x_enc /= stdev

# De-normalization at output
dec_out = dec_out * stdev + means
```
**Ý nghĩa:** Xử lý non-stationary patterns trong supply chain data

#### 🔧 **Categorical Features Support:**
```python
# Embedding cho categorical features
self.embeddings = nn.ModuleDict()
for cat_name, cat_dim in configs.categorical_dims.items():
    embed_dim = min(50, (cat_dim + 1) // 2)
    self.embeddings[cat_name] = nn.Embedding(cat_dim, embed_dim)

# Kết hợp với numerical features
if categorical_features:
    embedded_cats = [self.embeddings[name](indices) for name, indices in categorical_features.items()]
    cat_embedded = torch.cat(embedded_cats, dim=-1)
    x_enc = torch.cat([x_enc, cat_embedded], dim=-1)
    x_enc = self.input_projection(x_enc)
```

### 🌟 **QCAAPatchTF_Embedding.py** - Mô hình Chính (Phase 1)

#### 🎯 **Mục đích chính:**
Enhanced version của QCAAPatchTF với advanced embedding support và multi-market forecasting capabilities. Đây là model chính đạt **98.71% MSE improvement** trong Phase 1.

#### 🚀 **Tính năng nâng cao:**

##### **1. EmbeddingHead Class:**
```python
class EmbeddingHead(nn.Module):
    def __init__(self, d_model, target_window, n_markets=1, categorical_dims=None):
        # Embedding layers cho categorical features
        self.embeddings = nn.ModuleDict()
        for cat_name, cat_dim in categorical_dims.items():
            embed_dim = min(50, (cat_dim + 1) // 2)
            self.embeddings[cat_name] = nn.Embedding(cat_dim, embed_dim)
        
        # Final prediction cho multi-market
        self.linear = nn.Linear(d_model, target_window * n_markets)
```
**Innovation:** Specialized head cho multi-market output với embedding integration

##### **2. Channel Independence Strategy:**
```python
if self.channel_independence == 1:
    # Xử lý mỗi channel riêng biệt
    x_enc = x_enc.permute(0, 2, 1).reshape(-1, L)
    x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
    enc_out = self.W_P(x_enc)
else:
    # Xử lý tất cả channels cùng nhau
    patches = [channel_data.unfold(...) for channel_data in x_enc]
    x_enc = torch.stack(patches, dim=1)
```
**Flexibility:** Support cả independent và joint channel processing

##### **3. Multi-Market Output:**
```python
class FlattenHead(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.flatten(x)
        x = self.linear(x)  # Linear to target_window * n_markets
        # Reshape to [batch_size, target_window, n_markets]
        x = x.view(batch_size, self.target_window, self.n_markets)
        return x
```
**Output:** Direct multi-market predictions: [batch, pred_len, 3_markets]

#### 📊 **Phase 1 Configuration:**
```python
# Supply chain specific setup
configs.enc_in = 21          # 21 numerical features
configs.c_out = 3            # 3 markets (Europe, LATAM, USCA)
configs.seq_len = 7          # 7 days input sequence
configs.pred_len = 1         # 1 day prediction horizon
configs.d_model = 512        # Hidden dimension
configs.n_heads = 8          # Multi-head attention
configs.e_layers = 6         # Encoder layers
```

### 🌍 **MultiRegionQCAAPatchTF.py** - Mô hình Đa Vùng

#### 🎯 **Mục đích chính:**
Extension của base model để hỗ trợ region-specific modeling với shared base architecture.

#### 🧩 **Kiến trúc đặc biệt:**

##### **1. Region-Specific Heads:**
```python
class MultiRegionModel(nn.Module):
    def __init__(self, configs):
        self.region_heads = nn.ModuleList([
            nn.Linear(configs.d_model, configs.pred_len) 
            for _ in range(self.num_regions)
        ])
        self.region_embedding = nn.Embedding(self.num_regions, configs.d_model)
```

##### **2. Region-Enhanced Features:**
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, region_ids):
    base_features = self.base_model.encoder_forward(x_enc, x_mark_enc)
    region_embeds = self.region_embedding(region_ids)
    enhanced_features = base_features + region_embeds
    
    # Generate predictions per region
    predictions = [region_head(enhanced_features[region_mask]) 
                  for region_head in self.region_heads]
```

**Use case:** Experimental cho Phase 2 khi cần region-specific parameters

## Thiết kế Patch Embedding

### 🔄 **Patch Strategy:**

#### **Tại sao sử dụng Patches:**
1. **Computational Efficiency**: Giảm complexity từ O(L²) xuống O(P²)
2. **Local Pattern Capture**: Tốt hơn cho time series patterns
3. **Memory Efficiency**: Xử lý longer sequences với limited memory
4. **Locality Bias**: Inductive bias phù hợp cho temporal data

#### **Patch Processing Pipeline:**
```python
# 1. Input: [batch_size, seq_len, features] = [B, 7, 21]
# 2. Unfold: [B, 7, 21] -> [B, patch_num, patch_len, 21]
x_enc = x_enc.unfold(dimension=1, size=patch_len, step=patch_len)

# 3. Patch Embedding: [B, patch_num, patch_len] -> [B, patch_num, d_model]
enc_out = self.W_P(x_enc)

# 4. Transformer Encoding: [B, patch_num, d_model] -> [B, patch_num, d_model]
enc_out, _ = self.encoder(enc_out)

# 5. Head Projection: [B, patch_num, d_model] -> [B, pred_len, n_markets]
output = self.head(enc_out)
```

### ⚡ **Performance Characteristics:**

| Component | Input Shape | Output Shape | Computation |
|-----------|-------------|--------------|-------------|
| Patch Embedding | [B, 7, 21] | [B, 2, 512] | O(patch_len × d_model) |
| Transformer | [B, 2, 512] | [B, 2, 512] | O(patch_num²) |
| Head | [B, 2, 512] | [B, 1, 3] | O(d_model × markets) |

**Hiệu quả:** Giảm 87.5% computational cost so với full sequence attention

## Quantum Attention Integration

### 🔬 **QuantumAttention trong QCAAPatchTF:**

#### **Hybrid Attention Strategy:**
```python
# Alternating quantum and classical attention
for i in range(configs.e_layers):
    if i % 2 == 0 and self.use_quantum_attention:
        attention = QuantumAttention(
            num_qubits=4,
            entanglement_factor=0.5,
            attention_dropout=configs.dropout
        )
    else:
        attention = FullAttention(
            mask_flag=False,
            attention_dropout=configs.dropout
        )
```

#### **Quantum Circuit Design:**
```python
def variational_circuit(self, params):
    # Parameterized rotation gates
    for i in range(self.num_qubits):
        qml.RY(params[i], wires=i)
    
    # Entanglement layer
    for i in range(self.num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    return qml.expval(qml.PauliZ(0))
```

**Status:** Experimental feature cho research, không sử dụng trong production Phase 1

## Multi-Market Forecasting

### 🌍 **Supply Chain Multi-Market Strategy:**

#### **Market Configuration:**
- **Europe Market**: Market ID = 0, Weight = 0.35
- **LATAM Market**: Market ID = 1, Weight = 0.30  
- **USCA Market**: Market ID = 2, Weight = 0.35

#### **Kiến trúc Output:**
```python
# Output mô hình: [batch_size, pred_len, n_markets]
# Ví dụ: [32, 1, 3] cho batch=32, dự báo 1 ngày, 3 thị trường

# Dự báo theo thị trường cụ thể
europe_pred = output[:, :, 0]   # Dự báo Europe
latam_pred = output[:, :, 1]    # Dự báo LATAM
usca_pred = output[:, :, 2]     # Dự báo USCA
```

#### **Tính toán Loss với Trọng số Thị trường:**
```python
# Trong experiment class
criterion = WeightedMSELoss(market_weights=[0.35, 0.30, 0.35])
loss = criterion(predictions, targets)  # Tự động weighted theo thị trường
```

## Sử dụng Mô hình và Cấu hình

### 🛠️ **Thiết lập Training:**

#### **Cấu hình Tốt nhất Phase 1:**
```python
# Lựa chọn mô hình
args.model = 'QCAAPatchTF_Embedding'
args.data = 'SupplyChainEmbedding'

# Tham số kiến trúc
args.d_model = 512
args.n_heads = 8  
args.e_layers = 6
args.d_ff = 2048
args.dropout = 0.1
args.activation = 'gelu'

# Tham số nhiệm vụ
args.seq_len = 7
args.pred_len = 1
args.enc_in = 21
args.c_out = 3

# Tham số training
args.batch_size = 32
args.learning_rate = 0.0001
args.train_epochs = 10
```

#### **Khởi tạo Mô hình:**
```python
from models.QCAAPatchTF_Embedding import QCAAPatchTF_Embedding

# Khởi tạo mô hình
model = QCAAPatchTF_Embedding(configs)

# Forward pass
outputs = model(
    x_enc=input_data,           # [batch, seq_len, features]
    x_mark_enc=time_features,   # [batch, seq_len, time_dims]  
    x_dec=decoder_input,        # [batch, label_len + pred_len, features]
    x_mark_dec=decoder_time,    # [batch, label_len + pred_len, time_dims]
    categorical_features=cat_data  # Dict chứa categorical indices
)
# Output: [batch, pred_len, n_markets]
```

### 🔧 **Gỡ lỗi và Giám sát:**

#### **Thông tin Mô hình:**
```python
# Kiểm tra số parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tổng parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Kiểm tra kiến trúc mô hình
print(model)

# Kiểm tra kích thước output
with torch.no_grad():
    sample_output = model(sample_input)
    print(f"Kích thước output: {sample_output.shape}")
```

#### **Vấn đề Thường gặp và Giải pháp:**
```python
# Lỗi memory
if torch.cuda.is_available():
    model = model.cuda()
    # Sử dụng gradient checkpointing
    model.gradient_checkpointing = True

# Giá trị NaN
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Training chậm
# Sử dụng mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## Kết quả Phase 1

### 📊 **Performance Achievements:**

#### **QCAAPatchTF_Embedding Results:**
- **MSE Improvement**: 98.71% so với baseline
- **Model Size**: ~15M parameters
- **Training Time**: ~15 phút/epoch trên single GPU
- **Inference Speed**: ~2ms per batch
- **Memory Usage**: ~3GB peak GPU memory

#### **Architecture Benefits:**
- **Patch Efficiency**: 87.5% computation reduction
- **Multi-market Capability**: Simultaneous 3-market predictions
- **Embedding Integration**: Categorical feature support
- **Robust Training**: Stable convergence với early stopping

### 🎯 **Business Impact:**
- **Accuracy**: 91.78% forecasting accuracy
- **Multi-market Support**: Europe/LATAM/USCA simultaneous predictions  
- **Real-time Capability**: Sub-second inference time
- **Scalability**: Memory efficient cho production deployment

## Hướng phát triển Phase 2

### 🚀 **Cải tiến Mô hình:**

#### 1. **Cải thiện Kiến trúc:**
- **Attention Hiệu quả**: Các biến thể Linformer, Performer
- **Patches Phân cấp**: Patch embedding đa tầng
- **Kiến trúc Động**: Độ sâu/rộng mô hình thích ứng
- **Tối ưu Bộ nhớ**: Gradient checkpointing, chia sẻ mô hình

#### 2. **Tính năng Nâng cao:**
- **Tích hợp Dữ liệu Ngoài**: Thời tiết, chỉ số kinh tế
- **Attention Liên thị trường**: Mô hình phụ thuộc thị trường rõ ràng
- **Định lượng Không chắc chắn**: Dự báo xác suất
- **Dự báo Đa kỳ hạn**: Dự báo độ dài thay đổi

#### 3. **Tính năng Sản xuất:**
- **Phục vụ Mô hình**: Xuất ONNX, tối ưu TensorRT
- **Kiểm thử A/B**: Nhiều biến thể mô hình
- **Học Trực tuyến**: Cập nhật mô hình tăng dần
- **Giám sát**: Theo dõi hiệu suất, phát hiện drift

### 🔬 **Hướng Nghiên cứu:**
- **Tích hợp Quantum**: Thí nghiệm phần cứng quantum thực
- **Mô hình Nền tảng**: Mô hình supply chain được pre-train
- **Mô hình Nhân quả**: Khám phá nhân quả với attention mechanisms
- **Học Liên kết**: Training mô hình liên tổ chức

## Thực hành Tốt nhất

### ✅ **Hướng dẫn Lựa chọn Mô hình:**

#### **Sử dụng QCAAPatchTF khi:**
- Thí nghiệm baseline và proof-of-concept
- Tài nguyên tính toán hạn chế
- Nhiệm vụ dự báo đơn giản
- Nghiên cứu với quantum attention

#### **Sử dụng QCAAPatchTF_Embedding khi:**
- Triển khai sản xuất (được khuyến nghị)
- Dự báo đa thị trường
- Có categorical features
- Cần tối ưu hiệu suất

#### **Sử dụng MultiRegionQCAAPatchTF khi:**
- Yêu cầu mô hình hóa theo vùng cụ thể
- Nghiên cứu thử nghiệm Phase 2
- Cần tham số vùng tùy chỉnh

### ⚠️ **Lỗi Thường gặp và Khắc phục:**

#### **Vấn đề Training:**
- **Overfitting**: Sử dụng dropout và early stopping
- **Gradient Explosion**: Clip gradients với max_norm=1.0
- **Tràn Bộ nhớ**: Giảm batch size hoặc sequence length

#### **Vấn đề Kiến trúc:**
- **Kích thước Patch Sai**: Đảm bảo seq_len % patch_len == 0
- **Sai kích thước**: Xác minh embedding dimensions
- **Cấu hình Thị trường**: Kiểm tra c_out khớp với số thị trường

## 📋 **Tổng kết**

Các kiến trúc mô hình trong folder này đại diện cho công nghệ tiên tiến trong dự báo supply chain, với sự cân bằng tốt giữa độ chính xác và hiệu quả tính toán. QCAAPatchTF_Embedding đã chứng minh hiệu quả vượt trội trong Phase 1 với cải tiến 98.71% MSE và đạt 91.78% độ chính xác, sẵn sàng cho triển khai sản xuất hoặc phát triển Phase 2.
