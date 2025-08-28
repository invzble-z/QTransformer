# 🔧 TRAINING FLOW - HƯỚNG DẪN CHI TIẾT QUÁ TRÌNH HUẤN LUYỆN

## 🎯 **TỔNG QUAN LUỒNG HUẤN LUYỆN**

Tài liệu này mô tả chi tiết từng bước trong quá trình huấn luyện mô hình QTransformer, từ khi bắt đầu chạy script cho đến khi tạo ra kết quả cuối cùng.

### **Điểm khởi đầu**
```bash
# Cách khởi chạy huấn luyện
./scripts/long_term_forecast/QCAAPatchTF_SupplyChain_Embedding.sh
```

---

## 📋 **BƯỚC 1: KHỞI TẠO VÀ THIẾT LẬP**

### **1.1 Entry Point - File `run.py`**
- **Vị trí**: `/run.py` (thư mục gốc)
- **Vai trò**: Entry point chính của toàn bộ hệ thống
- **Chức năng chính**:
  ```python
  # Thiết lập seed cho reproducibility
  fix_seed = 2021
  random.seed(fix_seed)
  torch.manual_seed(fix_seed)
  np.random.seed(fix_seed)
  
  # Parsing arguments từ script
  parser = argparse.ArgumentParser()
  # ... định nghĩa các tham số
  ```

### **1.2 Xác định loại Experiment**
```python
# Logic chọn experiment class (dòng 150-154 trong run.py)
if args.task_name == 'long_term_forecast':
    if args.data in ['SupplyChainEmbedding', 'MultiRegionEmbedding'] or args.model in ['QCAAPatchTF_Embedding']:
        Exp = Exp_Long_Term_Forecast_Embedding  # ← Chọn class này cho project
    else:
        Exp = Exp_Long_Term_Forecast
```

### **1.3 Tạo setting name**
```python
# Tạo tên unique cho experiment (dòng 160-175)
setting = 'long_term_forecast_SupplyChain_Optimized_Phase1_v1_QCAAPatchTF_Embedding_...'
# Format: {task}_{model_id}_{model}_{data}_ft{features}_sl{seq_len}_...
```

---

## 📊 **BƯỚC 2: KHỞI TẠO EXPERIMENT CLASS**

### **2.1 Class `Exp_Long_Term_Forecast_Embedding`**
- **Vị trí**: `/exp/exp_long_term_forecasting_embedding.py`
- **Kế thừa**: `Exp_Basic` từ `/exp/exp_basic.py`
- **Chức năng khởi tạo**:
  ```python
  def __init__(self, args):
      super().__init__(args)
      
      # Load target scaler cho denormalization
      self.target_scaler = None
      scaler_path = './scalers/target_scaler.pkl'
      if Path(scaler_path).exists():
          self.target_scaler = joblib.load(scaler_path)
  ```

### **2.2 Thiết lập các thành phần chính**
1. **Model**: Gọi `_build_model()` → tạo instance của `QCAAPatchTF_Embedding`
2. **Optimizer**: Gọi `_select_optimizer()` → Adam optimizer
3. **Loss function**: Gọi `_select_criterion()` → `WeightedMSELoss` cho multi-market
4. **Data loaders**: Gọi `_get_data()` cho train/validation/test

---

## 🏗️ **BƯỚC 3: THIẾT LẬP DỮ LIỆU**

### **3.1 Data Provider Factory**
- **Vị trí**: `/data_provider/data_factory.py`
- **Chức năng**: Tạo dataset và dataloader
```python
def data_provider(args, flag):
    # flag có thể là 'train', 'val', hoặc 'test'
    Data = data_dict[args.data]  # SupplyChainEmbedding
    
    # Tạo dataset
    data_set = Data(args=args, root_path=args.root_path, ...)
    
    # Tạo dataloader với custom collate function
    collate_fn = embedding_collate_fn  # Xử lý categorical features
    data_loader = DataLoader(data_set, batch_size=args.batch_size, ...)
```

### **3.2 Dataset Loading**
- **Vị trí**: `/data_provider/data_loader_embedding.py`
- **Class**: `Dataset_SupplyChain_Embedding`
- **Chức năng**: 
  - Load file CSV từ `./dataset/supply_chain_optimized.csv`
  - Chuẩn bị sequences với length = 21 ngày
  - Tạo categorical features cho embedding
  - Normalization với StandardScaler

### **3.3 Cấu trúc dữ liệu trả về**
```python
# Mỗi batch gồm 5 thành phần:
batch_x,           # [batch_size, seq_len, features] = [16, 21, 51]
batch_y,           # [batch_size, label_len+pred_len, c_out] = [16, 7, 3]  
batch_x_mark,      # [batch_size, seq_len, time_features]
batch_y_mark,      # [batch_size, label_len+pred_len, time_features]
categorical_features  # Dict với các embedding features
```

---

## 🚀 **BƯỚC 4: QUÁ TRÌNH HUẤN LUYỆN**

### **4.1 Main Training Loop**
- **Vị trí**: `/exp/exp_long_term_forecasting_embedding.py` method `train()`
- **Cấu trúc chính**:
```python
def train(self, setting):
    # 1. Lấy data loaders
    train_data, train_loader = self._get_data(flag='train')
    vali_data, vali_loader = self._get_data(flag='val')
    test_data, test_loader = self._get_data(flag='test')
    
    # 2. Thiết lập early stopping
    path = './checkpoints/' + setting
    early_stopping = EarlyStopping(patience=args.patience)
    
    # 3. Training loop chính
    for epoch in range(args.train_epochs):
        # Training phase
        # Validation phase  
        # Early stopping check
```

### **4.2 Training Phase Chi Tiết**
```python
# Mỗi epoch gồm các bước:
for i, batch_data in enumerate(train_loader):
    # 1. Unpack dữ liệu
    batch_x, batch_y, batch_x_mark, batch_y_mark, categorical_features = batch_data
    
    # 2. Chuyển lên GPU
    batch_x = batch_x.float().to(self.device)
    # ... các tensor khác
    
    # 3. Chuẩn bị decoder input
    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
    
    # 4. Forward pass
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
    
    # 5. Tính loss
    f_dim = -1 if self.args.features == 'MS' else 0
    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # Lấy phần prediction
    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]   # Ground truth
    loss = criterion(outputs, batch_y)  # WeightedMSELoss
    
    # 6. Backward pass
    loss.backward()
    model_optim.step()
```

### **4.3 Weighted Loss Function**
```python
class WeightedMSELoss(nn.Module):
    """Trọng số cho 3 thị trường: Europe=0.35, LATAM=0.30, USCA=0.35"""
    def forward(self, predictions, targets):
        # predictions: [batch, 7_days, 3_markets]
        # targets: [batch, 7_days, 3_markets]
        
        mse_per_market = torch.mean((predictions - targets) ** 2, dim=(0, 1))
        weights = torch.tensor([0.35, 0.30, 0.35])  # Market weights
        weighted_loss = torch.sum(weights * mse_per_market)
        return weighted_loss
```

---

## 📈 **BƯỚC 5: VALIDATION VÀ EARLY STOPPING**

### **5.1 Validation Process**
```python
def vali(self, vali_data, vali_loader, criterion):
    total_loss = []
    self.model.eval()  # Chuyển sang evaluation mode
    
    with torch.no_grad():  # Không tính gradient
        for batch_data in vali_loader:
            # Tương tự training nhưng không backward
            outputs = self.model(...)
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
    
    return np.average(total_loss)
```

### **5.2 Early Stopping Logic**
```python
# Sau mỗi epoch:
vali_loss = self.vali(vali_data, vali_loader, criterion)
early_stopping(vali_loss, self.model, path)

if early_stopping.early_stop:
    print("Early stopping")
    break  # Dừng training

# Lưu best model tại: ./checkpoints/{setting}/checkpoint.pth
```

---

## 🔍 **BƯỚC 6: TESTING VÀ ĐÁNH GIÁ**

### **6.1 Testing Process**
- **Method**: `test()` trong `Exp_Long_Term_Forecast_Embedding`
- **Input**: Load best model từ checkpoint
- **Output**: Predictions và ground truth

```python
def test(self, setting, test=0):
    # 1. Load best model
    self.model.load_state_dict(torch.load(checkpoint_path))
    
    # 2. Run inference
    preds = []
    trues = []
    for batch_data in test_loader:
        outputs = self.model(...)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(batch_y.detach().cpu().numpy())
    
    # 3. Concatenate results
    preds = np.concatenate(preds, axis=0)  # [70_samples, 7_days, 3_markets]
    trues = np.concatenate(trues, axis=0)
```

### **6.2 Metrics Calculation với Denormalization**
```python
def _calculate_corrected_metrics(self, preds, trues):
    # 1. Load target scaler
    target_scaler = joblib.load('./scalers/target_scaler.pkl')
    
    # 2. Denormalize predictions và ground truth
    preds_denorm = target_scaler.inverse_transform(preds.reshape(-1, 1))
    trues_denorm = target_scaler.inverse_transform(trues.reshape(-1, 1))
    
    # 3. Tính toán metrics thực tế
    mae = np.mean(np.abs(preds_denorm - trues_denorm))     # 13.75 orders
    mse = np.mean((preds_denorm - trues_denorm) ** 2)     # 262.82
    mape = np.mean(np.abs((trues_denorm - preds_denorm) / trues_denorm)) * 100  # 8.22%
```

---

## 📁 **BƯỚC 7: LƯU TRỮ KẾT QUẢ**

### **7.1 Cấu trúc thư mục output**
```
./test_results/long_term_forecast_SupplyChain_Optimized_Phase1_v1_QCAAPatchTF_Embedding_.../
├── metrics.npy              # Metrics gốc [MAE, MSE, RMSE, MAPE, MSPE]
├── pred.npy                 # Predictions [70, 7, 3]
├── true.npy                 # Ground truth [70, 7, 3]
├── PHASE1_FINAL_SUMMARY.md  # Báo cáo tổng kết
├── PHASE1_DETAILED_ANALYSIS_REPORT.md  # Phân tích chi tiết
└── phase1_daily_performance.csv  # Hiệu suất theo ngày
```

### **7.2 Files checkpoint**
```
./checkpoints/long_term_forecast_SupplyChain_Optimized_Phase1_v1_.../
└── checkpoint.pth           # Best model weights
```

### **7.3 Logs training**
```
./logs/
└── training_{timestamp}.log # Chi tiết quá trình training
```

---

## 🔧 **THÀNH PHẦN KỸ THUẬT CHI TIẾT**

### **Model Architecture Flow**
1. **Input Processing**: 
   - Continuous features: `[batch, 21, 51]` → Embedding layers
   - Categorical features: Embedding tables → concat với continuous
   
2. **Transformer Encoder**:
   - MultiHead Attention với `n_heads=8`
   - Feed Forward Network với `d_ff=256`
   - Layer Normalization + Residual connections
   
3. **Decoder**:
   - Cross-attention với encoder outputs
   - Masked self-attention cho prediction sequence
   
4. **Output Projection**:
   - Linear layer: `d_model=64` → `c_out=3` (3 markets)
   - Shape: `[batch, 7, 3]` (7 ngày, 3 thị trường)

### **Memory và Performance**
- **GPU Memory**: ~2-4GB cho batch_size=16
- **Training Time**: ~3-5 phút cho 15 epochs
- **Convergence**: Early stopping thường xảy ra ở epoch 5-7

### **Key Parameters**
```python
seq_len=21      # Input sequence length (3 tuần)
pred_len=7      # Prediction length (1 tuần)  
enc_in=51       # Input features (đã engineered)
c_out=3         # Output markets (Europe, LATAM, USCA)
d_model=64      # Model dimension
n_heads=8       # Attention heads
e_layers=3      # Encoder layers
batch_size=16   # Training batch size
```

---

## 🎯 **TÓNG TẮT LUỒNG HOÀN CHỈNH**

1. **Script khởi chạy** → `run.py` 
2. **Parse arguments** → Tạo `Exp_Long_Term_Forecast_Embedding`
3. **Load data** → `data_factory.py` → `Dataset_SupplyChain_Embedding`
4. **Build model** → `QCAAPatchTF_Embedding` architecture
5. **Training loop** → Forward/backward pass với `WeightedMSELoss`
6. **Validation** → Early stopping monitoring
7. **Testing** → Best model inference
8. **Metrics** → Denormalized evaluation
9. **Save results** → Files trong `./test_results/`

### **Thời gian thực hiện**
- **Setup**: 10-30 giây
- **Training**: 2-5 phút (5-15 epochs)
- **Testing**: 10-20 giây
- **Total**: ~3-6 phút

### **Command để chạy**
```bash
# Từ thư mục gốc
./scripts/long_term_forecast/QCAAPatchTF_SupplyChain_Embedding.sh

# Hoặc trực tiếp
python run.py --task_name long_term_forecast --is_training 1 --model QCAAPatchTF_Embedding --data SupplyChainEmbedding [các args khác...]
```

---

*Tài liệu này mô tả chi tiết luồng training cho Phase 1. Để hiểu sâu hơn về từng component, tham khảo code trong các thư mục tương ứng.*
