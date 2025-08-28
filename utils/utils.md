# 🛠️ **Utils - Thư viện Hỗ trợ QTransformer**

## 📋 **Tổng quan Folder Utils**

Folder `utils/` chứa các công cụ và hàm hỗ trợ thiết yếu cho QTransformer project. Đây là collection của các utility functions được sử dụng xuyên suốt trong quá trình training, evaluation và preprocessing.

### **🗂️ Cấu trúc Files:**
```
utils/
├── __init__.py                 # Package initialization
├── ADFtest.py                  # Kiểm tra tính dừng của chuỗi thời gian
├── augmentation.py             # Data augmentation cho time series
├── dtw.py                      # Dynamic Time Warping implementation
├── dtw_metric.py              # DTW metrics và distance calculation
├── losses.py                   # Custom loss functions
├── m4_summary.py              # M4 dataset evaluation tools
├── masking.py                 # Attention masking utilities
├── metrics.py                 # Evaluation metrics
├── print_args.py              # Arguments printing và debugging
├── timefeatures.py            # Time-based feature engineering
└── tools.py                   # Training utilities và helpers
```

---

## 📊 **Chi tiết từng File**

### **1. ADFtest.py - Kiểm tra Tính dừng**

#### **🎯 Mục đích:**
Thực hiện Augmented Dickey-Fuller test để kiểm tra tính dừng (stationarity) của chuỗi thời gian. Đây là bước quan trọng trong phân tích time series.

#### **📋 Chức năng chính:**
```python
# Functions có sẵn:
calculate_ADF(root_path, data_path)           # ADF test cho tất cả columns
calculate_target_ADF(root_path, data_path, target)  # ADF test cho target specific
archADF(root_path, data_path)                # Average ADF statistic
```

#### **💡 Cách sử dụng:**
```python
from utils.ADFtest import calculate_target_ADF

# Kiểm tra tính dừng của order_count
adf_result = calculate_target_ADF(
    root_path="./dataset/", 
    data_path="supply_chain_optimized.csv",
    target="order_count"
)
```

#### **⚙️ Tham số quan trọng:**
- **maxlag=1**: Sử dụng 1 lag để test
- **Kết quả**: Array chứa ADF statistics và p-values
- **Ý nghĩa**: Statistic < -3.5 thường indicate stationarity

---

### **2. augmentation.py - Data Augmentation**

#### **🎯 Mục đích:**
Cung cấp các techniques để tăng cường dữ liệu time series, giúp model generalize tốt hơn và tránh overfitting.

#### **📋 Chức năng chính:**
```python
# Các augmentation methods:
jitter(x, sigma=0.03)                    # Thêm noise ngẫu nhiên
scaling(x, sigma=0.1)                    # Scale dữ liệu random
rotation(x)                              # Xoay và flip data
permutation(x, max_segments=5)           # Hoán vị segments
magnitude_warp(x, sigma=0.2, knot=4)     # Warp magnitude với splines
```

#### **💡 Cách sử dụng:**
```python
from utils.augmentation import jitter, scaling

# Áp dụng augmentation cho training data
augmented_data = jitter(original_data, sigma=0.05)
scaled_data = scaling(original_data, sigma=0.15)
```

#### **⚙️ Tham số điều chỉnh:**
- **sigma**: Độ mạnh của augmentation (0.01-0.2)
- **max_segments**: Số segments để permutation (1-10)
- **knot**: Số control points cho warping (2-8)

---

### **3. dtw.py - Dynamic Time Warping**

#### **🎯 Mục đích:**
Implementation của DTW algorithm để so sánh độ tương tự giữa các chuỗi thời gian có độ dài khác nhau.

#### **📋 Chức năng chính:**
```python
# Core DTW functions:
dtw(prototype, sample, return_flag, slope_constraint, window)
_traceback(DTW, slope_constraint)         # Tìm optimal path
```

#### **💡 Cách sử dụng:**
```python
from utils.dtw import dtw, RETURN_VALUE

# So sánh 2 time series
distance = dtw(series1, series2, return_flag=RETURN_VALUE)
```

#### **⚙️ Cấu hình DTW:**
- **slope_constraint**: "symmetric" hoặc "asymmetric"
- **window**: Window size để giới hạn warping
- **return_flag**: RETURN_VALUE, RETURN_PATH, hoặc RETURN_ALL

---

### **4. dtw_metric.py - DTW Distance Metrics**

#### **🎯 Mục đích:**
Optimized DTW implementation với các distance metrics khác nhau và warping constraints.

#### **📋 Chức năng chính:**
```python
# Advanced DTW với parameters:
dtw(x, y, dist, warp=1, w=inf, s=1.0)
```

#### **💡 Cách sử dụng:**
```python
from utils.dtw_metric import dtw
from scipy.spatial.distance import euclidean

# DTW với custom distance
distance = dtw(series1, series2, dist=euclidean, warp=2, w=10)
```

#### **⚙️ Tham số nâng cao:**
- **warp**: Số shifts được tính toán (1-5)
- **w**: Window size constraint
- **s**: Weight cho off-diagonal moves (0.5-2.0)

---

### **5. losses.py - Custom Loss Functions**

#### **🎯 Mục đích:**
Định nghĩa các loss functions đặc biệt cho time series forecasting, đặc biệt là MAPE loss.

#### **📋 Chức năng chính:**
```python
# Loss functions có sẵn:
class mape_loss(nn.Module)               # Mean Absolute Percentage Error
divide_no_nan(a, b)                      # Safe division avoiding NaN
```

#### **💡 Cách sử dụng:**
```python
from utils.losses import mape_loss

# Sử dụng MAPE loss trong training
criterion = mape_loss()
loss = criterion(insample, freq, forecast, target, mask)
```

#### **⚙️ Đặc điểm MAPE:**
- **Input**: insample, freq, forecast, target, mask
- **Output**: Percentage error không bị ảnh hưởng bởi NaN
- **Ứng dụng**: Forecasting với different scales

---

### **6. m4_summary.py - M4 Evaluation**

#### **🎯 Mục đích:**
Tools để evaluation model trên M4 dataset format, cung cấp standardized metrics cho time series forecasting.

#### **📋 Chức năng chính:**
```python
# M4 evaluation metrics:
mase(forecast, insample, outsample, frequency)  # Mean Absolute Scaled Error
smape_2(forecast, target)                       # Symmetric MAPE
mape(forecast, target)                          # Mean Absolute Percentage Error
class M4Summary                                 # Complete evaluation suite
```

#### **💡 Cách sử dụng:**
```python
from utils.m4_summary import mase, smape_2

# Evaluate forecast quality
mase_score = mase(predictions, insample_data, target, frequency=7)
smape_score = smape_2(predictions, target)
```

#### **⚙️ M4 Metrics:**
- **MASE**: Scale-independent error measure
- **sMAPE**: Symmetric percentage error (0-200%)
- **Frequency**: Seasonality cho MASE calculation

---

### **7. masking.py - Attention Masking**

#### **🎯 Mục đích:**
Cung cấp attention masks cho Transformer models, đảm bảo causal relationships và probability-based attention.

#### **📋 Chức năng chính:**
```python
# Masking classes:
class TriangularCausalMask()             # Causal masking cho decoder
class ProbMask()                         # Probability-based masking
```

#### **💡 Cách sử dụng:**
```python
from utils.masking import TriangularCausalMask

# Tạo causal mask cho attention
mask = TriangularCausalMask(B=32, L=336, device="cuda")
attention_output = attention_layer(query, key, value, attn_mask=mask.mask)
```

#### **⚙️ Mask Configuration:**
- **B**: Batch size
- **L**: Sequence length  
- **H**: Number of heads (cho ProbMask)
- **device**: "cpu" hoặc "cuda"

---

### **8. metrics.py - Evaluation Metrics**

#### **🎯 Mục đích:**
Collection of standard evaluation metrics cho regression và forecasting tasks.

#### **📋 Chức năng chính:**
```python
# Standard metrics:
RSE(pred, true)                          # Root Squared Error
CORR(pred, true)                         # Correlation coefficient
MAE(pred, true)                          # Mean Absolute Error
MSE(pred, true)                          # Mean Squared Error
RMSE(pred, true)                         # Root Mean Squared Error
MAPE(pred, true)                         # Mean Absolute Percentage Error
MSPE(pred, true)                         # Mean Squared Percentage Error
metric(pred, true)                       # All metrics together
```

#### **💡 Cách sử dụng:**
```python
from utils.metrics import metric, MAE, MSE

# Đánh giá model performance
mae, mse, rmse, mape, mspe = metric(predictions, ground_truth)

# Hoặc từng metric riêng
mae_score = MAE(predictions, ground_truth)
```

#### **⚙️ Metrics Giải thích:**
- **MAE**: Sai số tuyệt đối trung bình
- **MSE**: Sai số bình phương trung bình (Phase 1: 98.71% improvement)
- **RMSE**: Căn MSE, cùng đơn vị với data
- **MAPE**: Percentage error, scale-independent

---

### **9. print_args.py - Arguments Display**

#### **🎯 Mục đích:**
Hiển thị arguments và configuration của experiment một cách có tổ chức, giúp debugging và tracking.

#### **📋 Chức năng chính:**
```python
# Utility function:
print_args(args)                         # Format và print tất cả arguments
```

#### **💡 Cách sử dụng:**
```python
from utils.print_args import print_args

# Hiển thị config đầy đủ
print_args(args)
```

#### **⚙️ Sections được hiển thị:**
- **Basic Config**: Task name, model info
- **Data Loader**: Paths, features, target
- **Forecasting Task**: Sequence lengths, patterns
- **Model Parameters**: Architecture details
- **Run Parameters**: Training configuration

---

### **10. timefeatures.py - Time Feature Engineering**

#### **🎯 Mục đích:**
Tạo time-based features từ datetime index, giúp model hiểu được temporal patterns.

#### **📋 Chức năng chính:**
```python
# Time feature classes:
class SecondOfMinute(TimeFeature)        # Second trong minute [-0.5, 0.5]
class MinuteOfHour(TimeFeature)          # Minute trong hour [-0.5, 0.5]
class HourOfDay(TimeFeature)             # Hour trong day [-0.5, 0.5]
# Và nhiều classes khác cho day, week, month, year
```

#### **💡 Cách sử dụng:**
```python
from utils.timefeatures import HourOfDay, DayOfWeek
import pandas as pd

# Tạo time features
dates = pd.date_range('2024-01-01', periods=100, freq='H')
hour_feature = HourOfDay()(dates)
```

#### **⚙️ Feature Encoding:**
- **Range**: Tất cả features được normalize về [-0.5, 0.5]
- **Cyclic**: Capture tính chu kỳ của time
- **Multiple scales**: Second, minute, hour, day, week, month, year

---

### **11. tools.py - Training Utilities**

#### **🎯 Mục đích:**
Các công cụ hỗ trợ training process như learning rate scheduling, early stopping, và visualization.

#### **📋 Chức năng chính:**
```python
# Training utilities:
adjust_learning_rate(optimizer, epoch, args)   # LR scheduling
class EarlyStopping()                          # Early stopping mechanism
# Visualization và other helper functions
```

#### **💡 Cách sử dụng:**
```python
from utils.tools import adjust_learning_rate, EarlyStopping

# Setup early stopping
early_stopping = EarlyStopping(patience=10, verbose=True)

# Training loop
for epoch in range(train_epochs):
    adjust_learning_rate(optimizer, epoch, args)
    # ... training code ...
    early_stopping(val_loss, model, checkpoint_path)
    if early_stopping.early_stop:
        break
```

#### **⚙️ LR Scheduling Types:**
- **type1**: Exponential decay (0.5^epochs)
- **type2**: Step-wise predefined rates
- **cosine**: Cosine annealing schedule

---

## 🔧 **Hướng dẫn Sử dụng Utils**

### **1. Import Pattern:**
```python
# Standard imports cho QTransformer
from utils.metrics import metric, MAE, MSE
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.masking import TriangularCausalMask
from utils.timefeatures import time_features
```

### **2. Training Integration:**
```python
# Setup trong training script
early_stopping = EarlyStopping(patience=args.patience)
criterion = nn.MSELoss()  # Hoặc custom loss từ utils.losses

# Training loop với utils
for epoch in range(args.train_epochs):
    adjust_learning_rate(optimizer, epoch, args)
    
    # Training phase
    train_loss = train_epoch()
    
    # Validation phase  
    val_loss = validate_epoch()
    mae, mse, rmse, mape, mspe = metric(predictions, targets)
    
    # Early stopping check
    early_stopping(val_loss, model, checkpoint_path)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

### **3. Evaluation Workflow:**
```python
# Comprehensive evaluation
from utils.metrics import metric
from utils.ADFtest import calculate_target_ADF

# Model evaluation
predictions = model.predict(test_data)
mae, mse, rmse, mape, mspe = metric(predictions, targets)

# Data analysis
adf_stats = calculate_target_ADF(root_path, data_path, "order_count")
print(f"ADF statistic: {adf_stats[0][0]:.4f}")
```

---

## 📊 **Utils trong QTransformer Pipeline**

### **Data Flow với Utils:**
```
1. Raw Time Series Data
         ↓
2. ADFtest.py (stationarity check)
         ↓
3. timefeatures.py (temporal features)
         ↓
4. augmentation.py (data augmentation)
         ↓
5. Training với tools.py (LR scheduling, early stopping)
         ↓
6. masking.py (attention masks)
         ↓
7. losses.py (custom loss functions)
         ↓
8. metrics.py (evaluation)
         ↓
9. Final Performance Assessment
```

### **Phase 1 Utils Usage:**
- ✅ **metrics.py**: 98.71% MSE improvement tracking
- ✅ **tools.py**: Early stopping với patience=10
- ✅ **masking.py**: Causal attention cho QCAAPatchTF
- ✅ **timefeatures.py**: Daily frequency features
- ✅ **losses.py**: WeightedMSELoss cho multi-market

---

## 🚀 **Phase 2 Enhancements**

### **Planned Utils Improvements:**
1. **Advanced Augmentation**: Thêm GAN-based augmentation
2. **Custom Metrics**: Metrics đặc biệt cho supply chain
3. **Distributed Tools**: Support cho multi-GPU training
4. **Monitoring Utils**: Real-time performance tracking

### **New Utils Modules:**
```python
# Potential additions
utils/
├── quantum_metrics.py         # Quantum-inspired evaluation
├── supply_chain_losses.py     # Domain-specific losses  
├── distributed_tools.py       # Multi-GPU utilities
└── monitoring.py              # Real-time tracking
```

---

## 📋 **Tổng kết**

Folder `utils/` là backbone của QTransformer project, cung cấp:
- ✅ **Evaluation tools** hoàn chỉnh cho time series forecasting
- ✅ **Training utilities** giúp stable và efficient training
- ✅ **Data processing** tools cho time series analysis
- ✅ **Visualization và debugging** support
- ✅ **Extensible architecture** cho Phase 2 development

**🎯 Lưu ý quan trọng**: Utils được thiết kế modular và reusable, có thể dễ dàng extend cho các features mới trong Phase 2 mà không ảnh hưởng đến existing functionality.
