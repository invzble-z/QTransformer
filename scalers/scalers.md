# 📊 **Scalers - Chuẩn hóa Dữ liệu QTransformer**

## 📋 **Tổng quan Folder Scalers**

Folder `scalers/` chứa các fitted scalers và metadata được sử dụng để chuẩn hóa và denormalize dữ liệu trong QTransformer project. Đây là thành phần quan trọng cho việc training và inference của mô hình.

### **🗂️ Cấu trúc Files:**
```
scalers/
├── target_scaler.pkl       # StandardScaler cho target variable (order_count)
└── optimization_metadata.pkl  # Metadata tối ưu hóa và thông tin preprocessing
```

---

## 🎯 **Target Scaler (target_scaler.pkl)**

### **Mô tả:**
- **Loại**: `sklearn.preprocessing.StandardScaler`
- **Mục đích**: Chuẩn hóa target variable `order_count` cho training
- **Tầm quan trọng**: **CỰC KỲ QUAN TRỌNG** - Cần thiết để convert predictions về giá trị thực

### **Thông số Kỹ thuật:**
```python
# Scaler Statistics (từ Phase 1)
Mean (μ): 170.93
Standard Deviation (σ): 14.05
Formula: (x - μ) / σ
```

### **Quy trình Tạo Scaler:**
```python
# 1. Khởi tạo (trong data_optimization_preprocessing.ipynb)
from sklearn.preprocessing import StandardScaler
target_scaler = StandardScaler()

# 2. Fit với toàn bộ order_count data
target_scaler.fit(df[['order_count']])

# 3. Transform cho training
df['order_count_normalized'] = target_scaler.transform(df[['order_count']])

# 4. Lưu scaler
import joblib
joblib.dump(target_scaler, '../scalers/target_scaler.pkl')
```

### **Sử dụng trong Training:**
```python
# Load scaler trong experiment class
import joblib
from pathlib import Path

scaler_path = './scalers/target_scaler.pkl'
if Path(scaler_path).exists():
    target_scaler = joblib.load(scaler_path)
    print(f"🔧 Loaded target scaler from {scaler_path}")
```

### **Sử dụng cho Inference:**
```python
# Denormalize predictions
def denormalize_predictions(predictions, target_scaler):
    """
    Convert normalized predictions back to original scale
    
    Args:
        predictions: Model output [batch_size, pred_len, n_markets]
        target_scaler: Fitted StandardScaler
    
    Returns:
        denormalized_predictions: Predictions ở thang đo gốc
    """
    # Reshape cho scaler
    pred_reshaped = predictions.reshape(-1, 1)
    
    # Inverse transform
    denorm_pred = target_scaler.inverse_transform(pred_reshaped)
    
    # Reshape về format gốc
    return denorm_pred.reshape(predictions.shape)

# Sử dụng
target_scaler = joblib.load('./scalers/target_scaler.pkl')
real_predictions = denormalize_predictions(model_output, target_scaler)
```

## 📈 **Tích hợp với Pipeline**

### **Data Flow với Scalers:**
```
1. Raw Data (order_count: 150-200)
         ↓
2. StandardScaler.fit_transform()
         ↓  
3. Normalized Data (mean=0, std=1)
         ↓
4. Model Training/Inference
         ↓
5. Model Output (normalized predictions)
         ↓
6. StandardScaler.inverse_transform()
         ↓
7. Real Scale Predictions (150-200 range)
```