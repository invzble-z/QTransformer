# 🎯 Phân Tích Chiến Lược Training Model

## 📊 **Bối Cảnh Dự Án Hiện Tại**
- **Dữ liệu đã xử lý**: 765 bản ghi (255 ngày × 3 thị trường) với 21 features
- **Mục tiêu**: Dự đoán đa thị trường `[7_ngày, 3_thị_trường]`
- **Model**: QCAAPatchTF_Embedding được điều chỉnh cho đầu ra đa thị trường
- **Khoảng thời gian**: 2017-05-22 đến 2018-01-31 (dữ liệu đã đồng bộ và làm sạch)

---

## 🔄 **So Sánh Phương Pháp Training**

### **Lựa chọn 1: Luồng Script có sẵn (.sh + run.py)**

#### ✅ **Ưu điểm:**
1. **🏗️ Cơ sở hạ tầng sẵn sàng cho production**
   - Pipeline đã được thiết lập tốt với `run.py` + `.sh` scripts
   - Quản lý tham số và cấu hình chuẩn hóa
   - Hệ thống theo dõi thí nghiệm và checkpoint tích hợp sẵn

2. **🔧 Yêu cầu thay đổi code tối thiểu**
   - Chỉ cần cập nhật các tham số cấu hình trong script
   - Tái sử dụng `exp_long_term_forecasting_embedding.py` có sẵn
   - Model `QCAAPatchTF_Embedding.py` đã sẵn sàng

3. **📈 Khả năng mở rộng và tái hiện**
   - Dễ dàng điều chỉnh tham số và hyperparameter tuning
   - Ghi log thí nghiệm nhất quán
   - Giao diện command-line cho tự động hóa

4. **🎛️ Tính năng nâng cao**
   - Early stopping, điều chỉnh learning rate
   - Hỗ trợ Multi-GPU
   - Theo dõi metrics toàn diện

#### ❌ **Nhược điểm:**
1. **📝 Giới hạn phân tích tương tác**
   - Khó debug và phân tích quá trình training theo thời gian thực
   - Ít phản hồi trực quan trong quá trình training
   - Phải kiểm tra logs và files để theo dõi tiến độ

2. **🔧 Độ phức tạp cấu hình**
   - Nhiều tham số cần thiết lập chính xác
   - Khó thử nghiệm với các phương pháp tiền xử lý khác nhau
   - Ít linh hoạt cho rapid prototyping

#### 🛠️ **Thay đổi cần thiết:**
```bash
# Các thay đổi chính trong QCAAPatchTF_SupplyChain_Embedding.sh:
--data_path supply_chain_processed.csv    # ✅ Dataset đã cập nhật
--enc_in 21                               # ✅ 21 features
--c_out 3                                 # ✅ Đầu ra 3 thị trường  
--target order_count                      # ✅ Target mới
--features MS                             # ✅ Đa biến đến đa đầu ra
```

---

### **Lựa chọn 2: Luồng Training trong Jupyter Notebook (.ipynb)**

#### ✅ **Ưu điểm:**
1. **🔍 Phát triển và phân tích tương tác**
   - Trực quan hóa quá trình training theo thời gian thực
   - Debug và phân tích từng bước
   - Phản hồi và điều chỉnh ngay lập tức

2. **📊 Tích hợp trực quan phong phú**
   - Vẽ đồ thị training/validation curves
   - Trực quan hóa dự đoán trực tiếp
   - Kết hợp phân tích dữ liệu với tiền xử lý

3. **🧪 Thử nghiệm nhanh chóng**
   - Test hyperparameter nhanh
   - Điều chỉnh kiến trúc model dễ dàng
   - Pipeline tiền xử lý linh hoạt

4. **📖 Tài liệu và trình bày**
   - Câu chuyện rõ ràng từ tiền xử lý → training → đánh giá
   - Phân tích toàn diện trong một tài liệu
   - Tốt hơn cho nghiên cứu và thuyết trình

#### ❌ **Nhược điểm:**
1. **🔧 Nỗ lực phát triển đáng kể**
   - Cần implement lại training loop
   - Thiết lập theo dõi thí nghiệm thủ công
   - Tạo lại các tiện ích (early stopping, metrics, v.v.)

2. **🏗️ Giới hạn cơ sở hạ tầng**
   - Kém robust cho triển khai production
   - Checkpoint và resume thủ công
   - Khả năng mở rộng hạn chế cho thí nghiệm lớn

3. **📝 Rủi ro trùng lặp code**
   - Có thể trùng lặp chức năng hiện có
   - Chi phí bảo trì
   - Khả năng không nhất quán

#### 🛠️ **Cần implement:**
```python
# Các thành phần chính cần thực hiện:
1. Training loop với batching phù hợp
2. Logic validation và early stopping
3. Hệ thống checkpoint model
4. Tính toán và theo dõi metrics
5. Điều chỉnh learning rate
6. Tính toán loss đa thị trường
7. Tiện ích trực quan hóa
```

---

## 🎯 **KHUYẾN NGHỊ: Phương Pháp Kết Hợp**

### **Chính: Lựa chọn 1 (Script-Based) + Theo dõi nâng cao**

**Lý do:**
1. **⚡ Tốc độ đạt kết quả**: Thay đổi tối thiểu, nhanh có kết quả ban đầu
2. **🛡️ Độ ổn định đã được chứng minh**: Cơ sở hạ tầng hiện có đã test và ổn định
3. **🔄 Cải tiến lặp lại**: Bắt đầu với thiết lập cơ bản, nâng cao dần dần

**Chiến lược thực hiện:**
```bash
Giai đoạn 1: Thiết lập nhanh (1-2 giờ)
├── Cập nhật QCAAPatchTF_SupplyChain_Embedding.sh
├── Điều chỉnh data loader cho định dạng mới
├── Chạy thí nghiệm training ban đầu
└── Xác thực kết quả

Giai đoạn 2: Theo dõi nâng cao (Tùy chọn)
├── Tạo Jupyter notebook để phân tích kết quả
├── Thêm tiện ích trực quan hóa tùy chỉnh
├── Implement theo dõi metrics nâng cao
└── Tối ưu hóa hiệu suất
```

### **Phụ: Phát triển Notebook (Nâng cao tương lai)**

**Trường hợp sử dụng:**
- Phân tích training chi tiết và debug
- Quy trình thí nghiệm tùy chỉnh
- Thuyết trình nghiên cứu
- Nhu cầu trực quan hóa nâng cao

---

## 📋 **Kế Hoạch Hành Động Ngay**

### **Bước 1: Cập nhật Scripts hiện có** ⭐ (Khuyến nghị bắt đầu)
```bash
1. Điều chỉnh cấu hình data loader cho supply_chain_processed.csv
2. Cập nhật tham số model (enc_in=21, c_out=3)
3. Cấu hình hàm loss đa thị trường
4. Test pipeline training
5. Xác thực kết quả ban đầu
```

### **Bước 2: Tích hợp Data Loader**
```python
# Thay đổi cần thiết trong data_provider/data_loader_embedding.py:
- Xử lý đầu vào 21 features
- Hỗ trợ feature phân loại Market_encoded
- Định dạng target đa thị trường [batch, pred_len, 3_markets]
```

### **Bước 3: Cấu hình Model**
```python
# Điều chỉnh QCAAPatchTF_Embedding.py:
- Chiều đầu vào: 21 features
- Chiều đầu ra: 3 thị trường
- Tích hợp Market embedding
- Đầu ra multi-head cho dự đoán thị trường song song
```

---

## 🎯 **Yếu Tố Quan Trọng: Tích Hợp MLflow**

### **MLflow là gì và tại sao quan trọng?**
- **MLflow**: Nền tảng mã nguồn mở để quản lý lifecycle của machine learning
- **Chức năng chính**: Experiment tracking, model registry, deployment, reproducibility
- **Lợi ích**: Theo dõi metrics, lưu trữ models, so sánh experiments, versioning

### **📊 So Sánh Tích Hợp MLflow với 2 Flow**

#### **🏗️ Script-Based Flow + MLflow:**

**✅ Ưu điểm tích hợp:**
1. **� Tích hợp chuẩn công nghiệp**
   - MLflow được thiết kế cho command-line workflows
   - Dễ dàng thêm `mlflow.log_*()` vào training loops có sẵn
   - Tự động tracking experiments với minimal code changes

2. **📈 Experiment Management mạnh mẽ**
   - Tự động log parameters từ command-line arguments
   - Parallel experiments dễ dàng với different configs
   - Production-ready model registry integration

3. **🔄 Reproducibility tốt**
   - MLflow Projects cho packaging experiments
   - Environment management với conda/docker
   - Git integration cho version control

**❌ Thách thức:**
- Cần thêm MLflow logging code vào existing experiment classes
- Setup MLflow server/UI riêng biệt
- Learning curve cho MLflow concepts

#### **📓 Jupyter Notebook Flow + MLflow:**

**✅ Ưu điểm tích hợp:**
1. **�🔍 Interactive experiment analysis**
   - Live MLflow UI integration trong notebook
   - Immediate visualization của tracked metrics
   - Easy comparison của multiple runs

2. **🧪 Rapid experimentation**
   - Quick hyperparameter sweeps với MLflow tracking
   - Interactive model analysis và debugging
   - Flexible logging custom metrics và artifacts

**❌ Thách thức:**
- Manual experiment organization
- Khó scale cho production workflows  
- Notebook-based experiments ít structured hơn

### **🎯 Khuyến Nghị cho MLflow Integration:**

#### **Phase 1: Script-Based + Basic MLflow** ⭐ (Khuyến nghị)
```python
# Minimal MLflow integration trong exp_long_term_forecasting_embedding.py:
import mlflow
import mlflow.pytorch

def train():
    with mlflow.start_run():
        mlflow.log_params({
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'learning_rate': args.learning_rate,
            # ... other hyperparameters
        })
        
        for epoch in range(epochs):
            train_loss = train_epoch()
            val_loss = validate()
            
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss
            }, step=epoch)
        
        mlflow.pytorch.log_model(model, "model")
```

**Lý do chọn:**
- **⚡ Tích hợp nhanh**: Chỉ cần thêm vài dòng code
- **🛡️ Stability**: Existing workflow không bị disrupted  
- **📈 Scalability**: Dễ mở rộng cho multiple experiments
- **🔄 Future-proof**: Smooth transition sang advanced MLflow features

#### **Phase 2: Enhanced MLflow Features**
```python
# Advanced MLflow integration sau khi phase 1 stable:
1. MLflow Projects cho experiment packaging
2. Model Registry cho production deployment
3. MLflow UI dashboard cho team collaboration
4. Hyperparameter tuning với MLflow + Optuna
```

### **📋 MLflow Implementation Roadmap:**

#### **Giai đoạn 1: Basic Tracking (1-2 ngày)**
```bash
├── Setup MLflow server local
├── Add mlflow logging vào exp_long_term_forecasting_embedding.py
├── Log basic metrics (loss, accuracy) 
├── Save model artifacts
└── Test với single experiment
```

#### **Giai đoạn 2: Advanced Features (1 tuần)**
```bash
├── Parameter sweeps với MLflow experiments
├── Model comparison dashboard
├── Artifact management (plots, predictions)
├── Integration với existing checkpoint system
└── Team collaboration setup
```

#### **Giai đoạn 3: Production Integration (2-4 tuần)**
```bash
├── MLflow Model Registry
├── Automated model deployment
├── A/B testing framework
├── Model monitoring và drift detection
└── CI/CD pipeline integration
```

---

## 🔍 **Điểm Thảo Luận Bước Tiếp Theo**

1. **Tương thích Data Loader**: Data loader hiện có có xử lý được định dạng đã tiền xử lý không?
2. **Cấu trúc đầu ra Model**: Xác thực kiến trúc dự đoán đa thị trường
3. **Hàm Loss**: Chiến lược MSE cho từng thị trường vs. loss kết hợp
4. **Hyperparameters**: Giá trị seq_len, pred_len tối ưu cho dataset của chúng ta
5. **Metrics đánh giá**: Hiệu suất theo từng thị trường vs. hiệu suất tổng thể
6. **🆕 MLflow Setup**: Local server vs. cloud deployment cho experiment tracking

---

## 📈 **Tiêu Chí Thành Công**

### **Mục tiêu Giai đoạn 1:**
- ✅ Training thành công không có lỗi
- ✅ Loss hội tụ hợp lý
- ✅ Tạo ra dự đoán đa thị trường
- ✅ Thiết lập hiệu suất baseline
- ✅ 🆕 Basic MLflow tracking hoạt động

### **Mục tiêu Giai đoạn 2:**
- 📊 Cải thiện độ chính xác dự đoán
- 🎯 Phân tích hiệu suất theo từng thị trường
- 📈 Xác thực trực quan kết quả
- 🔧 Tối ưu hóa hyperparameters
- 📋 🆕 MLflow experiment comparison và model registry

---

## 🏆 **Kết Luận Cuối Cùng với MLflow**

### **Script-Based Flow THẮNG với MLflow:**

**Lý do chính:**
1. **🔧 MLflow được thiết kế cho production workflows** - tích hợp tự nhiên với script-based approach
2. **⚡ Faster time-to-value** - có thể setup basic tracking trong 30 phút
3. **📈 Better scalability** - dễ dàng chạy parallel experiments với different configs
4. **🛡️ Production readiness** - MLflow Model Registry và deployment features work best với structured workflows

### **Chiến Lược Triển Khai:**
```
Week 1: Script-based training + Basic MLflow tracking
Week 2: Experiment optimization + Advanced MLflow features  
Week 3: Model registry + Production deployment preparation
Week 4: Notebook-based analysis tools for deep-dive insights
```

**Khuyến nghị cuối cùng**: Bắt đầu với **Script-based flow** để có foundation vững chắc, sau đó bổ sung notebook-based analysis khi cần thiết. Approach này cân bằng giữa tốc độ đạt kết quả và tính linh hoạt cho các cải tiến tương lai.

## ---------------------------------------------------------------------------------------------------------------------

## 🔍 **Thảo Luận Chi Tiết Implementation**

### **1. 📊 Phân Tích Data Format & Compatibility**

**❓ Câu hỏi**: Data loader hiện tại có xử lý được format `supply_chain_processed.csv` không?

**🔍 Phân tích từ code hiện tại:**
- **Data loader hiện có**: `Dataset_SupplyChain_Embedding` và `Dataset_MultiRegion_Embedding`
- **Format expect**: `seller_Order_Region_processed.csv` với date column là `'order date (DateOrders)'`
- **Features handling**: Tự động phân loại categorical (có `_encoded`) và numerical features

**✅ Kết luận**: 
- **Cần tạo data loader mới** vì format hiện tại khác với `supply_chain_processed.csv`
- Cần thay đổi date column từ `'order date (DateOrders)'` → `'order_date_only'`
- Cần update target từ `'Order Item Quantity'` → `'order_count'`

**📋 Action Required**: Tạo `Dataset_SupplyChain_Processed` class mới dựa trên existing loader

---

### **2. 🎯 Phân Tích Multi-Market Output Strategy**

**❓ Câu hỏi**: Model hiện tại output như thế nào? Cần modify để output 3 markets?

**🔍 Phân tích từ QCAAPatchTF_Embedding:**
- **Current output**: `[batch, pred_len, 1]` - chỉ dự đoán 1 giá trị
- **Target cần thiết**: `[batch, pred_len=7, 3_markets]` - dự đoán 3 markets đồng thời
- **Architecture**: Model có `EmbeddingHead` hỗ trợ categorical features

**✅ Kết luận**:
- **Cần modify output layer** từ `pred_len` → `pred_len * 3_markets`
- **Market_encoded embedding** được handle trong `EmbeddingHead` 
- **Training strategy**: Train 1 model cho cả 3 markets (Option A)

**📋 Recommended Approach**:
```python
# Modify head output:
# Current: self.linear = nn.Linear(d_model, target_window)  # target_window = pred_len
# New:     self.linear = nn.Linear(d_model, target_window * num_markets)  # pred_len * 3
# Reshape output: [batch, pred_len * 3] → [batch, pred_len, 3]
```

---

### **3. 🔧 Phân Tích Configuration Parameters**

**❓ Câu hỏi**: `c_out=3` có nghĩa là gì? Parameters nào cần thay đổi?

**🔍 Phân tích từ run.py:**
- **`enc_in`**: Encoder input size (số features đầu vào) = 21
- **`c_out`**: Output size (số outputs) = 3 markets  
- **`target`**: Target column = 'order_count'
- **`features='MS'`**: Multivariate input, target output

**✅ Kết luận**:
```bash
# Cập nhật parameters trong script:
--data_path supply_chain_processed.csv  # ✅ Dataset mới
--enc_in 21                            # ✅ 21 features input
--c_out 3                              # ✅ 3 markets output
--target order_count                   # ✅ Target mới
--features MS                          # ✅ Multivariate → target
--seq_len 21                           # ✅ 3 tuần (21 ngày)
--pred_len 7                           # ✅ Dự đoán 7 ngày
```

---

### **4. 📈 Phân Tích Sequence Length & Data Split**

**❓ Câu hỏi**: Dataset 255 ngày có đủ? Train/val/test split như thế nào?

**🔍 Phân tích data size:**
- **Total data**: 255 days
- **Sequence length**: 21 days (theo đề xuất)
- **Prediction length**: 7 days
- **Usable sequences**: 255 - 21 - 7 + 1 = 228 sequences

**✅ Đánh giá**:
- **Data split**: 80/10/10 = ~182/23/23 sequences
- **Training samples**: 182 sequences × 3 markets = 546 samples
- **⚠️ Cảnh báo**: Data khá ít cho deep learning, có thể gặp overfitting

**📋 Recommendations**:
1. **Data Augmentation**: Sử dụng overlapping windows
2. **Transfer Learning**: Pre-train trên data tương tự nếu có
3. **Regularization**: Tăng dropout, weight decay
4. **Cross-validation**: 5-fold để tận dụng data tối đa

---

### **5. 🎛️ Market Embedding Integration**

**❓ Câu hỏi**: Embedding dimension cho Market nên bao nhiêu?

**🔍 Phân tích từ EmbeddingHead:**
```python
# Rule of thumb từ code hiện tại:
embed_dim = min(50, (cat_dim + 1) // 2)

# Cho Market (3 categories):
embed_dim = min(50, (3 + 1) // 2) = min(50, 2) = 2
```

**✅ Recommendations**:
- **Market embedding dim**: 2-4 dimensions (vì chỉ có 3 markets)
- **Purpose**: Học representation của từng market và relationships
- **Integration**: Concat với main features qua `feature_projection`

---

### **6. 📋 Loss Function Analysis**

**❓ Câu hỏi**: Loss function hiện tại là gì? Dùng chung hay riêng cho từng market?

**🔍 Phân tích từ exp_long_term_forecasting_embedding.py:**
```python
def _select_criterion(self):
    criterion = nn.MSELoss()  # Mean Squared Error
    return criterion

# Training loss calculation:
loss = criterion(outputs, batch_y)  # MSE across all predictions
```

**✅ Kết luận**:
- **Current loss**: MSE (Mean Squared Error)
- **Multi-market strategy**: Combined loss across all 3 markets
- **Calculation**: MSE của `[batch, 7_days, 3_markets]` predictions

**📋 Recommended Loss Strategy**:
```python
# Option 1: Combined MSE (khuyến nghị)
loss = MSE(pred_all_markets, true_all_markets)

# Option 2: Weighted per market (nếu markets có importance khác nhau)
loss = w1*MSE(pred_market1, true_market1) + w2*MSE(pred_market2, true_market2) + w3*MSE(pred_market3, true_market3)
```

---

### **7. 📈 Business Metrics & Evaluation**

**❓ Câu hỏi**: Metrics nào quan trọng cho business case?

**📊 Recommended Metrics**:
1. **MAE (Mean Absolute Error)**: Dễ hiểu cho business (orders/day)
2. **MAPE (Mean Absolute Percentage Error)**: % error cho từng market
3. **RMSE (Root Mean Square Error)**: Penalty cao cho outliers
4. **Market-specific metrics**: Performance riêng cho từng market

**🎯 Business Thresholds (đề xuất)**:
```python
# Accuracy thresholds:
MAE < 15 orders/day/market     # Acceptable
MAE < 10 orders/day/market     # Good  
MAE < 5 orders/day/market      # Excellent

# MAPE thresholds:
MAPE < 10%   # Acceptable
MAPE < 7%    # Good
MAPE < 5%    # Excellent
```

---

### **8. 🛠️ Implementation Priority (Final)**

**📋 Updated Implementation Order**:
```
1. Tạo Dataset_SupplyChain_Processed loader mới (1-2 giờ)
2. Modify QCAAPatchTF_Embedding output layer (1 giờ)  
3. Update script configuration parameters (30 phút)
4. Test training pipeline với sample data (1 giờ)
5. Validate multi-market predictions (30 phút)
6. Full training experiment (2-4 giờ)
7. Results evaluation và metrics analysis (1 giờ)
```

**⚡ Total estimated time**: 6-9 giờ để có kết quả đầu tiên
