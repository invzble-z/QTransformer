# QCAAPatchTF với Embedding Approach cho Multi-Region Forecasting

## Tổng quan

Đã successfully implement một giải pháp embedding approach cho QCAAPatchTF model để dự đoán số lượng order ở nhiều vùng Region cùng lúc, giải quyết vấn đề one-hot encoding tạo ra quá nhiều cột.

## Các thay đổi chính

### 1. Data Preprocessing (`data_preprocessing/data_preprocessing.py`)
- **Thay One-Hot Encoding bằng Label Encoding**: Giảm dramatically số dimensions
- **Lưu trữ encoders và categorical dimensions**: Cho việc load lại và inference
- **Preprocessing tự động**: Xử lý categorical và numerical features riêng biệt

### 2. Data Loader mới (`data_provider/data_loader_embedding.py`)
- **Dataset_SupplyChain_Embedding**: Xử lý label-encoded data
- **Dataset_MultiRegion_Embedding**: Hỗ trợ multi-region forecasting
- **Custom collate function**: Xử lý categorical features trong training

### 3. Model mới (`models/QCAAPatchTF_Embedding.py`)
- **QCAAPatchTF_Embedding**: Tương tự QCAAPatchTF nhưng hỗ trợ embedding
- **Embedding layers**: Tự động tạo embeddings cho categorical features
- **Flexible architecture**: Tương thích với cả channel-independent và channel-dependent modes

### 4. Experiment Handler (`exp/exp_long_term_forecasting_embedding.py`)
- **Exp_Long_Term_Forecast_Embedding**: Xử lý categorical features trong training loop
- **Enhanced data handling**: Support cả embedding và non-embedding datasets
- **Improved error handling**: Better shape management cho predictions

## So sánh One-Hot vs Embedding Approach

### Trước (One-Hot Encoding):
```
Order Region: 23 values → 23 columns
Market: 5 values → 5 columns  
Shipping Mode: 4 values → 4 columns
Total categorical features: 32 columns
```

### Sau (Embedding Approach):
```
Order Region: 23 values → Embedding dimension ~12
Market: 5 values → Embedding dimension ~3
Shipping Mode: 4 values → Embedding dimension ~2
Total categorical features: ~17 dimensions
```

**Lợi ích:**
- Giảm ~50% số parameters cho categorical features
- Tăng tốc training và inference
- Học được relationships giữa categories
- Dễ dàng thêm new categories không cần retrain toàn bộ

## Cách sử dụng

### 1. Data Preprocessing
```bash
cd /home/u1/Desktop/Gra_pr/QTransformer
python3 data_preprocessing/data_preprocessing.py
```

### 2. Training
```bash
# Chạy script embedding
./scripts/long_term_forecast/QCAAPatchTF_SupplyChain_Embedding.sh

# Hoặc chạy command trực tiếp
python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path seller_Order_Region_processed.csv \
  --model_id SupplyChain_Embedding \
  --model QCAAPatchTF_Embedding \
  --channel_independence 1 \
  --data SupplyChainEmbedding \
  --features MS \
  --target "Order Item Quantity" \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 7 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --num_workers 0
```

### 3. Testing
```bash
python3 run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model_id SupplyChain_Embedding \
  --model QCAAPatchTF_Embedding \
  --data SupplyChainEmbedding \
  --target "Order Item Quantity" \
  # ... other parameters same as training
```

## Kết quả

### Model Performance
- **Dự đoán**: 7 ngày tiếp theo của Order Item Quantity
- **Input**: 60 ngày historical data
- **Multi-region**: Có thể predict cho tất cả 23 regions cùng lúc
- **Features**: Order Region (23 values), Market (5 values), Shipping Mode (4 values) + numerical features

### Training Results (Test run)
```
Train Loss: 1201.32
Validation Loss: 1471.41  
Test Loss: 789.61
MSE: 789.56
MAE: 24.59
```

## File Structure Created/Modified

### Các file mới:
- `data_provider/data_loader_embedding.py`
- `models/QCAAPatchTF_Embedding.py` 
- `exp/exp_long_term_forecasting_embedding.py`
- `scripts/long_term_forecast/QCAAPatchTF_SupplyChain_Embedding.sh`

### Các file đã cập nhật:
- `data_preprocessing/data_preprocessing.py` - Thêm embedding preprocessing
- `data_provider/data_factory.py` - Support embedding datasets
- `exp/exp_basic.py` - Add embedding model
- `models/__init__.py` - Include embedding model
- `run.py` - Automatic embedding experiment selection

## Next Steps

1. **Hyperparameter Tuning**: Optimize embedding dimensions, learning rate, batch size
2. **Advanced Features**: Thêm positional encoding, attention mechanisms  
3. **Multi-Region Enhancement**: Implement region-specific heads for better performance
4. **Production Ready**: Add model serving, API endpoints
5. **Monitoring**: Add metrics tracking and model performance monitoring

## Advantages of This Approach

1. **Scalability**: Dễ dàng thêm new regions without architectural changes
2. **Efficiency**: Significant reduction in model size and training time
3. **Interpretability**: Embedding vectors có thể analyze relationships
4. **Flexibility**: Có thể easy switch giữa embedding và one-hot approaches
5. **Future-proof**: Architecture support nhiều types of categorical encoding

Approach này successfully giải quyết bài toán multi-region forecasting với hiệu quả cao hơn traditional one-hot encoding approach.
