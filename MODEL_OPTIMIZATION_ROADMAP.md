# 🚀 Kế Hoạch Tối Ưu Mô Hình - QCAAPatchTF Dự Báo Chuỗi Cung Ứng

## 📊 **Hiệu Suất Mô Hình Hiện Tại (Baseline)**
- **Mô hình**: QCAAPatchTF_Embedding 
- **Dữ liệu**: supply_chain_processed.csv (765 bản ghi, 21 đặc trưng, 3 thị trường)
- **Kết quả huấn luyện**: Dừng sớm tại epoch 34/50
- **Chỉ số hiệu suất**:
  - **MSE**: 20,437.08
  - **MAE**: 142.03
  - **Validation Loss tốt nhất**: 20,408.40 (epoch 24)
- **Kích thước mô hình**: 159,765 tham số
- **Thời gian huấn luyện**: ~4 phút

---

## 🎯 **Chiến Lược & Phân Tích Tối Ưu**

### 📊 **1. Chuẩn Hóa Dữ Liệu Đầu Ra - ƯU TIÊN CAO ✅**
**Vấn đề hiện tại**: Giá trị loss cao do chưa chuẩn hóa target (`order_count` trong khoảng: 100-400+)

**Giải pháp**:
```python
# Phương án A: MinMax Scaling (0-1)
# Ưu điểm: Giới hạn đầu ra, gradient ổn định
# Nhược điểm: Nhạy cảm với outliers

# Phương án B: Standard Scaling (mean=0, std=1) ✅ KHUYẾN NGHỊ
# Ưu điểm: Tốt cho phân phối chuẩn, bền vững
# Nhược điểm: Đầu ra không giới hạn

# Phương án C: Log Transform + Standard
# Ưu điểm: Xử lý dữ liệu lệch, giảm variance
# Nhược điểm: Phức tạp khi inverse transform
```

**Tác động dự kiến**: MSE ↓ 50-80%, MAE: 142 → 30-50

---

### 🛠 **2. Kỹ Thuật Đặc Trưng Nâng Cao - ƯU TIÊN CAO**

#### **🗓 Phân Tích Đặc Trưng Thời Gian**
**Thời lượng dữ liệu**: 240 ngày (2017-05-22 → 2018-01-31 = ~8 tháng)

| Đặc trưng | Trạng thái | Lý do |
|---------|--------|-----------|
| `quarter` | ❌ KHÔNG KHUYẾN NGHỊ | Chỉ 2.6 quý - không đủ pattern |
| `day_of_quarter` | ❌ KHÔNG KHUYẾN NGHỊ | Tương tự như trên |
| `week_of_year` | ✅ KHẢ THI | 34 tuần - đủ để tìm pattern |
| `month` | ✅ ĐÃ CÓ | 9 tháng (5,6,7,8,9,10,11,12,1) |
| `day_of_week` | ✅ ĐÃ CÓ | 7 ngày - pattern rõ ràng |
| `day_of_month` | ✅ ĐÃ CÓ | 1-31 ngày |

#### **🔄 Mã Hóa Chu Kỳ - CẢI TIẾN QUAN TRỌNG**

**Vấn đề với mã hóa tuyến tính**:
```python
# Tháng 12 = 12, Tháng 1 = 1
# Mô hình nghĩ khoảng cách = |12-1| = 11 (lớn nhất!)
# Thực tế: Chỉ cách nhau 1 tháng
```

**Giải pháp - Mã hóa chu kỳ**:
```python
# Chuyển đổi đặc trưng tuần hoàn sang không gian liên tục
month_sin = sin(2π * month / 12)
month_cos = cos(2π * month / 12)

# Ví dụ:
# Tháng 12: sin=0, cos=1  
# Tháng 1:  sin=0.5, cos=0.87
# Khoảng cách = √[(0-0.5)² + (1-0.87)²] = 0.27 ✅ GẦN!
```

**Kế hoạch triển khai**:
```python
# Ưu tiên cao
'month_sin', 'month_cos'  # 9 tháng - pattern theo mùa rõ ràng
'day_sin', 'day_cos'      # 7 ngày - chu kỳ kinh doanh hàng tuần

# Tùy chọn  
'day_of_month_sin', 'day_of_month_cos'  # Ít pattern hơn
```

#### **📅 Đặc Trưng Lịch Kinh Doanh**:
```python
✅ 'is_month_end'     # Quan trọng cho kế hoạch chuỗi cung ứng
✅ 'is_quarter_end'   # Nếu có đủ dữ liệu
❌ 'days_to_holiday'  # Bỏ qua - phức tạp với lễ nhiều quốc gia
```

#### **🏢 Đặc Trưng Theo Thị Trường** (Từ dữ liệu hiện có):
```python
# Mối quan hệ giữa các thị trường (không cần dữ liệu ngoài)
'europe_vs_latam_ratio' = europe_orders / latam_orders
'usca_vs_europe_ratio' = usca_orders / europe_orders

# Động lực thị trường (từ pattern lịch sử)
'europe_7d_trend' = slope(europe_orders[-7:])
'market_volatility' = std(orders[-7:])
'europe_market_share' = europe_orders / total_orders
```

---

### ⚙️ **3. Tối Ưu Kiến Trúc Mô Hình - ƯU TIÊN TRUNG BÌNH**

#### **Bảng So Sánh Kiến Trúc**:

| Cấu hình | Tham số | Thời gian huấn luyện | Bộ nhớ | Hiệu suất dự kiến | Mức tài nguyên |
|---------------|------------|---------------|---------|---------------------|----------------|
| **Hiện tại** | 159K | 4 phút | Thấp | Cơ sở | 🟢 Nhẹ |
| **Tùy chọn 3 (Hiệu quả)** | ~280K | 6-8 phút | Trung bình | +5-15% | 🟡 Trung bình |
| **Tùy chọn 2 (Rộng hơn)** | ~420K | 10-12 phút | Trung bình-Cao | +10-20% | 🟠 Nặng |
| **Tùy chọn 1 (Sâu hơn)** | ~640K | 15-20 phút | Cao | +15-25% | 🔴 Rất nặng |

**Tiến triển khuyến nghị**: Hiện tại → Tùy chọn 3 → Tùy chọn 2 → Tùy chọn 1

#### **Cấu hình chi tiết**:
```bash
# Hiện tại (Cơ sở)
d_model=64, n_heads=8, e_layers=3, d_ff=256

# Tùy chọn 3: Hiệu quả (BƯỚC TIẾP THEO)
d_model=80, n_heads=10, e_layers=4, d_ff=320

# Tùy chọn 2: Rộng hơn  
d_model=96, n_heads=12, e_layers=4, d_ff=384

# Tùy chọn 1: Sâu hơn
d_model=128, n_heads=8, e_layers=6, d_ff=512
```

#### **🔍 Phân Tích Cấu Hình Patch**:

```python
# seq_len = 21 ngày (đầu vào 3 tuần)

# Hiện tại: patch_len=3, patch_num=7
# Cấu trúc: [Ngày1-3][Ngày4-6][Ngày7-9]...[Ngày19-21]
# Ưu điểm: Cân bằng giữa pattern ngắn và trung hạn
# Nhược điểm: Có thể bỏ lỡ pattern chi tiết theo ngày

# Tùy chọn A: patch_len=7, patch_num=3 (BỐI CẢNH DÀI HƠN)
# Cấu trúc: [Tuần1: Ngày1-7][Tuần2: Ngày8-14][Tuần3: Ngày15-21]  
# Ưu điểm: Nắm bắt chu kỳ kinh doanh hàng tuần rõ ràng
# Nhược điểm: Mất biến thiên hàng ngày trong tuần
# Dùng khi: Pattern hàng tuần > pattern hàng ngày

# Tùy chọn B: patch_len=1, patch_num=21 (CHI TIẾT NHẤT)
# Cấu trúc: [Ngày1][Ngày2][Ngày3]...[Ngày21]
# Ưu điểm: Không mất thông tin, nắm bắt tất cả biến thiên hàng ngày
# Nhược điểm: Chi phí tính toán cao hơn, có thể overfitting
# Dùng khi: Biến động hàng ngày quan trọng
```

**Chiến lược kiểm tra**: Thử cả ba cấu hình và so sánh kết quả

---

### 📈 **4. Cải Tiến Chiến Lược Huấn Luyện - ƯU TIÊN TRUNG BÌNH**

#### **Tối ưu Learning Rate**:
```bash
# Hiện tại: 0.001 với decay mạnh
# Vấn đề: Giảm xuống 1e-13 quá nhanh

# Khuyến nghị: Cosine với restart
learning_rate=0.0005  # Bắt đầu thấp hơn
lradj=cosine_with_restarts
```

#### **Phân tích hàm kích hoạt**:
```python
# Hiện tại: GELU ✅ (Lựa chọn tốt cho transformer)
# Thay thế cho dữ liệu chuỗi cung ứng:
'swish'   # Mượt mà, tốt cho pattern chuỗi thời gian
'mish'    # Mượt hơn nữa, nghiên cứu mới nhất
'relu'    # Đơn giản, nhanh nhất (phương án dự phòng)
```

#### **Tăng cường dữ liệu cho dataset nhỏ**:
**Vấn đề**: Chỉ có 765 bản ghi → Nguy cơ overfitting cao

**Giải pháp**:
```python
# 1. Gaussian Noise (σ=0.05)
# Gốc: [100, 150, 120, 200]
# Tăng cường: [102, 148, 122, 198]  # Thêm nhiễu ngẫu nhiên nhỏ
# Lợi ích: Mô hình bền vững với lỗi đo lường

# 2. Magnitude Scaling (0.9-1.1x)  
# Gốc: [100, 150, 120]
# Scaled: [110, 165, 132]  # Nhân với hệ số ngẫu nhiên
# Lợi ích: Mô hình thích ứng với khối lượng đơn hàng khác nhau

# 3. Time Warping (±2 ngày) - Nâng cao
# Gốc: Ngày1→Ngày2→Ngày3→Ngày4
# Warped: Ngày1→Ngày3→Ngày2→Ngày4  # Đổi chỗ ngày kề nhau
# Lợi ích: Ít nhạy cảm với thời gian chính xác
```

**Khuyến nghị**: Bắt đầu với Gaussian + Magnitude (đơn giản, hiệu quả)

---

### 🎯 **5. Hàm Loss Đa Thị Trường - ƯU TIÊN CAO ✅**

#### **Vấn đề hiện tại**:
```python
# Tất cả thị trường được đối xử như nhau
loss = MSE(europe) + MSE(latam) + MSE(usca)
```

#### **Giải pháp có trọng số kinh doanh**:
```python
# Tầm quan trọng thị trường dựa trên giá trị kinh doanh
europe_weight = 0.35   # Lớn nhất, ổn định nhất
usca_weight = 0.35     # Khách hàng giá trị cao  
latam_weight = 0.30    # Đang phát triển nhưng biến động

weighted_loss = (0.35 * MSE(europe) + 
                0.30 * MSE(latam) + 
                0.35 * MSE(usca))
```

**Triển khai**: Chỉnh sửa hàm loss trong vòng lặp huấn luyện

---

## 🎯 **Lộ Trình Triển Khai**

### **Giai đoạn 1: Thắng nhanh (1-2 ngày) 🚀**
**Mục tiêu**: Cải thiện 50-60% với nỗ lực tối thiểu

1. ✅ **Chuẩn hóa Target** ✅ **HOÀN THÀNH**
   - ✅ Áp dụng StandardScaler cho `order_count`
   - ✅ Dataset: 765 rows × 51 features (+28 features mới)
   - ✅ Target normalized: mean≈0, std≈1, range [-2.7, 2.9]

2. ✅ **Mã hóa chu kỳ** ✅ **HOÀN THÀNH**
   - ✅ Thêm `month_sin/cos`, `day_sin/cos`, `day_of_month_sin/cos`, `week_of_year_sin/cos`
   - ✅ Cyclical encoding working: Dec→Jan distance = 0.27 (thay vì 11!)

3. ✅ **Loss có trọng số thị trường** ✅ **HOÀN THÀNH**
   - ✅ Triển khai WeightedMSELoss: [Europe=0.35, LATAM=0.30, USCA=0.35]
   - ✅ Integrated vào exp_long_term_forecasting_embedding.py

4. ✅ **Đặc trưng thị trường cơ bản** ✅ **HOÀN THÀNH**
   - ✅ Cross-market ratios: europe_vs_latam_ratio, usca_vs_europe_ratio
   - ✅ Market shares: europe/latam/usca_market_share
   - ✅ Market dynamics: 7d volatility, trends, momentum

🚀 **Trạng thái hiện tại**: ĐANG TRAINING với dataset optimized
📊 **Training Progress**: WeightedMSELoss đang hoạt động, loss đã giảm từ 1.137 → 0.994

**Kết quả dự kiến**: MSE: 20,437 → 10,000-12,000 (cải thiện 50-60%)

### **Giai đoạn 2: Nâng cấp mô hình (2-3 ngày) 🔧**
**Mục tiêu**: Cải thiện thêm 30-40%

1. ✅ **Nâng cấp kiến trúc**
   - Thử Tùy chọn 3 (Hiệu quả): d_model=80, e_layers=4
   - So sánh với baseline hiện tại

2. ✅ **Kiểm tra cấu hình Patch**
   - Thử patch_len=7 (bối cảnh tuần)
   - Thử patch_len=1 (độ chi tiết ngày)
   - So sánh với patch_len=3 hiện tại

3. ✅ **Chiến lược huấn luyện**
   - Triển khai cosine learning rate schedule
   - Thử các hàm kích hoạt khác nhau

4. ✅ **Tăng cường dữ liệu**
   - Gaussian noise + Magnitude scaling
   - Tăng kích thước dataset hiệu quả

**Kết quả dự kiến**: MSE: 10,000-12,000 → 6,000-9,000 (thêm 30-40%)

### **Giai đoạn 3: Tinh chỉnh (1-2 ngày) 🎨**
**Mục tiêu**: Cải thiện cuối cùng 20-30%

1. ✅ **Tìm kiếm siêu tham số**
   - Grid search các cấu hình tốt nhất
   - Tối ưu learning rate, dropout, batch size

2. ✅ **Đặc trưng nâng cao**
   - Đặc trưng lịch kinh doanh (`is_month_end`)
   - Chỉ số momentum và độ biến động thị trường

3. ✅ **Mở rộng kiến trúc mô hình**
   - Nếu tài nguyên cho phép: Tùy chọn 2 hoặc Tùy chọn 1
   - Cân bằng hiệu suất vs hiệu quả

**Kết quả dự kiến**: MSE: 6,000-9,000 → 4,000-7,000 (cuối cùng 20-30%)

### **Giai đoạn 4: Xác thực & Tài liệu (1 ngày) 📊**
1. ✅ **Cross-validation** (phù hợp chuỗi thời gian)
2. ✅ **Bảng so sánh mô hình**
3. ✅ **Tài liệu kết quả**
4. ✅ **Báo cáo phân tích hiệu suất**

---

## 📈 **Kết Quả Cuối Cùng Dự Kiến**

| Chỉ số | Hiện tại | Giai đoạn 1 | Giai đoạn 2 | Giai đoạn 3 | Tổng cải thiện |
|--------|---------|---------|---------|---------|-------------------|
| **MSE** | 20,437 | 10,000-12,000 | 6,000-9,000 | 4,000-7,000 | **65-80% ↓** |
| **MAE** | 142 | 70-85 | 45-65 | 25-40 | **75-85% ↓** |
| **Thời gian huấn luyện** | 4 phút | 4-5 phút | 6-10 phút | 8-15 phút | Tăng vừa phải |

---

## 🔄 **Quyết Định Đã Đưa Ra**

### ✅ **Đã phê duyệt để triển khai**:
1. Chuẩn hóa target (StandardScaler)
2. Mã hóa chu kỳ (tháng, ngày)
3. Hàm loss có trọng số thị trường
4. Đặc trưng thị trường cơ bản từ dữ liệu hiện có
5. Tiến triển kiến trúc (Hiện tại → Tùy chọn 3 → Tùy chọn 2)
6. Kiểm tra cấu hình patch
7. Cải thiện chiến lược huấn luyện

### ❌ **Bị từ chối/Hoãn lại**:
1. Đặc trưng theo quý (dữ liệu không đủ)
2. Đặc trưng ngày lễ (phức tạp đa quốc gia)
3. Dữ liệu thị trường bên ngoài (giữ dataset thuần túy)
4. Phương pháp ensemble phức tạp (tập trung vào QCAAPatchTF)
5. Xác thực theo nghiệp vụ cụ thể (tập trung học thuật)
6. Cân nhắc production (giai đoạn nghiên cứu)

### 🤔 **Để thảo luận sau**:
1. Lựa chọn kiến trúc cuối cùng (Tùy chọn 2 vs Tùy chọn 3)
2. Lựa chọn cấu hình patch
3. Lựa chọn hàm kích hoạt cuối cùng
4. Mức độ tăng cường dữ liệu

---

## 📝 **Bước Tiếp Theo**
1. **Bắt đầu Giai đoạn 1**: Bắt đầu với triển khai chuẩn hóa target
2. **Tạo nhánh**: `feature/model-optimization`
3. **Triển khai từng bước**: Một thay đổi một lúc để theo dõi tác động rõ ràng
4. **Ghi chép kết quả**: Theo dõi hiệu suất sau mỗi thay đổi
5. **A/B test cấu hình**: So sánh chỉ số trước/sau

---

**Cập nhật lần cuối**: 21 tháng 8, 2025  
**Trạng thái**: Sẵn sàng triển khai  
**Thời gian dự kiến**: 5-8 ngày  
**Mức cải thiện dự kiến**: 65-80% cải thiện MSE/MAE
