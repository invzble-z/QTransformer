# 📊 DATA PREPROCESSING - FOLDER XỬ LÝ DỮ LIỆU

## 🎯 **MỤC ĐÍCH CHUNG**

Folder `data_preprocessing` chứa các notebook và script để xử lý dữ liệu thô từ giao dịch chuỗi cung ứng thành định dạng time series phù hợp cho mô hình QCAAPatchTF_Embedding. Đây là bước quan trọng nhất trong pipeline, chuyển đổi từ 147K giao dịch rời rạc thành chuỗi thời gian có cấu trúc cho 3 thị trường.

---

## 📁 **CÁC FILE TRONG FOLDER**

### **1. `data_preprocessing.ipynb` - PIPELINE XỬ LÝ DỮ LIỆU CHÍNH**

#### 🔧 **Mục đích**
Notebook chính thực hiện toàn bộ pipeline xử lý dữ liệu từ raw data thành format sẵn sàng cho training. Implement chiến lược "Synchronized Timeline" (Option A) để đảm bảo đồng bộ dữ liệu giữa 3 thị trường.

#### 📋 **Các bước xử lý chính**

**Bước 1-2: Import Libraries & Initialize Preprocessor**
- Import các thư viện cần thiết: pandas, numpy, sklearn, matplotlib
- Tạo class `SupplyChainPreprocessor` với các tham số cấu hình
- Thiết lập thông số: `min_orders_per_market=10`, `outlier_threshold=3.5`

**Bước 3: Load Raw Data**
- Đọc file `DataCoSupplyChain_Synchronized.csv` (147,041 giao dịch)
- Chuyển đổi cột thời gian từ string sang datetime
- Phân tích timeline gốc và validate dữ liệu

**Bước 4: Timeline Synchronization**
- **Chiến lược Option A**: Tìm ngày đầu tiên cả 3 markets đều có ≥10 orders/ngày
- Loại bỏ dữ liệu trước synchronization point
- **Kết quả**: Chỉ mất 11.7% dữ liệu, giữ lại 88.3% dữ liệu chất lượng

**Bước 5: Daily Aggregation**
- Chuyển từ transaction-level → daily time series
- **Aggregation rules**:
  - `Count`: Order Id (biến target)
  - `Mean`: Price, discount, profit ratios, shipping days, risk
  - `Sum`: Quantity, sales, order total
  - `List`: Customer segments, categories (cho diversity calculation)

**Bước 6: Time Features Engineering**
- Tạo 5 time features: `day_of_week`, `day_of_month`, `month`, `is_weekend`, `days_since_start`
- Đảm bảo mô hình học được temporal patterns

**Bước 7: Advanced Feature Engineering với Enhanced Outlier Handling**
- **Customer segment percentages**: 3 features (Consumer, Corporate, Home Office)
- **Category diversity index**: Simpson's diversity cho product categories
- **Price volatility**: Coefficient of variation
- **Outlier treatment**: IQR method với rolling window median replacement

**Bước 8-9: Export & Validation**
- Lưu file `supply_chain_processed.csv` (765 records = 255 days × 3 markets)
- Tạo `feature_mapping.json` và `preprocessing_stats.json`
- Validation với interactive charts

#### 📊 **Kết quả đạt được**
- **Timeline**: 2017-05-22 đến 2018-01-31 (255 ngày)
- **Features**: 21 total (20 numerical + 1 categorical encoding)
- **Markets**: Europe (0), LATAM (1), USCA (2)
- **Data quality**: Excellent (no missing values, no duplicates)
- **Target variable**: `order_count` (trung bình ~170 orders/ngày/market)

---

### **2. `data_visualization.ipynb` - KHÁM PHÁ VÀ TRỰC QUAN HÓA DỮ LIỆU**

#### 🔧 **Mục đích**
Notebook dành cho Exploratory Data Analysis (EDA) của dữ liệu raw. Giúp hiểu rõ đặc điểm, patterns, và chất lượng dữ liệu trước khi xử lý. Tạo các biểu đồ tương tác để phân tích trends theo thời gian và thị trường.

#### 📋 **Nội dung chính**

**Cell 1-3: Setup & Data Loading**
- Import libraries: pandas, numpy, matplotlib, seaborn, plotly
- Load file `DataCoSupplyChain_Synchronized.csv`
- Hiển thị thông tin cơ bản: 147,041 rows × 54 columns, ~282MB

**Cell 4-5: Structural Analysis**
- Phân tích các cột thời gian: `order date (DateOrders)`, `shipping date (DateOrders)`
- Identify important columns: quantity, sales, revenue, profit, price
- Thống kê mô tả cho 54 numeric columns

**Cell 6: Data Quality Check**
- Kiểm tra missing values: `Product Description` (100%), `Order Zipcode` (83%)
- Check duplicate records: 0 duplicates
- Analyze categorical columns: 84 countries, 16 regions, 600 states

**Cell 7-8: Basic Visualizations**
- Histogram distribution cho các numeric columns
- Correlation matrix heatmap
- Box plots cho distribution analysis

**Cell 9-11: Time Series Preparation**
- Convert `order date` thành datetime format
- Tạo `order_date_only` cho daily analysis
- Phân tích Markets và Order Regions

**Cell 12-13: Interactive Time Series Charts**
- **Chart 1**: Daily orders by Market (3 markets) - line chart với plotly
- **Chart 2**: Daily orders by Order Region (16 regions) - multi-line chart
- Hover information và interactive legends

**Cell 14: Summary Statistics**
- Top Markets: rankings theo total orders
- Top Order Regions: geographical analysis
- Daily statistics: mean, max, min, std deviation
- Timeline analysis: date range và total days

#### 📊 **Insights chính**
- **Địa lý**: Tập trung ở Puerto Rico và EE.UU
- **Temporal**: Data span từ 2017-2018
- **Volume**: Trung bình ~577 orders/ngày across all markets
- **Seasonal patterns**: Observable trends theo Market và Region

---

### **3. `data_optimization_preprocessing.ipynb` - PREPROCESSING TỐI ƯU CHO PHASE 1**

#### 🔧 **Mục đích**
Notebook được tạo đặc biệt cho Phase 1 optimization. Thực hiện advanced preprocessing với focus vào feature engineering chất lượng cao và optimization cho QCAAPatchTF_Embedding model. Tạo ra dataset `supply_chain_optimized.csv` đã được fine-tuned.

#### 📋 **Workflow chính**

**Cell 1-3: Advanced Setup**
- Import enhanced libraries bao gồm statistical tools
- Load và validate raw data với error handling
- Setup advanced configuration parameters

**Cell 4-6: Smart Data Loading & Validation**
- Intelligent date parsing với multiple format support
- Data quality assessment với statistical tests
- Market validation và geographical analysis

**Cell 7-9: Enhanced Timeline Synchronization**
- Improved synchronization algorithm
- Statistical analysis of synchronization impact
- Data loss minimization strategies

**Cell 10-12: Advanced Aggregation**
- Multi-level aggregation strategies
- Business logic rules cho different metrics
- Advanced outlier detection during aggregation

**Cell 13-15: Sophisticated Feature Engineering**
- **Market dynamics features**: Cross-market correlations
- **Business calendar features**: Holidays, fiscal periods
- **Advanced time features**: Cyclical encoding, trend components
- **Statistical features**: Rolling statistics, volatility measures

**Cell 16-18: Optimization for Model**
- Feature selection dựa trên correlation analysis
- StandardScaler optimization cho neural networks
- Target variable optimization với proper scaling

**Cell 19-20: Quality Assurance**
- Advanced validation metrics
- Statistical tests cho data integrity
- Performance benchmarking

**Cell 21-22: Optimized Export**
- Export `supply_chain_optimized.csv` với enhanced features
- Create comprehensive metadata files
- Generate optimization reports

#### 📊 **Kết quả tối ưu**
- **Enhanced features**: 51 features (vs 21 trong basic preprocessing)
- **Better quality**: Advanced outlier handling và missing value treatment
- **Model-ready**: Optimal format cho QCAAPatchTF_Embedding
- **Scalers included**: Target scaler và feature scalers sẵn sàng

---

## 🔄 **WORKFLOW TỔNG THỂ**

### **Thứ tự thực hiện**
1. **Khám phá dữ liệu**: Chạy `data_visualization.ipynb` để hiểu data
2. **Preprocessing cơ bản**: Chạy `data_preprocessing.ipynb` cho pipeline chính
3. **Optimization**: Chạy `data_optimization_preprocessing.ipynb` cho Phase 1

### **Input & Output**
- **Input**: `../dataset/DataCoSupplyChain_Synchronized.csv` (147K transactions)
- **Output chính**: `../dataset/supply_chain_optimized.csv` (765 records, 51 features)
- **Metadata**: Feature mappings, preprocessing stats, scalers


## 🎯 **HƯỚNG DẪN SỬ DỤNG CHO NGƯỜI KẾ NHIỆM**

### **Để modify preprocessing**
1. **Thay đổi tham số**: Chỉnh `min_orders_per_market`, `outlier_threshold` trong class init
2. **Thêm features**: Modify function `engineer_features()` 
3. **Thay đổi aggregation**: Update `agg_rules` trong `aggregate_to_daily()`
4. **Custom validation**: Thêm charts trong validation section

### **Troubleshooting**
- **File not found**: Đảm bảo `DataCoSupplyChain_Synchronized.csv` có trong `../dataset/`
- **Memory issues**: Giảm chunk size trong data loading
- **Feature engineering errors**: Check data types và handle missing values
- **Timeline sync issues**: Adjust `min_orders_per_market` parameter

---

*Folder này chứa toàn bộ logic preprocessing cho dự án. Tất cả notebooks đều có documentation chi tiết và có thể chạy độc lập để testing hoặc modification.*
