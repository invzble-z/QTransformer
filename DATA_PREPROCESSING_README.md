# 📊 Supply Chain Time Series Forecasting - Data Preprocessing Strategy

## 🎯 PROJECT OVERVIEW

**Objective:** Dự đoán tổng số orders hàng ngày cho 3 markets (USCA, LATAM, Europe) trong 7 ngày tới

**Input:** Transaction-level supply chain data  
**Output:** Daily order counts per market for next 7 days  
**Model:** QCAAPatchTF_Embedding (Modified)

---

## 📋 DATASET ANALYSIS

### Raw Data Characteristics:
- **Size:** 147,041 transactions × 54 features
- **Timeline:** 2017-01-18 to 2018-01-31 (378 days, 309 active days)
- **Markets:** 3 (USCA: 52.5K, LATAM: 51.6K, Europe: 42.9K orders)
- **No duplicates, minimal missing values**

### Key Insights:
- **Daily average:** 476 orders/day (±83.3 std)
- **Peak day:** 584 orders (2017-11-02)  
- **Data quality:** Good temporal coverage, balanced market distribution

---

## 🏗️ DATA TRANSFORMATION STRATEGY

### 1. **AGGREGATION APPROACH**

#### **From Transaction → Time Series:**
```python
# Transform transaction-level to daily aggregated time series
daily_data = df.groupby(['order_date_only', 'Market']).agg({    
    # Target variable
    'Order Id': 'count',                    # Daily order count (our target)
    
    # Raw numerical features (to be averaged)
    'Days for shipping (real)': 'mean',
    'Late_delivery_risk': 'mean',
    'Order Item Product Price': 'mean',
    'Order Item Discount Rate': 'mean',
    'Order Item Profit Ratio': 'mean',
    'Order Profit Per Order': 'mean',
    
    # For engineered features
    'Order Item Quantity': 'sum',          # Total quantity per day
    'Sales': 'sum',                        # Total sales per day  
    'Order Item Total': 'mean',            # Avg order value
    'Customer Segment': 'first',           # For diversity calculation
    'Category Name': 'first'               # For diversity calculation
}).reset_index()

# Then create time features from 'order_date_only'
daily_data['day_of_week'] = daily_data['order_date_only'].dt.dayofweek
daily_data['day_of_month'] = daily_data['order_date_only'].dt.day  
daily_data['month'] = daily_data['order_date_only'].dt.month
daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6])
daily_data['days_since_start'] = (daily_data['order_date_only'] - start_date).dt.days
```

#### **Timeline Cleaning:**
- **Valid start date:** First day all 3 markets have ≥10 orders
- **Expected removal:** ~30-60 initial days (unbalanced period)
- **Final timeline:** ~250-280 training days

#### **⚠️ CRITICAL: Multi-Market Timeline Synchronization**

**Problem:** Markets bắt đầu hoạt động vào thời điểm khác nhau
- Market A có thể bắt đầu từ ngày 1
- Market B có thể bắt đầu từ ngày 15  
- Market C có thể bắt đầu từ ngày 30

**Solution Strategy:**

```python
def find_synchronized_start_date(df, min_orders_per_market=10):
    """
    Tìm ngày đầu tiên mà TẤT CẢ 3 markets đều có >= min_orders_per_market
    """
    # Đếm orders hàng ngày cho từng market
    daily_counts = df.groupby(['order_date_only', 'Market']).size().reset_index(name='daily_orders')
    
    # Pivot để có ma trận [date x market]
    market_matrix = daily_counts.pivot(index='order_date_only', columns='Market', values='daily_orders').fillna(0)
    
    # Tìm ngày đầu tiên tất cả markets >= threshold
    valid_days = (market_matrix >= min_orders_per_market).all(axis=1)
    synchronized_start_date = valid_days[valid_days == True].index[0]
    
    return synchronized_start_date

# Áp dụng timeline cleaning
start_date = find_synchronized_start_date(df, min_orders_per_market=10)
df_synchronized = df[df['order_date_only'] >= start_date]

print(f"Original timeline: {df['order_date_only'].min()} to {df['order_date_only'].max()}")
print(f"Synchronized timeline: {start_date} to {df['order_date_only'].max()}")
print(f"Removed {(start_date - df['order_date_only'].min()).days} days for synchronization")
```

**Impact Assessment:**
```python
# Kiểm tra data loss per market
removed_data = df[df['order_date_only'] < start_date]
for market in ['USCA', 'LATAM', 'Europe']:
    market_removed = removed_data[removed_data['Market'] == market].shape[0]
    market_total = df[df['Market'] == market].shape[0]
    loss_pct = (market_removed / market_total) * 100
    print(f"{market}: Lost {market_removed:,} orders ({loss_pct:.1f}%)")
```

### 2. **FEATURE ENGINEERING**

#### **Input Features (21 total):**

**⚠️ IMPORTANT NOTE:** Time features sẽ được **tạo mới** từ cột `order date (DateOrders)`, không có sẵn trong raw dataset.

**Numerical Features từ Raw Data (6):**
```python
raw_numerical_features = [
    'Days for shipping (real)',     # From dataset - avg per day/market
    'Late_delivery_risk',           # From dataset - rate per day/market  
    'Order Item Product Price',     # From dataset - avg per day/market
    'Order Item Discount Rate',     # From dataset - avg per day/market
    'Order Item Profit Ratio',      # From dataset - avg per day/market
    'Order Profit Per Order'        # From dataset - avg per day/market
]
```

**Engineered Numerical Features (9):**
```python
engineered_features = [
    'order_count',                  # Target-related: daily order count per market
    'order_quantity_total',         # Sum of Order Item Quantity per day/market
    'customer_segment_consumer_pct', # % Consumer vs Corporate/Home Office
    'customer_segment_corporate_pct', # % Corporate 
    'customer_segment_home_office_pct', # % Home Office
    'category_diversity_index',     # Simpson's diversity of Category Name
    'sales_total',                  # Sum of Sales per day/market
    'order_item_total_avg',         # Avg Order Item Total per day/market
    'price_volatility'              # Std dev of prices within day/market
]
```

**Time Features (5) - TẠO MỚI từ 'order date (DateOrders)':**
```python
time_features = [
    'day_of_week',      # 0-6 (Monday=0) - CREATED from datetime
    'day_of_month',     # 1-31 - CREATED from datetime
    'month',            # 1-12 - CREATED from datetime  
    'is_weekend',       # Boolean - CREATED from day_of_week
    'days_since_start'  # Trend component - CREATED from start date
]
# Note: Removed 'quarter' as timeline is only ~1 year (not enough quarters)
```

**Categorical Features (Embedded):**
```python
categorical_features = {
    'Market': 3,              # USCA, LATAM, Europe (existing in dataset)
    # Note: Removed Shipping_Mode as per user request
}
# Embedding dimension: Market (16) = 16 total embedded features
```

**TOTAL: 6 + 9 + 5 + 1 (Market embedded as 16) = 21 features**

### 3. **TARGET VARIABLE DESIGN**

#### **Multi-Market Output Structure:**
```python
# Shape: [batch_size, pred_len=7, num_markets=3]
target_shape = (batch_size, 7, 3)

# Market order: [USCA, LATAM, Europe]
# Day order: [Day+1, Day+2, ..., Day+7]
```

#### **Normalization Strategy:**
- **Method:** StandardScaler per market
- **Reason:** Different market scales (USCA ~170/day vs Europe ~140/day)

---

## 🧹 DATA CLEANING PIPELINE

### 1. **Outlier Detection & Treatment**

#### **Daily Order Count Outliers:**
```python
# Method: Modified Z-Score per market
threshold = 3.5
outliers = abs(zscore(daily_orders)) > threshold

# Treatment: Winsorization (cap at 95th percentile)
daily_orders_clean = winsorize(daily_orders, limits=[0.05, 0.05])
```

#### **Feature Outliers:**
```python
# Method: IQR-based detection
Q1, Q3 = percentile(feature, [25, 75])
IQR = Q3 - Q1
outliers = (feature < Q1 - 1.5*IQR) | (feature > Q3 + 1.5*IQR)

# Treatment: 3-day rolling median smoothing
feature_smoothed = rolling_median(feature, window=3)
```

### 2. **Missing Value Handling**

#### **Minimal Missing Data:**
- `Product Description`: 100% missing → Drop column
- `Order Zipcode`: 83% missing → Drop column  
- Other features: <1% missing → Forward fill

#### **Missing Days:**
```python
# Create complete date range for synchronized period
synchronized_start = find_synchronized_start_date(df)
full_date_range = pd.date_range(synchronized_start, df['order_date_only'].max(), freq='D')

# Fill missing days with market-specific interpolation
for market in ['USCA', 'LATAM', 'Europe']:
    market_data = daily_agg[daily_agg['Market'] == market].set_index('order_date_only')
    market_data = market_data.reindex(full_date_range)
    
    # Handle missing days (weekends, low activity days)
    market_data = market_data.interpolate(method='linear')
    
    # For categorical features, forward fill
    categorical_cols = ['customer_segment_dominant', 'category_dominant']
    market_data[categorical_cols] = market_data[categorical_cols].fillna(method='ffill')
```

#### **⚠️ Alternative Approach: Market-specific Modeling**
```python
# Option 2: Train separate models per market (if synchronization removes too much data)
def create_market_specific_datasets(df):
    market_datasets = {}
    
    for market in ['USCA', 'LATAM', 'Europe']:
        market_df = df[df['Market'] == market].copy()
        
        # Each market uses its own timeline
        market_start = market_df['order_date_only'].min()
        market_end = market_df['order_date_only'].max()
        
        # Filter for minimum activity threshold
        daily_counts = market_df.groupby('order_date_only').size()
        valid_days = daily_counts[daily_counts >= 5].index  # Lower threshold per market
        
        market_datasets[market] = market_df[market_df['order_date_only'].isin(valid_days)]
    
    return market_datasets

# Use this approach if synchronized timeline is too restrictive
```

---

## 🔄 MODEL INPUT/OUTPUT DESIGN

### **QCAAPatchTF_Embedding Modifications**

#### **Original vs Modified:**
```python
# Original QCAAPatchTF:
input_shape = [batch, seq_len, n_features]
output_shape = [batch, pred_len, n_features]  # All features predicted

# Modified QCAAPatchTF_Embedding:
input_shape = [batch, seq_len, n_features]
output_shape = [batch, pred_len, n_markets]  # Only target predicted
```

#### **Architecture Changes:**
1. **Multi-head output:** 3 separate heads for 3 markets
2. **Embedding integration:** Categorical features → learned embeddings
3. **Feature projection:** Combine numerical + embedded → d_model dimension

### **Embedding Strategy:**
```python
# Market embedding only (removed Shipping_Mode)
market_embed_dim = 16
market_embedding = nn.Embedding(3, market_embed_dim)

# Total embedded dimension: 16
# Combined with 20 numerical/time features = 36 total features
```

---

## ⚖️ TRAINING STRATEGY

### **Data Splitting (Time Series)**
```python
# Total timeline: ~280 days
train_split = 0.7    # ~196 days (70%)
val_split = 0.15     # ~42 days (15%)  
test_split = 0.15    # ~42 days (15%)

# Ensure chronological order
train_end = start_date + timedelta(days=196)
val_end = train_end + timedelta(days=42)
test_end = val_end + timedelta(days=42)
```

### **Model Configuration**
```python
model_config = {
    'seq_len': 60,           # 60 days input history
    'pred_len': 7,           # 7 days prediction
    'enc_in': 36,            # Total input features (20 numerical/time + 16 embedded)
    'c_out': 3,              # 3 markets output
    'd_model': 512,          # Model dimension
    'n_heads': 8,            # Attention heads
    'e_layers': 3,           # Encoder layers
    'dropout': 0.1,
    'embed_type': 'learned'  # For categorical features
}
```

### **Loss Function (Hybrid Weighted)**
```python
def hybrid_weighted_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    # Time weighting (recent days more important)
    time_weights = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    
    # Market weighting (based on volume)
    market_weights = torch.tensor([0.4, 0.35, 0.25])  # USCA, LATAM, Europe
    
    # Combined loss
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
    weighted_loss = (alpha * time_weights + beta * market_weights) * mse_loss
    
    return weighted_loss.mean()
```

---

## 📈 EVALUATION METRICS

### **Primary Metrics:**
1. **MAPE** (Mean Absolute Percentage Error) - Business interpretable
2. **MAE** (Mean Absolute Error) - Robust to outliers  
3. **RMSE** (Root Mean Square Error) - Penalty for large errors

### **Market-specific Evaluation:**
```python
metrics_per_market = {
    'USCA': calculate_metrics(y_true_usca, y_pred_usca),
    'LATAM': calculate_metrics(y_true_latam, y_pred_latam),
    'Europe': calculate_metrics(y_true_europe, y_pred_europe)
}

# Overall weighted average
overall_metrics = weighted_average(metrics_per_market, market_weights)
```

---

## 🔧 IMPLEMENTATION PIPELINE

### **📁 File Structure & Responsibilities:**

```
data_preprocessing/
├── data_visualization.ipynb      # 📊 EDA & Interactive visualizations  
├── data_preprocessing.ipynb      # 🛠️  Preprocessing pipeline (STRUCTURE ONLY)
└── README.md                     # 📋 This strategy document
```

**File Purposes:**
- **`data_visualization.ipynb`**: Exploratory Data Analysis, trend analysis, plotly charts
- **`data_preprocessing.ipynb`**: **Empty structure** prepared for implementing Option A preprocessing pipeline
- **`DATA_PREPROCESSING_README.md`**: Complete technical documentation and strategy guide

### **Step 1: Data Preparation**
```python
def preprocess_supply_chain_data(df):
    # 1. Timeline synchronization - CRITICAL STEP
    synchronized_start = find_synchronized_start_date(df, min_orders_per_market=10)
    df_clean = df[df['order_date_only'] >= synchronized_start].copy()
    
    print(f"Timeline synchronized from {synchronized_start}")
    print(f"Removed {(synchronized_start - df['order_date_only'].min()).days} days")
    
    # 2. Daily aggregation by market (using actual column names)
    daily_agg = df_clean.groupby(['order_date_only', 'Market']).agg({
        'Order Id': 'count',  # Target: daily order count
        'Days for shipping (real)': 'mean',
        'Late_delivery_risk': 'mean', 
        'Order Item Product Price': 'mean',
        'Order Item Discount Rate': 'mean',
        'Order Item Profit Ratio': 'mean',
        'Order Profit Per Order': 'mean',
        'Order Item Quantity': 'sum',
        'Sales': 'sum',
        'Order Item Total': 'mean',
        'Customer Segment': lambda x: x.mode()[0],  # Most frequent segment
        'Category Name': 'nunique'  # Category diversity per day
    }).reset_index()
    
    # 3. Create complete timeline for all markets
    full_timeline = create_complete_timeline(daily_agg, synchronized_start)
    
    # 4. Create time features from order_date_only
    daily_with_time = create_time_features(full_timeline)
    
    # 5. Handle outliers and missing values
    daily_clean = clean_outliers_and_missing(daily_with_time)
    
    # 6. Engineer additional features (diversity indices, etc.)
    daily_featured = engineer_features(daily_clean)
    
    # 7. Create time series format for model
    timeseries_data = create_timeseries_format(daily_featured)
    
    return timeseries_data
```

### **Step 2: Model Training**
```python
# Use QCAAPatchTF_Embedding with custom config
model = QCAAPatchTF_Embedding(config)

# Training loop with time series validation
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    if val_loss < best_val_loss:
        save_model(model, f'best_model_epoch_{epoch}.pt')
```

### **Step 3: Prediction & Evaluation**
```python
# Generate predictions for test set
test_predictions = model.predict(test_data)

# Evaluate per market and overall
results = evaluate_predictions(test_predictions, test_targets)
```

---

## 📊 EXPECTED OUTCOMES

### **Model Performance Targets:**
- **Overall MAPE:** < 15%
- **Day 1-3 MAPE:** < 10% (short-term accuracy)
- **Day 4-7 MAPE:** < 20% (medium-term planning)

### **Business Value:**
- **Supply planning:** 7-day rolling forecasts
- **Inventory optimization:** Market-specific demand patterns
- **Resource allocation:** Cross-market insights

---

## 🔄 PROJECT TRACKING

### **Version History:**
- **v1.0:** Initial data exploration and strategy design
- **v1.1:** TBD - Implementation phase
- **v1.2:** TBD - Model training and optimization

### **Next Steps:**
1. ✅ Data exploration and analysis
2. ⏳ Implement preprocessing pipeline
3. ⏳ Modify QCAAPatchTF_Embedding model
4. ⏳ Training and hyperparameter tuning
5. ⏳ Evaluation and business validation

### **Key Decisions Made:**
- **Model:** QCAAPatchTF_Embedding (modified for multi-market output)
- **Prediction:** Multi-step direct (7 days at once)
- **Features:** 36 features (20 numerical/time + 16 embedded Market)
- **Loss:** Hybrid weighted (time + market importance)
- **Validation:** Time series split
- **Removed:** Shipping_Mode feature (as per user request)

---

## 📝 NOTES

### **Important Considerations:**
1. **Seasonality:** Monitor for weekly/monthly patterns
2. **Market correlation:** USCA-LATAM might be correlated
3. **External factors:** Model doesn't include holidays/events
4. **Scalability:** Pipeline designed for easy feature addition

### **Risk Mitigation:**
- **Overfitting:** Dropout + early stopping
- **Data leakage:** Strict temporal split
- **Market bias:** Weighted loss function
- **Outliers:** Robust preprocessing pipeline

---

## 📅 TIMELINE SYNCHRONIZATION ANALYSIS

### **Expected Timeline Issues:**

Based on supply chain dataset characteristics, different markets likely have:

#### **Typical Market Launch Patterns:**
```python
# Expected market timeline (hypothetical)
market_timelines = {
    'USCA': '2017-01-18',      # First to launch (established market)
    'LATAM': '2017-02-15',     # Second wave expansion  
    'Europe': '2017-03-10'     # Last to launch (new market entry)
}

# Impact on data availability
total_timeline = '2017-01-18 to 2018-01-31'  # 378 days
synchronized_timeline = '2017-03-10 to 2018-01-31'  # ~327 days
data_loss = '~51 days of early USCA/LATAM data'
```

#### **Decision Matrix:**

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Synchronized Timeline** | Clean multi-market modeling, No missing markets | Loses early single-market data | When data loss < 20% |
| **Market-specific Models** | Uses all available data, Market-tailored features | Complex deployment, No cross-market learning | When data loss > 30% |
| **Hybrid Approach** | Best of both worlds | More complex implementation | When markets have different patterns |

#### **Implementation Decision Logic:**
```python
def choose_timeline_strategy(df):
    # Calculate data loss for synchronization
    sync_start = find_synchronized_start_date(df)
    total_records = len(df)
    sync_records = len(df[df['order_date_only'] >= sync_start])
    data_loss_pct = ((total_records - sync_records) / total_records) * 100
    
    if data_loss_pct < 15:
        return "synchronized_timeline"
    elif data_loss_pct < 30:
        return "hybrid_approach" 
    else:
        return "market_specific_models"
        
    print(f"Data loss for synchronization: {data_loss_pct:.1f}%")
    print(f"Recommended approach: {strategy}")
```

### **Risk Mitigation:**
- **Monitor data quality** during synchronization
- **Validate market balance** after timeline cleaning  
- **Consider business context** (market launch strategies)
- **Implement fallback** to market-specific models if needed

### **✅ FINAL DECISION: Synchronized Timeline (Option A)**

**Chosen Strategy:** 🔄 **Option A: Synchronized Timeline**

**Rationale:**
- **Clean multi-market modeling:** Single unified model predicting all 3 markets simultaneously
- **Model output:** `[batch, 7_days, 3_markets]` as originally planned
- **Cross-market learning:** Model can learn shared patterns between markets
- **Deployment simplicity:** One model for all markets
- **Acceptable data loss:** Expected 10-20% data loss for synchronization is manageable

**Implementation Approach:**
```python
# Use find_synchronized_start_date() function
synchronized_start = find_synchronized_start_date(df, min_orders_per_market=10)
df_final = df[df['order_date_only'] >= synchronized_start]

# Expected outcome:
# - Original: ~378 days total timeline
# - Synchronized: ~250-280 days usable timeline  
# - Data loss: ~51-100 days (~13-26% acceptable range)
```

**Benefits of This Choice:**
- ✅ Unified forecasting system
- ✅ Cross-market pattern detection
- ✅ Simpler model architecture
- ✅ Easier business interpretation
- ✅ Future scalability (new markets)

**Trade-offs Accepted:**
- ⚠️ Loss of early single-market data
- ⚠️ Shorter training timeline
- ⚠️ Potential underutilization of market-specific patterns

---

## 📝 DETAILED FEATURE MAPPING

### **Raw Dataset Columns → Processed Features:**

#### **Available in Raw Dataset:**
```python
# These columns exist in DataCoSupplyChain_Synchronized.csv
raw_columns_used = [
    'order date (DateOrders)',        # → time features (6 features)
    'Market',                         # → embedded feature (16 dim)
    'Days for shipping (real)',       # → daily average
    'Late_delivery_risk',             # → daily rate
    'Order Item Product Price',       # → daily average
    'Order Item Discount Rate',       # → daily average
    'Order Item Profit Ratio',        # → daily average
    'Order Profit Per Order',         # → daily average
    'Order Id',                       # → daily count (TARGET)
    'Order Item Quantity',            # → daily sum
    'Sales',                          # → daily sum
    'Order Item Total',               # → daily average
    'Customer Segment',               # → diversity calculation
    'Category Name'                   # → diversity calculation
]
```

#### **Engineered Features (Created during preprocessing):**
```python
created_features = {
    # Time features from 'order date (DateOrders)'
    'day_of_week': 'df["order_date_only"].dt.dayofweek',
    'day_of_month': 'df["order_date_only"].dt.day',
    'month': 'df["order_date_only"].dt.month', 
    'is_weekend': 'df["day_of_week"].isin([5, 6])',
    'days_since_start': '(df["order_date_only"] - start_date).dt.days',
    
    # Diversity features
    'customer_segment_consumer_pct': 'calc_segment_percentage("Consumer")',
    'customer_segment_corporate_pct': 'calc_segment_percentage("Corporate")', 
    'customer_segment_home_office_pct': 'calc_segment_percentage("Home Office")',
    'category_diversity_index': 'calc_simpson_diversity(category_counts)',
    'price_volatility': 'daily_price_std_dev'
}
```

### **Final Feature Count Breakdown:**
- **Raw numerical features:** 6 (from existing columns)
- **Engineered numerical features:** 9 (calculated)
- **Time features:** 5 (from datetime conversion)
- **Embedded categorical:** 16 (Market embedding)
- **TOTAL:** 36 features

---

*Last updated: August 20, 2025*  
*Author: AI Assistant*  
*Project: QTransformer Supply Chain Forecasting*
