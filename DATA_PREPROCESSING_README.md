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
    'Order_Item_Quantity': 'count',  # Target: daily order count
    'Days_for_shipping_real': 'mean',
    'Order_Item_Product_Price': 'mean',
    'Order_Item_Discount_Rate': 'mean',
    'Order_Item_Profit_Ratio': 'mean',
    'Order_Profit_Per_Order': 'mean',
    'Late_delivery_risk': 'mean'
}).reset_index()
```

#### **Timeline Cleaning:**
- **Valid start date:** First day all 3 markets have ≥10 orders
- **Expected removal:** ~30-60 initial days (unbalanced period)
- **Final timeline:** ~250-280 training days

### 2. **FEATURE ENGINEERING**

#### **Input Features (15 total):**

**Numerical Features (9):**
```python
numerical_features = [
    'days_shipping_avg',           # Average shipping days
    'late_delivery_rate',          # Risk of late delivery (%)
    'product_price_avg',           # Average product price
    'discount_rate_avg',           # Average discount rate
    'profit_ratio_avg',            # Average profit ratio
    'profit_per_order_avg',        # Average profit per order
    'order_quantity_total',        # Total quantity (not count)
    'customer_segment_diversity',   # Simpson's diversity index
    'category_diversity_index'     # Product category diversity
]
```

**Time Features (6):**
```python
time_features = [
    'day_of_week',      # 0-6 (Monday=0)
    'day_of_month',     # 1-31
    'month',            # 1-12
    'quarter',          # 1-4
    'is_weekend',       # Boolean
    'days_since_start'  # Trend component
]
```

**Categorical Features (Embedded):**
```python
categorical_features = {
    'Market': 3,              # USCA, LATAM, Europe
    'Shipping_Mode_dominant': 4  # Most frequent mode per day
}
```

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
# Create complete date range
full_date_range = pd.date_range(start_date, end_date, freq='D')

# Fill missing days with interpolation
for market in markets:
    market_data = market_data.reindex(full_date_range)
    market_data = market_data.interpolate(method='linear')
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
# Market embedding
market_embed_dim = 16
market_embedding = nn.Embedding(3, market_embed_dim)

# Shipping mode embedding  
shipping_embed_dim = 8
shipping_embedding = nn.Embedding(4, shipping_embed_dim)

# Total embedded dimension: 16 + 8 = 24
# Combined with 9 numerical + 6 time = 39 total features
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
    'enc_in': 39,            # Total input features
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

### **Step 1: Data Preparation**
```python
def preprocess_supply_chain_data(df):
    # 1. Clean timeline and filter valid dates
    df_clean = filter_valid_timeline(df)
    
    # 2. Daily aggregation by market
    daily_agg = aggregate_daily_features(df_clean)
    
    # 3. Handle outliers and missing values
    daily_clean = clean_outliers_and_missing(daily_agg)
    
    # 4. Feature engineering
    daily_featured = engineer_features(daily_clean)
    
    # 5. Create time series format
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
- **Features:** 39 features (9 numerical + 6 time + 24 embedded)
- **Loss:** Hybrid weighted (time + market importance)
- **Validation:** Time series split

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

*Last updated: August 20, 2025*  
*Author: AI Assistant*  
*Project: QTransformer Supply Chain Forecasting*
