import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# Read the CSV file
dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/70k_DataCoSupplyChainDataset.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
# Sắp xếp theo thời gian từ cũ đến mới
df = df.sort_values('order date (DateOrders)', ascending=True)

print(df.head())
print(f"Total number of orders: {len(df)}")
print(df.columns.to_list())

buyer_features = ['Customer Country', 'Customer Segment',
                  'Order Item Product Price', 'Order Item Discount Rate', 'Order Item Profit Ratio',
                  'Order Profit Per Order', 'Customer Id']

seller_features = ['Order Region', 'Market', 'Shipping Mode', 'Days for shipping (real)',
                   'Late_delivery_risk', 'Order Item Product Price',
                   'Order Item Discount Rate', 'Order Item Profit Ratio', 'Order Profit Per Order']

# Hàm tổng hợp chuỗi thời gian với COMPLETE DATE FILLING
def create_complete_time_series(df, group_cols, target_col='Order Item Quantity', freq='D'):
    """Tạo time series hoàn chỉnh với tất cả ngày, fill 0 cho missing dates"""
    
    print(f"Creating complete time series for grouping by: {group_cols}")
    
    # Tạo complete date range
    date_range = pd.date_range(
        start=df['order date (DateOrders)'].min(),
        end=df['order date (DateOrders)'].max(),
        freq=freq
    )
    
    # Get unique values for grouping columns
    unique_groups = df[group_cols].drop_duplicates()
    print(f"Found {len(unique_groups)} unique groups")
    
    all_complete_data = []
    
    for idx, group_values in unique_groups.iterrows():
        print(f"Processing group {idx+1}/{len(unique_groups)}: {dict(group_values)}")
        
        # Filter data for this group
        mask = True
        for col in group_cols:
            mask = mask & (df[col] == group_values[col])
        group_df = df[mask].copy()
        
        if len(group_df) == 0:
            continue
            
        # Aggregate daily data for this group
        daily_agg = group_df.groupby('order date (DateOrders)').agg({
            target_col: 'sum',  # Sum quantity for multiple orders same day
            **{col: 'mean' for col in df.columns if col not in ['order date (DateOrders)', target_col] + group_cols and df[col].dtype in ['float64', 'int64']},
            **{col: 'first' for col in df.columns if col not in ['order date (DateOrders)', target_col] + group_cols and df[col].dtype == 'object'}
        }).reset_index()
        
        # Create complete time series with all dates
        complete_df = pd.DataFrame({'order date (DateOrders)': date_range})
        
        # Add group values to all rows
        for col in group_cols:
            complete_df[col] = group_values[col]
        
        # Merge with actual data
        complete_df = complete_df.merge(daily_agg, on='order date (DateOrders)', how='left')
        
        # Fill missing values
        complete_df[target_col] = complete_df[target_col].fillna(0)  # No orders = 0 quantity
        
        # Fill other numerical columns with median/mean of this group
        numerical_cols = [col for col in complete_df.columns if complete_df[col].dtype in ['float64', 'int64'] and col != target_col]
        for col in numerical_cols:
            if col not in group_cols:  # Don't fill group columns
                group_median = daily_agg[col].median()
                complete_df[col] = complete_df[col].fillna(group_median)
        
        # Fill categorical columns with mode of this group
        categorical_cols = [col for col in complete_df.columns if complete_df[col].dtype == 'object']
        for col in categorical_cols:
            if col not in group_cols and col != 'order date (DateOrders)':
                group_mode = daily_agg[col].mode()
                if len(group_mode) > 0:
                    complete_df[col] = complete_df[col].fillna(group_mode[0])
        
        all_complete_data.append(complete_df)
    
    # Combine all groups
    final_df = pd.concat(all_complete_data, ignore_index=True)
    final_df = final_df.sort_values('order date (DateOrders)', ascending=True)
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Date range: {final_df['order date (DateOrders)'].min()} to {final_df['order date (DateOrders)'].max()}")
    print(f"Missing data filled successfully!")
    
    return final_df

# Tạo dữ liệu COMPLETE cho người mua và người bán
buyer_levels = [['Customer Country']]
seller_levels = [['Order Region']]

print("\n=== PROCESSING BUYER DATA ===")
for level in buyer_levels:
    print(f"\nProcessing buyer level: {level}")
    ts_data = create_complete_time_series(df, level)
    output_file = f'dataset/buyer_{"_".join(level).replace(" ", "_")}_complete.csv'
    ts_data.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

print("\n=== PROCESSING SELLER DATA ===")
for level in seller_levels:
    print(f"\nProcessing seller level: {level}")
    ts_data = create_complete_time_series(df, level)
    output_file = f'dataset/seller_{"_".join(level).replace(" ", "_")}_complete.csv'
    ts_data.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

# Mã hóa và chuẩn hóa sử dụng Label Encoding - FIXED VERSION
def preprocess_data_with_embedding_complete(file_path, features):
    """Enhanced preprocessing cho complete time series data"""
    df = pd.read_csv(file_path)
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df = df.sort_values('order date (DateOrders)', ascending=True)
    
    print(f"\nProcessing file: {file_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['order date (DateOrders)'].min()} to {df['order date (DateOrders)'].max()}")
    
    # Identify categorical and numerical columns
    available_features = [col for col in features if col in df.columns]
    cat_cols = [col for col in available_features if df[col].dtype == 'object']
    num_cols = [col for col in available_features if df[col].dtype != 'object']
    
    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")
    
    # Dictionary để lưu encoders và categorical dimensions
    encoders = {}
    categorical_dims = {}
    
    # Start with basic columns
    df_processed = df[['order date (DateOrders)', 'Order Item Quantity']].copy()
    
    # Label encoding cho categorical features
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        categorical_dims[col] = len(le.classes_)
        print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Chuẩn hóa numerical features
    scaler = StandardScaler()
    if num_cols:
        # Handle any remaining NaN values
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        
        scaled = scaler.fit_transform(df[num_cols])
        df_scaled = pd.DataFrame(scaled, columns=num_cols, index=df.index)
        df_processed = pd.concat([df_processed, df_scaled], axis=1)
        print(f"Scaled {len(num_cols)} numerical columns")
    
    # Check for any remaining NaN values
    nan_counts = df_processed.isnull().sum()
    if nan_counts.sum() > 0:
        print("Warning: Found NaN values after processing:")
        print(nan_counts[nan_counts > 0])
        # Fill remaining NaN with 0
        df_processed = df_processed.fillna(0)
    
    # Lưu encoders và scaler
    output_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace('.csv', '')
    
    with open(f'{output_dir}/{base_name}_encoders.pkl', 'wb') as f:
        pickle.dump({'label_encoders': encoders, 'scaler': scaler, 'categorical_dims': categorical_dims}, f)
    
    # Lưu processed data
    output_file = file_path.replace('.csv', '_processed.csv')
    df_processed.to_csv(output_file, index=False)
    print(f"Saved processed data to: {output_file}")
    
    # Return dimensions for model configuration
    total_features = len(cat_cols) + len(num_cols) + 1  # +1 cho target
    return total_features, categorical_dims

# Xử lý tất cả file với embedding approach - COMPLETE VERSION
buyer_files = [f'dataset/buyer_{"_".join(level).replace(" ", "_")}_complete.csv' for level in buyer_levels]
seller_files = [f'dataset/seller_{"_".join(level).replace(" ", "_")}_complete.csv' for level in seller_levels]

print("\n=== PROCESSING COMPLETE BUYER DATA ===")
buyer_results = []
for f in buyer_files:
    try:
        result = preprocess_data_with_embedding_complete(f, buyer_features)
        buyer_results.append(result)
    except Exception as e:
        print(f"Error processing {f}: {e}")
        buyer_results.append((0, {}))

print("\n=== PROCESSING COMPLETE SELLER DATA ===")
seller_results = []
for f in seller_files:
    try:
        result = preprocess_data_with_embedding_complete(f, seller_features)
        seller_results.append(result)
    except Exception as e:
        print(f"Error processing {f}: {e}")
        seller_results.append((0, {}))

print("\n=== FINAL RESULTS ===")
print("=== Complete Embedding-based Preprocessing Results ===")
for i, (total_features, cat_dims) in enumerate(buyer_results):
    if total_features > 0:
        print(f"Buyer Level {i+1}: Total features = {total_features}, Categorical dims = {cat_dims}")

for i, (total_features, cat_dims) in enumerate(seller_results):
    if total_features > 0:
        print(f"Seller Level {i+1}: Total features = {total_features}, Categorical dims = {cat_dims}")

print("\n✅ Complete time series data preprocessing finished!")
print("✅ All missing dates filled with 0 for Order Item Quantity")
print("✅ Model now can learn patterns including 'no orders' = 0")
