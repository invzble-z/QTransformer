import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# Read the CSV file
# dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/example_DataCoSupplyChainDataset.csv'
# dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/DataCoSupplyChainDataset.csv'
dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/70k_DataCoSupplyChainDataset.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
# Sắp xếp theo thời gian từ cũ đến mới
df = df.sort_values('order date (DateOrders)', ascending=True)


# Lọc đơn hàng hoàn thành
# df = df[df['Order Status'] == 'COMPLETE']
print(df.head())
print(f"Total number of orders: {len(df)}")
print(df.columns.to_list())



buyer_features = ['Customer Country', 'Customer Segment',
                  'Order Item Product Price', 'Order Item Discount Rate', 'Order Item Profit Ratio',
                  'Order Profit Per Order', 'Customer Id']

seller_features = ['Order Region', 'Market', 'Shipping Mode', 'Days for shipping (real)',
                   'Late_delivery_risk', 'Order Item Product Price',
                   'Order Item Discount Rate', 'Order Item Profit Ratio', 'Order Profit Per Order']

# Hàm tổng hợp chuỗi thời gian
def create_time_series(df, group_cols, target_col='Order Item Quantity', freq='D'):
    ts_data = df.groupby(group_cols + [pd.Grouper(key='order date (DateOrders)', freq=freq)])[target_col].sum().reset_index()
    features = [col for col in df.columns if col not in ['order date (DateOrders)', target_col] + group_cols]
    feature_data = df.groupby(group_cols + [pd.Grouper(key='order date (DateOrders)', freq=freq)])[features].first().reset_index()
    ts_data = ts_data.merge(feature_data, on=group_cols + ['order date (DateOrders)'], how='left')
    ts_data = ts_data.sort_values('order date (DateOrders)', ascending=True)
    return ts_data

# Tạo dữ liệu cho người mua và người bán
buyer_levels = [['Customer Country']]
seller_levels = [['Order Region']]

for level in buyer_levels:
    ts_data = create_time_series(df, level)
    ts_data.to_csv(f'dataset/buyer_{"_".join(level).replace(" ", "_")}.csv', index=False)

for level in seller_levels:
    ts_data = create_time_series(df, level)
    ts_data.to_csv(f'dataset/seller_{"_".join(level).replace(" ", "_")}.csv', index=False)



# Mã hóa và chuẩn hóa sử dụng Label Encoding
def preprocess_data_with_embedding(file_path, features):
    df = pd.read_csv(file_path)
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df = df.sort_values('order date (DateOrders)', ascending=True)
    
    cat_cols = [col for col in features if col in df.columns and df[col].dtype == 'object']
    num_cols = [col for col in features if col in df.columns and df[col].dtype != 'object']
    
    # Dictionary để lưu encoders và categorical dimensions
    encoders = {}
    categorical_dims = {}
    
    # Label encoding cho categorical features
    df_processed = df[['order date (DateOrders)', 'Order Item Quantity']].copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
        categorical_dims[col] = len(le.classes_)
    
    # Chuẩn hóa numerical features
    scaler = StandardScaler()
    if num_cols:
        scaled = scaler.fit_transform(df[num_cols])
        df_scaled = pd.DataFrame(scaled, columns=num_cols, index=df.index)
        df_processed = pd.concat([df_processed, df_scaled], axis=1)
    
    # Lưu encoders và scaler
    output_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace('.csv', '')
    
    with open(f'{output_dir}/{base_name}_encoders.pkl', 'wb') as f:
        pickle.dump({'label_encoders': encoders, 'scaler': scaler, 'categorical_dims': categorical_dims}, f)
    
    # Lưu processed data
    df_processed.to_csv(file_path.replace('.csv', '_processed.csv'), index=False)
    
    # Return dimensions for model configuration
    total_features = len(cat_cols) + len(num_cols) + 1  # +1 cho target
    return total_features, categorical_dims

# Mã hóa và chuẩn hóa (legacy function - kept for compatibility)
def preprocess_data(file_path, features):
    df = pd.read_csv(file_path)
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df = df.sort_values('order date (DateOrders)', ascending=True)
    
    cat_cols = [col for col in features if df[col].dtype == 'object']
    num_cols = [col for col in features if df[col].dtype != 'object']
    
    # Mã hóa đặc trưng danh mục với Label Encoding
    encoders = {}
    df_processed = df[['order date (DateOrders)', 'Order Item Quantity']].copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Chuẩn hóa đặc trưng số
    scaler = StandardScaler()
    if num_cols:
        scaled = scaler.fit_transform(df[num_cols])
        df_scaled = pd.DataFrame(scaled, columns=num_cols, index=df.index)
        df_processed = pd.concat([df_processed, df_scaled], axis=1)
    
    # Kết hợp
    df_processed.to_csv(file_path.replace('.csv', '_processed.csv'), index=False)
    return len(cat_cols) + len(num_cols) + 1  # +1 cho Order Item Quantity

# Xử lý tất cả file với embedding approach
buyer_files = [f'dataset/buyer_{"_".join(level).replace(" ", "_")}.csv' for level in buyer_levels]
seller_files = [f'dataset/seller_{"_".join(level).replace(" ", "_")}.csv' for level in seller_levels]

buyer_results = [preprocess_data_with_embedding(f, buyer_features) for f in buyer_files]
seller_results = [preprocess_data_with_embedding(f, seller_features) for f in seller_files]

print("=== Embedding-based Preprocessing Results ===")
for i, (total_features, cat_dims) in enumerate(buyer_results):
    print(f"Buyer Level {i+1}: Total features = {total_features}, Categorical dims = {cat_dims}")

for i, (total_features, cat_dims) in enumerate(seller_results):
    print(f"Seller Level {i+1}: Total features = {total_features}, Categorical dims = {cat_dims}")

# Legacy processing (commented out to avoid confusion)
# buyer_enc_in = [preprocess_data(f, buyer_features) for f in buyer_files]
# seller_enc_in = [preprocess_data(f, seller_features) for f in seller_files]
# print(f"Buyer encoded input dimensions: {buyer_enc_in}")
# print(f"Seller encoded input dimensions: {seller_enc_in}")
