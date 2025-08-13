import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_multi_region_data(df):
    """Prepare data for multi-region forecasting"""
    
    # Label encoding thay vì one-hot
    categorical_columns = ['Order Region', 'Customer Country', 'Market', 'Shipping Mode']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Create time series data grouped by region
    regions = df['Order Region'].unique()
    regional_data = []
    
    for region in regions:
        region_df = df[df['Order Region'] == region].copy()
        
        # Aggregate by date
        daily_data = region_df.groupby('order date (DateOrders)').agg({
            'Order Item Quantity': 'sum',
            'Order Item Product Price': 'mean',
            'Order Item Discount Rate': 'mean',
            'Order Item Profit Ratio': 'mean',
            'Order Profit Per Order': 'mean',
            'Order Region_encoded': 'first'
        }).reset_index()
        
        regional_data.append(daily_data)
    
    # Combine all regions with proper indexing
    combined_data = pd.concat(regional_data, ignore_index=True)
    combined_data = combined_data.sort_values('order date (DateOrders)')
    
    return combined_data, label_encoders, regions