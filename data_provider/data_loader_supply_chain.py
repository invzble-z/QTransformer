import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_SupplyChain_Processed(Dataset):
    """
    Dataset loader cho supply_chain_processed.csv
    Hỗ trợ multi-market forecasting với Market_encoded embedding
    """
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='supply_chain_processed.csv',
                 target='order_count', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        
        # Set sequence lengths
        if size is None:
            self.seq_len = 21 if args.seq_len is None else args.seq_len  # 3 tuần
            self.label_len = 21 if args.label_len is None else args.label_len
            self.pred_len = 7 if args.pred_len is None else args.pred_len  # Dự đoán 7 ngày
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # Initialize parameters
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        
        # Markets definition
        self.markets = ['Europe', 'LATAM', 'USCA']
        self.market_encoding = {'Europe': 0, 'LATAM': 1, 'USCA': 2}
        
        print(f"🔧 Dataset_SupplyChain_Processed initialized for {flag}")
        print(f"📊 Seq_len: {self.seq_len}, Pred_len: {self.pred_len}")
        print(f"🎯 Target: {self.target}, Features: {self.features}")
        
        self.__read_data__()

    def __read_data__(self):
        """Đọc và xử lý dữ liệu supply chain"""
        print(f"📂 Loading {self.data_path}...")
        
        # Load data
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        print(f"✅ Loaded {len(df_raw)} records")
        
        # Convert date column and sort
        df_raw['order_date_only'] = pd.to_datetime(df_raw['order_date_only'])
        df_raw = df_raw.sort_values(['order_date_only', 'Market'])
        
        # Rename date column for consistency with timefeatures
        df_raw.rename(columns={'order_date_only': 'date'}, inplace=True)
        
        # Define feature columns (excluding date, target, market info)
        exclude_cols = ['date', 'Market', self.target]
        if 'Market_encoded' in df_raw.columns:
            exclude_cols.append('Market_encoded')
            
        # Numerical features (21 features total)
        numerical_cols = [col for col in df_raw.columns if col not in exclude_cols]
        categorical_cols = ['Market_encoded'] if 'Market_encoded' in df_raw.columns else []
        
        print(f"📋 Found {len(numerical_cols)} numerical features: {numerical_cols[:5]}...")
        print(f"📋 Found {len(categorical_cols)} categorical features: {categorical_cols}")
        
        # Store column info
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Prepare feature matrix based on features parameter
        if self.features == 'M':
            # Multivariate - exclude target
            feature_cols = numerical_cols + categorical_cols
            df_data = df_raw[feature_cols]
        elif self.features == 'MS':
            # Multivariate with target (most common for forecasting)
            feature_cols = numerical_cols + categorical_cols
            df_data = df_raw[feature_cols]
            # Target as first column for compatibility
            target_col = df_raw[[self.target]]
            df_data = pd.concat([target_col, df_data], axis=1)
        elif self.features == 'S':
            # Univariate
            df_data = df_raw[[self.target]]
            
        # Data split: 80/10/10 theo thảo luận
        num_train = int(len(df_data) * 0.8)
        num_test = int(len(df_data) * 0.1)
        num_vali = len(df_data) - num_train - num_test
        
        # Borders for train/val/test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        print(f"📊 Data split - Train: {border1s[0]}-{border2s[0]}, Val: {border1s[1]}-{border2s[1]}, Test: {border1s[2]}-{border2s[2]}")
        
        # Scale numerical features
        self.scaler = StandardScaler()
        if self.scale and len(numerical_cols) > 0:
            train_data = df_data[border1s[0]:border2s[0]]
            numerical_data = train_data[numerical_cols] if self.features != 'S' else train_data
            self.scaler.fit(numerical_data.values)
            print("📈 Fitted scaler on training data")
        
        # Prepare final datasets
        self.data_x = df_data[border1:border2]
        self.data_y = df_raw[self.target][border1:border2]
        
        # Market information for embedding
        self.market_data = df_raw['Market'][border1:border2]
        if 'Market_encoded' in df_raw.columns:
            self.market_encoded_data = df_raw['Market_encoded'][border1:border2]
        else:
            # Create market encoding if not exists
            self.market_encoded_data = self.market_data.map(self.market_encoding)
        
        # Time features
        cols_data = df_raw['date'][border1:border2]
        if self.timeenc == 0:
            self.data_stamp = time_features(pd.to_datetime(cols_data.values), freq=self.freq)
            self.data_stamp = self.data_stamp.transpose(1, 0)
        elif self.timeenc == 1:
            self.data_stamp = time_features(pd.to_datetime(cols_data.values), freq=self.freq)
            self.data_stamp = self.data_stamp.transpose(1, 0)
            
        print(f"✅ Data preparation completed:")
        print(f"   📊 X shape: {self.data_x.shape}")
        print(f"   📊 Y shape: {self.data_y.shape}")
        print(f"   📊 Time stamp shape: {self.data_stamp.shape}")
        
    def __getitem__(self, index):
        """Lấy một sample cho training/validation/test"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Input sequence (features)
        seq_x = self.data_x.iloc[s_begin:s_end].values
        
        # Target sequence  
        seq_y = self.data_y.iloc[r_begin:r_end].values
        if len(seq_y.shape) == 1:
            seq_y = seq_y.reshape(-1, 1)
            
        # Time features
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # Market information for embedding
        market_encoded_seq = self.market_encoded_data.iloc[s_begin:s_end].values
        
        # Process features
        categorical_data = {}
        
        # Scale numerical features
        if self.scale and len(self.numerical_cols) > 0:
            if self.features == 'MS':
                # Skip target column (first column) when scaling
                numerical_part = seq_x[:, 1:1+len(self.numerical_cols)]
                numerical_part = self.scaler.transform(numerical_part)
                seq_x[:, 1:1+len(self.numerical_cols)] = numerical_part
            elif self.features == 'M':
                numerical_part = seq_x[:, :len(self.numerical_cols)]
                numerical_part = self.scaler.transform(numerical_part)
                seq_x[:, :len(self.numerical_cols)] = numerical_part
                
        # Prepare categorical features for embedding
        if len(self.categorical_cols) > 0:
            categorical_data['Market'] = torch.tensor(market_encoded_seq, dtype=torch.long)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, categorical_data

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse transform cho predictions"""
        return self.scaler.inverse_transform(data) if self.scale else data


class Dataset_SupplyChain_MultiMarket(Dataset):
    """
    Enhanced version hỗ trợ multi-market output trực tiếp
    Output shape: [batch, pred_len, 3_markets]
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='supply_chain_processed.csv',
                 target='order_count', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        
        self.base_dataset = Dataset_SupplyChain_Processed(
            args, root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns
        )
        
        # Copy necessary attributes
        for attr in ['seq_len', 'label_len', 'pred_len', 'markets', 'market_encoding']:
            setattr(self, attr, getattr(self.base_dataset, attr))
            
        print(f"🎯 Multi-market dataset initialized for {flag}")
        
    def __getitem__(self, index):
        """
        Enhanced getitem để support multi-market output
        """
        # Get base data
        seq_x, seq_y, seq_x_mark, seq_y_mark, categorical_data = self.base_dataset[index]
        
        # seq_y hiện tại: [seq_len, 1] 
        # Cần reshape để support multi-market: [pred_len, 3_markets]
        
        # For now, replicate single market prediction to 3 markets
        # This will be properly handled when we have restructured data
        if len(seq_y.shape) == 2 and seq_y.shape[1] == 1:
            # [pred_len, 1] -> [pred_len, 3]
            seq_y_multi = np.repeat(seq_y, 3, axis=1)
        else:
            seq_y_multi = seq_y
            
        return seq_x, seq_y_multi, seq_x_mark, seq_y_mark, categorical_data
    
    def __len__(self):
        return len(self.base_dataset)
    
    def inverse_transform(self, data):
        return self.base_dataset.inverse_transform(data)
