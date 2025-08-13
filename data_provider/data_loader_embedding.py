import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import torch

warnings.filterwarnings('ignore')

class Dataset_SupplyChain_Embedding(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='seller_Order_Region_processed.csv',
                 target='Order Item Quantity', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 if args.seq_len is None else args.seq_len
            self.label_len = 24 if args.label_len is None else args.label_len
            self.pred_len = 24 if args.pred_len is None else args.pred_len
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
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
        
        # Load encoders and categorical dimensions
        base_name = data_path.replace('_processed.csv', '')
        encoder_path = os.path.join(root_path, f'{base_name}_encoders.pkl')
        
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoder_data = pickle.load(f)
                self.label_encoders = encoder_data['label_encoders']
                self.categorical_dims = encoder_data['categorical_dims']
                self.feature_scaler = encoder_data['scaler']
        else:
            self.label_encoders = {}
            self.categorical_dims = {}
            self.feature_scaler = None
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Convert date column
        date_col = 'order date (DateOrders)'
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        df_raw = df_raw.sort_values(date_col)
        
        # Separate categorical and numerical columns
        categorical_cols = [col for col in df_raw.columns if col.endswith('_encoded')]
        numerical_cols = [col for col in df_raw.columns if col not in [date_col, self.target] + categorical_cols]
        
        # Store column information
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        # Prepare features based on 'features' parameter
        cols_data = df_raw[date_col]
        
        if self.features == 'M':
            # Multivariate - exclude target 
            feature_cols = numerical_cols + categorical_cols
            df_data = df_raw[feature_cols]
        elif self.features == 'MS':
            # Multivariate with target
            feature_cols = numerical_cols + categorical_cols
            df_data = df_raw[feature_cols]
            # Add target as first column
            target_col = df_raw[[self.target]]
            df_data = pd.concat([target_col, df_data], axis=1)
        elif self.features == 'S':
            # Univariate
            df_data = df_raw[[self.target]]
        
        # Train/Val/Test split
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Scale numerical features only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            if len(numerical_cols) > 0:
                numerical_data = train_data[numerical_cols]
                self.scaler.fit(numerical_data.values)
                
        # Prepare final data
        self.data_x = df_data[border1:border2]
        self.data_y = df_raw[self.target][border1:border2]
        
        # Time encoding
        if self.timeenc == 0:
            self.data_stamp = time_features(pd.to_datetime(cols_data.values), freq=self.freq)
            self.data_stamp = self.data_stamp.transpose(1, 0)
        elif self.timeenc == 1:
            self.data_stamp = time_features(pd.to_datetime(cols_data.values), freq=self.freq)
            self.data_stamp = self.data_stamp.transpose(1, 0)
            
        self.data_stamp = self.data_stamp[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x.iloc[s_begin:s_end].values
        seq_y = self.data_y.iloc[r_begin:r_end].values
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # Ensure seq_y has correct shape
        if len(seq_y.shape) == 1:
            seq_y = seq_y.reshape(-1, 1)
        
        # Process categorical and numerical data separately
        categorical_data = {}
        numerical_data = []
        
        if len(self.categorical_cols) > 0:
            cat_indices = [self.data_x.columns.get_loc(col) for col in self.categorical_cols]
            cat_data = seq_x[:, cat_indices].astype(int)
            
            # Create categorical feature dict for embedding
            for i, col in enumerate(self.categorical_cols):
                categorical_data[col.replace('_encoded', '')] = torch.tensor(cat_data[:, i], dtype=torch.long)
                
        if len(self.numerical_cols) > 0:
            num_indices = [self.data_x.columns.get_loc(col) for col in self.numerical_cols]
            numerical_data = seq_x[:, num_indices]
            
            if self.scale:
                numerical_data = self.scaler.transform(numerical_data)
                
        # Combine for seq_x (for compatibility with existing model structure)
        if len(numerical_data) > 0:
            seq_x_processed = numerical_data
        else:
            seq_x_processed = np.zeros((seq_x.shape[0], 1))  # Placeholder if no numerical features

        return seq_x_processed, seq_y, seq_x_mark, seq_y_mark, categorical_data

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scale else data


class Dataset_MultiRegion_Embedding(Dataset):
    """Multi-region dataset that can predict for all regions simultaneously"""
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='seller_Order_Region_processed.csv',
                 target='Order Item Quantity', scale=True, timeenc=0, freq='d', 
                 region_col='Order_Region_encoded'):
        
        if size is None:
            self.seq_len = 24 if args.seq_len is None else args.seq_len
            self.label_len = 24 if args.label_len is None else args.label_len
            self.pred_len = 24 if args.pred_len is None else args.pred_len
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
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
        self.region_col = region_col
        
        # Load encoders
        base_name = data_path.replace('_processed.csv', '')
        encoder_path = os.path.join(root_path, f'{base_name}_encoders.pkl')
        
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoder_data = pickle.load(f)
                self.label_encoders = encoder_data['label_encoders']
                self.categorical_dims = encoder_data['categorical_dims']
                self.feature_scaler = encoder_data['scaler']
        else:
            self.label_encoders = {}
            self.categorical_dims = {}
            self.feature_scaler = None
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        date_col = 'order date (DateOrders)'
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        
        # Group by region and create separate time series for each region
        regions = df_raw[self.region_col].unique()
        self.num_regions = len(regions)
        self.regions = regions
        
        # Store regional data
        self.regional_data = {}
        
        for region in regions:
            region_df = df_raw[df_raw[self.region_col] == region].copy()
            region_df = region_df.sort_values(date_col)
            
            # Separate features
            categorical_cols = [col for col in region_df.columns if col.endswith('_encoded') and col != self.region_col]
            numerical_cols = [col for col in region_df.columns if col not in [date_col, self.target, self.region_col] + categorical_cols]
            
            # Prepare features
            if self.features == 'MS':
                feature_data = pd.concat([
                    region_df[[self.target]], 
                    region_df[numerical_cols],
                    region_df[categorical_cols]
                ], axis=1)
            elif self.features == 'M':
                feature_data = pd.concat([
                    region_df[numerical_cols],
                    region_df[categorical_cols]
                ], axis=1)
            else:  # 'S'
                feature_data = region_df[[self.target]]
                
            self.regional_data[region] = {
                'data': feature_data,
                'target': region_df[self.target],
                'timestamp': region_df[date_col],
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols
            }
        
        # Create combined dataset for training
        self._create_combined_sequences()
        
    def _create_combined_sequences(self):
        """Create sequences that can be used for multi-region training"""
        all_sequences = []
        all_targets = []
        all_timestamps = []
        all_regions = []
        
        for region_id, region_data in self.regional_data.items():
            data = region_data['data']
            target = region_data['target'] 
            timestamps = region_data['timestamp']
            
            # Create sequences for this region
            for i in range(len(data) - self.seq_len - self.pred_len + 1):
                seq = data.iloc[i:i+self.seq_len].values
                tgt = target.iloc[i+self.seq_len:i+self.seq_len+self.pred_len].values
                ts = timestamps.iloc[i:i+self.seq_len].values
                
                all_sequences.append(seq)
                all_targets.append(tgt)
                all_timestamps.append(ts)
                all_regions.append(region_id)
        
        self.sequences = np.array(all_sequences)
        self.targets = np.array(all_targets)
        self.timestamps = all_timestamps
        self.region_ids = np.array(all_regions)
        
        # Train/Val/Test split
        n_samples = len(self.sequences)
        num_train = int(n_samples * 0.7)
        num_test = int(n_samples * 0.2)
        
        if self.set_type == 0:  # train
            start, end = 0, num_train
        elif self.set_type == 1:  # val
            start, end = num_train, num_train + (n_samples - num_train - num_test)
        else:  # test
            start, end = n_samples - num_test, n_samples
            
        self.sequences = self.sequences[start:end]
        self.targets = self.targets[start:end]
        self.timestamps = self.timestamps[start:end]
        self.region_ids = self.region_ids[start:end]

    def __getitem__(self, index):
        seq_x = self.sequences[index]
        seq_y = self.targets[index]
        region_id = self.region_ids[index]
        
        # Time features (simplified)
        seq_x_mark = np.zeros((self.seq_len, 4))  # Placeholder for time features
        seq_y_mark = np.zeros((self.pred_len, 4))  # Placeholder for time features
        
        # Return region_id as additional info
        return seq_x, seq_y, seq_x_mark, seq_y_mark, {'region_id': torch.tensor(region_id, dtype=torch.long)}

    def __len__(self):
        return len(self.sequences)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scale else data
