from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_SupplyChain #  Dataset_ETT_minute, , Dataset_M4, PSMSegLoader, \
    # MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.data_loader_embedding import Dataset_SupplyChain_Embedding, Dataset_MultiRegion_Embedding
# from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch

def embedding_collate_fn(batch):
    """Custom collate function for embedding-based datasets"""
    if len(batch[0]) == 5:  # Has categorical features
        seq_x, seq_y, seq_x_mark, seq_y_mark, categorical_features = zip(*batch)
        
        # Stack regular tensors
        seq_x = torch.stack([torch.FloatTensor(x) for x in seq_x])
        seq_y = torch.stack([torch.FloatTensor(y) for y in seq_y])
        seq_x_mark = torch.stack([torch.FloatTensor(x) for x in seq_x_mark])
        seq_y_mark = torch.stack([torch.FloatTensor(y) for y in seq_y_mark])
        
        # Handle categorical features
        batch_categorical = {}
        if categorical_features[0]:  # Check if categorical features exist
            for key in categorical_features[0].keys():
                batch_categorical[key] = torch.stack([cat_feat[key] for cat_feat in categorical_features])
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, batch_categorical
    else:
        # Standard collate for non-embedding datasets
        return torch.utils.data.dataloader.default_collate(batch)

# data_dict = {
#     'ETTh1': Dataset_ETT_hour,
#     'ETTh2': Dataset_ETT_hour,
#     'ETTm1': Dataset_ETT_minute,
#     'ETTm2': Dataset_ETT_minute,
#     'custom': Dataset_Custom,
#     'm4': Dataset_M4,
#     'PSM': PSMSegLoader,
#     'MSL': MSLSegLoader,
#     'SMAP': SMAPSegLoader,
#     'SMD': SMDSegLoader,
#     'SWAT': SWATSegLoader,
#     'UEA': UEAloader
# }

data_dict = {
    'SmartLogistics': Dataset_Custom,
    'SupplyChain': Dataset_SupplyChain,
    'SupplyChainEmbedding': Dataset_SupplyChain_Embedding,
    'MultiRegionEmbedding': Dataset_MultiRegion_Embedding
}



def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    # # elif args.task_name == 'classification':
    # if args.task_name == 'classification':
    #     drop_last = False
    #     data_set = Data(
    #         args = args,
    #         root_path=args.root_path,
    #         flag=flag,
    #     )

    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last,
    #         collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    #     )
    #     return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        
        # Use custom collate function for embedding datasets
        collate_fn = embedding_collate_fn if args.data in ['SupplyChainEmbedding', 'MultiRegionEmbedding'] else None
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn)
        return data_set, data_loader
