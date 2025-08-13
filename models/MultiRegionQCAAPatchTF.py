import torch
import torch.nn as nn
from models.QCAAPatchTF import Model as QCAAPatchTFModel

class MultiRegionModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_regions = configs.num_regions
        self.base_model = QCAAPatchTFModel(configs)
        
        # Region-specific projection heads
        self.region_heads = nn.ModuleList([
            nn.Linear(configs.d_model, configs.pred_len) 
            for _ in range(self.num_regions)
        ])
        
        # Region embedding
        self.region_embedding = nn.Embedding(self.num_regions, configs.d_model)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, region_ids):
        # Get base features from QCAAPatchTF
        base_features = self.base_model.encoder_forward(x_enc, x_mark_enc)
        
        # Add region-specific information
        region_embeds = self.region_embedding(region_ids)
        enhanced_features = base_features + region_embeds
        
        # Generate predictions for each region
        predictions = []
        for i, region_head in enumerate(self.region_heads):
            region_mask = (region_ids == i)
            if region_mask.any():
                region_features = enhanced_features[region_mask]
                region_pred = region_head(region_features)
                predictions.append(region_pred)
        
        return torch.cat(predictions, dim=0)