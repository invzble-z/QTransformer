#!/bin/bash

model_name=QCAAPatchTF_Embedding

# Multi-region Supply Chain Forecasting with Embedding
echo "Starting Multi-Region Supply Chain Forecasting with Embedding Approach..."

python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path seller_Order_Region_processed.csv \
  --model_id SupplyChain_MultiRegion_Embedding \
  --model $model_name \
  --channel_independence 1 \
  --data SupplyChainEmbedding \
  --features MS \
  --target "Order Item Quantity" \
  --seq_len 30 \
  --label_len 30 \
  --pred_len 7 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --des 'EmbeddingApproach' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --num_workers 0



# Alternative: Pure Multi-Region approach
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path seller_Order_Region_processed.csv \
#   --model_id SupplyChain_MultiRegion_Pure \
#   --model $model_name \
#   --channel_independence 0 \
#   --data MultiRegionEmbedding \
#   --features MS \
#   --seq_len 60 \
#   --label_len 30 \
#   --pred_len 7 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 1 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --des 'PureMultiRegionExp' \
#   --itr 1 \
#   --batch_size 16 \
#   --learning_rate 0.0001 \
#   --train_epochs 10
