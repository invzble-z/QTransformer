#!/bin/bash

# Supply Chain Multi-Market Forecasting using QCAAPatchTF_Embedding
# Dataset: supply_chain_processed.csv (765 records, 21 features, 3 markets)
# Target: Multi-market order_count prediction [7_days, 3_markets]

model_name=QCAAPatchTF_Embedding
data_name=SupplyChainProcessed

# Model Configuration
seq_len=21          # Input sequence length (3 weeks)
label_len=0         # No label length needed for forecasting
pred_len=7          # Prediction length (1 week ahead)
enc_in=21           # Number of input features
c_out=3             # Number of output markets (Europe, LATAM, USCA)
d_model=64          # Model dimension
n_heads=8           # Number of attention heads
e_layers=3          # Number of encoder layers
d_ff=256            # Feed-forward dimension
batch_size=32       # Batch size

# Training Configuration
learning_rate=0.001
train_epochs=50
patience=10
dropout=0.1

# Data Configuration
data_path=supply_chain_processed.csv
target=order_count
features=MS         # Multivariate forecasting with univariate target
freq=d              # Daily frequency

# Channel independence for better multi-variate handling
channel_independence=1
factor=3

# Experiment Configuration
itr=1               # Number of iterations

# Create results directory
mkdir -p ./results/${model_name}/${data_name}/

# Run the experiment
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model $model_name \
  --data $data_name \
  --features $features \
  --target $target \
  --freq $freq \
  --checkpoints ./checkpoints/ \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --c_out $c_out \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_ff $d_ff \
  --dropout $dropout \
  --factor $factor \
  --channel_independence $channel_independence \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --des 'MultiMarket_SupplyChain_Forecasting' \
  --itr $itr \
  --train_epochs $train_epochs \
  --patience $patience \
  --use_amp
