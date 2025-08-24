#!/bin/bash

# Supply Chain Multi-Market Forecasting với Dataset Optimized - Giai đoạn 1
# Dataset: supply_chain_optimized.csv (765 records, 51 features, 3 markets)
# Target: Multi-market order_count prediction [7_days, 3_markets] với target đã normalize

model_name=QCAAPatchTF_Embedding
data_name=SupplyChainOptimized

# Model Configuration
seq_len=21          # Input sequence length (3 weeks)
label_len=0         # No label length needed for forecasting
pred_len=7          # Prediction length (1 week ahead)
enc_in=51           # Number of input features (đã tăng từ 21 lên 51)
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
data_path=supply_chain_optimized.csv
target=order_count
features=MS         # Multivariate forecasting with univariate target
freq=d              # Daily frequency

# Channel independence for better multi-variate handling
channel_independence=1
factor=3

# Experiment Configuration
itr=1               # Number of iterations

# Create logs directory
LOG_DIR="./logs/phase1_optimization"
mkdir -p ${LOG_DIR}

# Create results directory
mkdir -p ./results/${model_name}/${data_name}/

# Generate timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/phase1_training_${TIMESTAMP}.log"

echo "🚀 Starting Phase 1 Optimized Training - $(date)" | tee ${LOG_FILE}
echo "📁 Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "📊 Dataset: ${data_path} (51 features)" | tee -a ${LOG_FILE}
echo "🎯 Model: ${model_name} with WeightedMSELoss" | tee -a ${LOG_FILE}
echo "⚙️  Config: seq_len=${seq_len}, pred_len=${pred_len}, enc_in=${enc_in}, c_out=${c_out}" | tee -a ${LOG_FILE}
echo "=" | tee -a ${LOG_FILE}

# Run the experiment
source .venv/bin/activate
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id SupplyChain_Optimized_Phase1_v1 \
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
  --des 'Phase1_Optimized_SupplyChain_Forecasting' \
  --itr $itr \
  --train_epochs $train_epochs \
  --patience $patience \
  --use_amp 2>&1 | tee -a ${LOG_FILE}

# Log completion
echo "" | tee -a ${LOG_FILE}
echo "🏁 Training completed - $(date)" | tee -a ${LOG_FILE}
echo "📊 Log saved to: ${LOG_FILE}" | tee -a ${LOG_FILE}

# Show final results summary
echo "" | tee -a ${LOG_FILE}
echo "📈 FINAL RESULTS SUMMARY:" | tee -a ${LOG_FILE}
echo "=========================" | tee -a ${LOG_FILE}
tail -20 ${LOG_FILE} | grep -E "(Test Loss|MSE|MAE|Best)" | tee -a ${LOG_FILE}

# Create symlink to latest log
ln -sf $(basename ${LOG_FILE}) ${LOG_DIR}/latest_phase1_training.log
echo "🔗 Symlink created: ${LOG_DIR}/latest_phase1_training.log" | tee -a ${LOG_FILE}
