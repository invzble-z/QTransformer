#!/bin/bash

# PHASE 2 OPTIMIZATION SETUP
# Architecture enhancement + External data integration

echo "🚀 PHASE 2 OPTIMIZATION SETUP"
echo "============================="
echo "📊 Base: Phase 1 achieved 98.68% MSE improvement"
echo "🎯 Goal: Architecture scaling + External data"
echo ""

# Create Phase 2 directories
echo "📁 Setting up Phase 2 structure..."
mkdir -p logs/phase2_optimization
mkdir -p data_preprocessing/phase2
mkdir -p models/phase2
mkdir -p results/phase2

# Phase 2 Model Configuration - Scaled up
cat > scripts/long_term_forecast/QCAAPatchTF_Phase2_Enhanced.sh << 'EOF'
#!/bin/bash

# PHASE 2: Enhanced Architecture + External Data
# Target: MSE ≤ 200, MAE ≤ 10, MAPE ≤ 6%

model_name=QCAAPatchTF_Embedding
data_name=SupplyChainOptimized

# Enhanced Model Configuration - SCALED UP
seq_len=21
label_len=0
pred_len=7
enc_in=51           # Will increase to ~70-80 with external data
c_out=3
d_model=128         # Increased from 64
n_heads=16          # Increased from 8  
e_layers=4          # Increased from 3
d_ff=512            # Increased from 256
batch_size=16       # Reduced due to larger model

# Advanced Training Configuration
learning_rate=0.0005  # Lower for stability
train_epochs=100     # More epochs for complex model
patience=15          # More patience for convergence
dropout=0.15         # Slightly higher for regularization

# Data Configuration
data_path=supply_chain_phase2_enhanced.csv  # Will be created
target=order_count
features=MS
freq=d

# Advanced Features
channel_independence=1
factor=3

# Experiment Configuration
itr=1

# Create logs directory for Phase 2
LOG_DIR="./logs/phase2_optimization"
mkdir -p ${LOG_DIR}

# Generate timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/phase2_enhanced_${TIMESTAMP}.log"

echo "🚀 Starting Phase 2 Enhanced Training - $(date)" | tee ${LOG_FILE}
echo "📁 Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "📊 Dataset: ${data_path} (~80 features)" | tee -a ${LOG_FILE}
echo "🎯 Model: ${model_name} ENHANCED" | tee -a ${LOG_FILE}
echo "⚙️  Config: d_model=${d_model}, n_heads=${n_heads}, e_layers=${e_layers}" | tee -a ${LOG_FILE}
echo "🎪 Target: MSE ≤ 200, MAE ≤ 10, MAPE ≤ 6%" | tee -a ${LOG_FILE}
echo "=" | tee -a ${LOG_FILE}

# Run enhanced training
source .venv/bin/activate
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id SupplyChain_Phase2_Enhanced_v1 \
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
  --des 'Phase2_Enhanced_Architecture' \
  --itr $itr \
  --train_epochs $train_epochs \
  --patience $patience \
  --use_amp 2>&1 | tee -a ${LOG_FILE}

# Log completion
echo "" | tee -a ${LOG_FILE}
echo "🏁 Phase 2 Enhanced Training completed - $(date)" | tee -a ${LOG_FILE}
echo "📊 Log saved to: ${LOG_FILE}" | tee -a ${LOG_FILE}

# Show final results
echo "" | tee -a ${LOG_FILE}
echo "📈 PHASE 2 RESULTS SUMMARY:" | tee -a ${LOG_FILE}
echo "==========================" | tee -a ${LOG_FILE}
tail -20 ${LOG_FILE} | grep -E "(Original|Corrected|denormalized)" | tee -a ${LOG_FILE}

# Create symlink
ln -sf $(basename ${LOG_FILE}) ${LOG_DIR}/latest_phase2_training.log
echo "🔗 Symlink: ${LOG_DIR}/latest_phase2_training.log" | tee -a ${LOG_FILE}
EOF

chmod +x scripts/long_term_forecast/QCAAPatchTF_Phase2_Enhanced.sh

echo "✅ Phase 2 Enhanced script created"
echo ""

# Create Phase 2 Feature Engineering notebook template
cat > data_preprocessing/phase2/phase2_external_data_integration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 2: External Data Integration\n",
    "\n",
    "**Goal**: Add economic indicators, holidays, and advanced features\n",
    "**Target**: ~80 features (vs 51 current)\n",
    "**Expected**: MSE ≤ 200, MAE ≤ 10, MAPE ≤ 6%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from datetime import datetime, timedelta\n",
    "import holidays\n",
    "\n",
    "# Load Phase 1 optimized dataset\n",
    "df = pd.read_csv('../dataset/supply_chain_optimized.csv')\n",
    "print(f\"📊 Phase 1 dataset: {df.shape}\")\n",
    "print(f\"📋 Current features: {list(df.columns)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Economic Indicators"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Add economic indicators\n",
    "# - GDP growth rates by market\n",
    "# - Inflation rates\n",
    "# - Currency exchange rates\n",
    "# - Commodity prices\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Holiday and Calendar Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Add holiday features\n",
    "# - National holidays by market\n",
    "# - Shopping seasons\n",
    "# - Fiscal year effects\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Advanced Feature Interactions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Create interaction features\n",
    "# - Market x Product interactions\n",
    "# - Price x Market x Time\n",
    "# - Cross-market correlations\n",
    "pass"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✅ Phase 2 feature engineering notebook template created"
echo ""

# Create Phase 2 tracking file
cat > PHASE2_PROGRESS_TRACKING.md << 'EOF'
# 📊 PHASE 2 PROGRESS TRACKING

## 🎯 **OBJECTIVES**
- **MSE Target**: ≤ 200 (25% improvement từ 270)
- **MAE Target**: ≤ 10 (28% improvement từ 13.8)  
- **MAPE Target**: ≤ 6% (28% improvement từ 8.3%)
- **Features**: ~80 features (vs 51 current)

## 📋 **IMPLEMENTATION CHECKLIST**

### Week 1: Architecture Enhancement
- [ ] Scale model: d_model=128, d_ff=512, n_heads=16, e_layers=4
- [ ] Implement ensemble framework (5 models)
- [ ] Advanced attention mechanisms
- [ ] Batch size optimization for larger model

### Week 2: External Data Integration  
- [ ] Economic indicators (GDP, inflation, currency)
- [ ] Holiday calendars by market
- [ ] Commodity prices
- [ ] Supply chain event calendar

### Week 3: Advanced Features
- [ ] Feature interactions (Market x Product x Time)
- [ ] Cross-market correlation features
- [ ] Lag optimization analysis
- [ ] Feature importance analysis

### Week 4: Advanced Training
- [ ] Huber + Weighted loss combinations
- [ ] Temporal focus loss weighting
- [ ] Hyperparameter optimization (Optuna)
- [ ] Cross-validation framework

### Week 5: Production Ready
- [ ] Full pipeline automation
- [ ] Performance monitoring
- [ ] Documentation completion
- [ ] Deployment preparation

## 📈 **MILESTONES**

### Milestone 1: Architecture (Week 1)
**Target**: MSE ≤ 250, MAE ≤ 12

### Milestone 2: External Data (Week 2-3)  
**Target**: MSE ≤ 220, MAE ≤ 11

### Milestone 3: Advanced Features (Week 3-4)
**Target**: MSE ≤ 200, MAE ≤ 10

### Milestone 4: Production (Week 5)
**Target**: MSE ≤ 180, MAE ≤ 9, MAPE ≤ 6%

## 🚀 **NEXT ACTIONS**

1. **Immediate** (Next 2 days):
   - [ ] Scale up model architecture
   - [ ] Test enhanced configuration
   - [ ] Validate performance improvement

2. **Short-term** (Week 1):
   - [ ] Implement ensemble framework
   - [ ] External data source identification
   - [ ] Holiday calendar integration

3. **Medium-term** (Week 2-3):
   - [ ] Economic data integration
   - [ ] Feature interaction mining
   - [ ] Advanced loss functions

---
*Created: 2025-08-22 01:03:48*
*Phase 1 Baseline: MSE=270, MAE=13.8, MAPE=8.3%*
EOF

echo "✅ Phase 2 progress tracking created"
echo ""

echo "🎉 PHASE 2 SETUP COMPLETED!"
echo "=========================="
echo "📁 Files created:"
echo "   • scripts/long_term_forecast/QCAAPatchTF_Phase2_Enhanced.sh"
echo "   • data_preprocessing/phase2/phase2_external_data_integration.ipynb"
echo "   • PHASE2_PROGRESS_TRACKING.md"
echo ""
echo "🚀 Next steps:"
echo "   1. Wait for current Phase 1 training to complete"
echo "   2. Review final Phase 1 results"
echo "   3. Execute Phase 2 enhanced architecture"
echo ""
echo "📊 Phase 2 Targets:"
echo "   • MSE: 270 → ≤200 (25% improvement)"
echo "   • MAE: 13.8 → ≤10 (28% improvement)"  
echo "   • MAPE: 8.3% → ≤6% (28% improvement)"
