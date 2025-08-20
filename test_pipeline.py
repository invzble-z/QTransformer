#!/usr/bin/env python3
"""
Pipeline Testing for QCAAPatchTF_Embedding Multi-Market Supply Chain Forecasting
Author: GitHub Copilot
Date: 2025-08-21

This file tests the complete training pipeline step by step:
1. Data loading and preprocessing validation
2. Model initialization and architecture validation
3. Forward pass testing
4. Training loop testing
5. Multi-market output validation
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime

# Add project root to path
sys.path.append('.')

def print_step(step_num, description):
    """Print formatted step header"""
    print(f"\n{'='*60}")
    print(f"🔧 STEP {step_num}: {description}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def test_step_1_data_loading():
    """Test data loading and basic validation"""
    print_step(1, "DATA LOADING & VALIDATION")
    
    # Check dataset exists
    dataset_path = './dataset/supply_chain_processed.csv'
    if not os.path.exists(dataset_path):
        print_error(f"Dataset not found: {dataset_path}")
        return False
    
    # Load and inspect dataset
    df = pd.read_csv(dataset_path)
    print_success(f"Dataset loaded: {df.shape}")
    
    print_info(f"Columns: {df.columns.tolist()}")
    print_info(f"Market distribution: {df['Market'].value_counts().to_dict()}")
    print_info(f"Date range: {df['order_date_only'].min()} to {df['order_date_only'].max()}")
    
    # Check required columns
    required_cols = ['order_date_only', 'Market', 'order_count', 'Market_encoded']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print_error(f"Missing required columns: {missing_cols}")
        return False
    
    print_success("All required columns present")
    return True

def test_step_2_data_loader():
    """Test custom data loader functionality"""
    print_step(2, "DATA LOADER TESTING")
    
    from data_provider.data_loader_supply_chain import Dataset_SupplyChain_Processed
    from data_provider.data_factory import data_provider
    
    print_success("Data loader imports successful")
    
    # Create test arguments
    args = argparse.Namespace()
    args.task_name = 'long_term_forecast'  # Missing attribute causing the error
    args.data = 'SupplyChainProcessed'
    args.root_path = './dataset/'
    args.data_path = 'supply_chain_processed.csv'
    args.seq_len = 21
    args.label_len = 0
    args.pred_len = 7
    args.features = 'MS'
    args.target = 'order_count'
    args.embed = 'timeF'
    args.freq = 'd'
    args.seasonal_patterns = None
    args.batch_size = 4  # Small batch for testing
    args.num_workers = 0
    
    # Test data loading
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    print_success(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Test batch loading
    for i, batch in enumerate(train_loader):
        if len(batch) == 5:
            seq_x, seq_y, seq_x_mark, seq_y_mark, categorical_features = batch
            print_success("Batch format correct (5 elements)")
            print_info(f"   seq_x: {seq_x.shape} (expected: [batch, 21, 21])")
            print_info(f"   seq_y: {seq_y.shape} (expected: [batch, 7, 1])")
            print_info(f"   categorical_features: {list(categorical_features.keys()) if categorical_features else 'None'}")
            
            if categorical_features and "Market_encoded" in categorical_features:
                print_info(f"   Market_encoded: {categorical_features['Market_encoded'].shape}")
                print_info(f"   Market values: {categorical_features['Market_encoded'][:, -1].tolist()}")
            
            break
        else:
            print_error(f"Unexpected batch format: {len(batch)} elements")
            return False
    
    return True

def test_step_3_model_initialization():
    """Test model initialization and architecture"""
    print_step(3, "MODEL INITIALIZATION")
    
    from models.QCAAPatchTF_Embedding import QCAAPatchTF_Embedding
    
    # Create test config
    config = argparse.Namespace()
    config.task_name = 'long_term_forecast'
    config.seq_len = 21
    config.pred_len = 7
    config.enc_in = 21
    config.c_out = 3  # Multi-market output
    config.d_model = 64
    config.n_heads = 8
    config.e_layers = 3
    config.d_ff = 256
    config.dropout = 0.1
    config.factor = 3
    config.channel_independence = 1
    config.activation = 'gelu'  # Missing activation attribute
    config.categorical_dims = {'Market_encoded': 3}
    
    # Initialize model
    model = QCAAPatchTF_Embedding(config)
    print_success("Model initialized successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_info(f"Total parameters: {total_params:,}")
    print_info(f"Trainable parameters: {trainable_params:,}")
    
    # Test model architecture
    print_info(f"Patch length: {model.patch_len}")
    print_info(f"Patch number: {model.patch_num}")
    print_info(f"Output markets: {model.c_out}")
    
    return model, config

def test_step_4_forward_pass(model, config):
    """Test model forward pass with sample data"""
    print_step(4, "FORWARD PASS TESTING")
    
    if model is None:
        print_error("Model not available for testing")
        return False
    
    # Create sample input data
    batch_size = 2
    seq_x = torch.randn(batch_size, config.seq_len, config.enc_in)
    seq_x_mark = torch.randn(batch_size, config.seq_len, 4)  # Time features
    seq_y = torch.randn(batch_size, config.pred_len, 1)
    seq_y_mark = torch.randn(batch_size, config.pred_len, 4)
    
    # Sample categorical features
    categorical_features = {
        'Market_encoded': torch.randint(0, 3, (batch_size, config.seq_len))
    }
    
    print_info(f"Input shapes:")
    print_info(f"   seq_x: {seq_x.shape}")
    print_info(f"   categorical Market_encoded: {categorical_features['Market_encoded'].shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(seq_x, seq_x_mark, seq_y, seq_y_mark, categorical_features)
    
    print_success(f"Forward pass successful!")
    print_info(f"Output shape: {output.shape} (expected: [{batch_size}, {config.pred_len}, {config.c_out}])")
    
    # Validate output dimensions
    expected_shape = (batch_size, config.pred_len, config.c_out)
    if output.shape == expected_shape:
        print_success("Output shape matches expected multi-market format")
    else:
        print_error(f"Output shape mismatch! Expected: {expected_shape}, Got: {output.shape}")
        return False
    
    # Check output values
    print_info(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print_info(f"Output mean: {output.mean().item():.4f}")
    
    return True

def test_step_5_training_compatibility():
    """Test training loop compatibility"""
    print_step(5, "TRAINING COMPATIBILITY CHECK")
    
    try:
        from exp.exp_long_term_forecasting_embedding import Exp_Long_Term_Forecast_Embedding
        
        # Create experiment args
        args = argparse.Namespace()
        args.task_name = 'long_term_forecast'
        args.model = 'QCAAPatchTF_Embedding'
        args.data = 'SupplyChainProcessed'
        args.root_path = './dataset/'
        args.data_path = 'supply_chain_processed.csv'
        args.seq_len = 21
        args.label_len = 0
        args.pred_len = 7
        args.enc_in = 21
        args.c_out = 3
        args.d_model = 64
        args.n_heads = 8
        args.e_layers = 3
        args.d_ff = 256
        args.dropout = 0.1
        args.factor = 3
        args.channel_independence = 1
        args.batch_size = 4
        args.learning_rate = 0.001
        args.features = 'MS'
        args.target = 'order_count'
        args.embed = 'timeF'
        args.freq = 'd'
        args.seasonal_patterns = None
        args.checkpoints = './test_checkpoints/'
        args.num_workers = 0
        args.patience = 3
        args.train_epochs = 2  # Very short for testing
        args.des = 'test_run'
        args.itr = 1
        args.use_amp = False
        args.use_gpu = torch.cuda.is_available()
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0'
        
        # Initialize experiment
        exp = Exp_Long_Term_Forecast_Embedding(args)
        print_success("Experiment initialized successfully")
        
        # Test getting data loaders
        train_data, train_loader = exp._get_data(flag='train')
        val_data, val_loader = exp._get_data(flag='val')
        
        print_success(f"Data loaders created - Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Test model building
        model = exp._build_model()
        print_success("Model built successfully through experiment")
        
        return True
        
    except Exception as e:
        print_error(f"Training compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_test():
    """Run complete pipeline test"""
    print(f"\n🚀 STARTING FULL PIPELINE TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run tests step by step
    results = {}
    
    # Step 1: Data Loading
    print("\n" + "="*60)
    print("Starting Step 1: Data Loading")
    print("="*60)
    results['data_loading'] = test_step_1_data_loading()
    if not results['data_loading']:
        print_error("❌ CRITICAL: Data loading failed. Cannot continue.")
        return False
    
    # Step 2: Data Loader
    print("\n" + "="*60)
    print("Starting Step 2: Data Loader")
    print("="*60)
    results['data_loader'] = test_step_2_data_loader()
    if not results['data_loader']:
        print_error("❌ CRITICAL: Data loader failed. Cannot continue.")
        return False
    
    # Step 3: Model Initialization
    print("\n" + "="*60)
    print("Starting Step 3: Model Initialization")
    print("="*60)
    model, config = test_step_3_model_initialization()
    results['model_init'] = model is not None
    if not results['model_init']:
        print_error("❌ CRITICAL: Model initialization failed. Cannot continue.")
        return False
    
    # Step 4: Forward Pass
    print("\n" + "="*60)
    print("Starting Step 4: Forward Pass")
    print("="*60)
    results['forward_pass'] = test_step_4_forward_pass(model, config)
    
    # Final summary
    print_step("FINAL", "TEST SUMMARY")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print_success("🎉 ALL TESTS PASSED! Pipeline is ready for training.")
        return True
    else:
        print_error(f"⚠️  {total_tests - passed_tests} test(s) failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
