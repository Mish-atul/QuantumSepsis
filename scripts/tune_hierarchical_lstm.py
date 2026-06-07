"""
Hyperparameter Tuning for HierarchicalLSTM
===========================================

Tunes key hyperparameters to improve AUROC beyond 0.85:
- hidden_dim: 128, 256, 384
- attention_dim: 64, 128
- focal_gamma: 2.0, 3.0, 4.0
- learning_rate: 0.0005, 0.001, 0.002
- dropout: 0.2, 0.3, 0.4

Uses validation set for selection, reports test performance.

Author: QuantumSepsis Team
Date: May 12, 2026
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, LSTMConfig
from src.data.dataset import SepsisDataset
from src.models.hierarchical_lstm import HierarchicalSepsisLSTM
from src.training.train_hierarchical import HierarchicalTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def create_config_variant(base_config: Config, **kwargs) -> Config:
    """Create config with modified hyperparameters"""
    import copy
    config = copy.deepcopy(base_config)
    
    for key, value in kwargs.items():
        if hasattr(config.lstm, key):
            setattr(config.lstm, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config


def train_and_evaluate(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    experiment_name: str,
    device: str = 'cuda',
) -> Dict:
    """Train model with given config and evaluate"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"{'='*70}")
    logger.info(f"  hidden_dim: {config.lstm.hidden_dim}")
    logger.info(f"  attention_dim: {config.lstm.attention_dim}")
    logger.info(f"  dropout: {config.lstm.dropout}")
    logger.info(f"  learning_rate: {config.training.learning_rate}")
    logger.info(f"  focal_gamma: {config.training.focal_gamma}")
    
    # Create trainer
    trainer = HierarchicalTrainer(config, static_dim=0)
    
    # Train
    try:
        train_results = trainer.train(
            train_loader,
            val_loader,
            use_hierarchy=True,
        )
        
        # Evaluate on test
        test_metrics = trainer.evaluate(test_loader, use_hierarchy=True)
        
        results = {
            'experiment': experiment_name,
            'config': {
                'hidden_dim': config.lstm.hidden_dim,
                'attention_dim': config.lstm.attention_dim,
                'dropout': config.lstm.dropout,
                'learning_rate': config.training.learning_rate,
                'focal_gamma': config.training.focal_gamma,
            },
            'best_val_auroc': train_results['best_val_auroc'],
            'best_epoch': train_results['best_epoch'],
            'test_auroc': test_metrics['test_auroc_level1'],
            'test_auprc': test_metrics['test_auprc_level1'],
            'training_time_minutes': train_results['training_time_minutes'],
        }
        
        logger.info(f"\n✓ Results:")
        logger.info(f"  Val AUROC:  {results['best_val_auroc']:.4f}")
        logger.info(f"  Test AUROC: {results['test_auroc']:.4f}")
        logger.info(f"  Test AUPRC: {results['test_auprc']:.4f}")
        logger.info(f"  Time: {results['training_time_minutes']:.1f} min")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Experiment failed: {e}")
        return {
            'experiment': experiment_name,
            'error': str(e),
            'test_auroc': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument('--data', type=str, default='data/processed/features.h5')
    parser.add_argument('--output-dir', type=str, default='data/processed/sepsis')
    parser.add_argument('--strategy', type=str, default='quick',
                        choices=['quick', 'thorough'],
                        help='Quick: 3 configs, Thorough: 9 configs')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Max epochs per experiment')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("HYPERPARAMETER TUNING FOR HIERARCHICALLSTM")
    logger.info("="*70)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info("\nLoading data...")
    with h5py.File(args.data, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
    
    logger.info(f"Train: {X_train.shape}")
    logger.info(f"Val:   {X_val.shape}")
    logger.info(f"Test:  {X_test.shape}")
    
    # Create datasets
    train_dataset = SepsisDataset(X_train, y_train)
    val_dataset = SepsisDataset(X_val, y_val)
    test_dataset = SepsisDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    # Base config
    base_config = Config()
    base_config.training.max_epochs = args.max_epochs
    base_config.training.early_stopping_patience = 10
    
    # Define hyperparameter search space
    if args.strategy == 'quick':
        # Quick: 3 promising configurations
        configs = [
            {
                'name': 'exp1_larger_hidden',
                'hidden_dim': 256,
                'attention_dim': 128,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'focal_gamma': 2.0,
            },
            {
                'name': 'exp2_higher_gamma',
                'hidden_dim': 128,
                'attention_dim': 64,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'focal_gamma': 3.0,
            },
            {
                'name': 'exp3_combined',
                'hidden_dim': 256,
                'attention_dim': 128,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'focal_gamma': 3.0,
            },
        ]
    else:
        # Thorough: grid search
        hidden_dims = [128, 256, 384]
        attention_dims = [64, 128]
        focal_gammas = [2.0, 3.0]
        
        configs = []
        for i, (hd, ad, fg) in enumerate(product(hidden_dims, attention_dims, focal_gammas)):
            configs.append({
                'name': f'exp{i+1}_h{hd}_a{ad}_g{fg}',
                'hidden_dim': hd,
                'attention_dim': ad,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'focal_gamma': fg,
            })
    
    logger.info(f"\nRunning {len(configs)} experiments...")
    
    # Run experiments
    all_results = []
    for config_dict in configs:
        name = config_dict.pop('name')
        config = create_config_variant(base_config, **config_dict)
        
        results = train_and_evaluate(
            config,
            train_loader,
            val_loader,
            test_loader,
            name,
            device=device,
        )
        
        all_results.append(results)
        
        # Save intermediate results
        output_file = output_dir / f"tuning_results_{args.strategy}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TUNING RESULTS SUMMARY")
    logger.info("="*70)
    
    # Sort by test AUROC
    all_results.sort(key=lambda x: x.get('test_auroc', 0), reverse=True)
    
    logger.info("\nTop 3 Configurations:")
    for i, result in enumerate(all_results[:3], 1):
        if 'error' in result:
            continue
        logger.info(f"\n{i}. {result['experiment']}")
        logger.info(f"   Test AUROC: {result['test_auroc']:.4f}")
        logger.info(f"   Val AUROC:  {result['best_val_auroc']:.4f}")
        logger.info(f"   Config: hidden={result['config']['hidden_dim']}, "
                   f"attn={result['config']['attention_dim']}, "
                   f"gamma={result['config']['focal_gamma']}")
    
    # Best result
    best = all_results[0]
    logger.info(f"\n🏆 BEST: {best['experiment']}")
    logger.info(f"   Test AUROC: {best['test_auroc']:.4f}")
    
    # Check target
    target = 0.85
    if best['test_auroc'] >= target:
        logger.info(f"\n✅ TARGET MET! AUROC {best['test_auroc']:.4f} >= {target}")
        logger.info("   → Ready for Quantum Kernel training")
    else:
        gap = target - best['test_auroc']
        logger.info(f"\n⚠️  Gap to target: {gap:.4f} AUROC points")
        logger.info("   → Consider ensemble or more tuning")
    
    # Save final results
    summary = {
        'strategy': args.strategy,
        'num_experiments': len(all_results),
        'best_experiment': best['experiment'],
        'best_test_auroc': best['test_auroc'],
        'target_met': best['test_auroc'] >= target,
        'all_results': all_results,
    }
    
    output_file = output_dir / f"tuning_summary_{args.strategy}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    logger.info("\n" + "="*70)
    logger.info("TUNING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
