"""
Test FOU-Inspired Models with Synthetic Data
============================================

Quick verification that all components work before running on real data.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_hierarchical_model():
    """Test HierarchicalSepsisLSTM architecture."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Hierarchical Model Architecture")
    logger.info("=" * 60)
    
    from src.models.hierarchical_lstm import HierarchicalSepsisLSTM
    from src.config import get_default_config
    
    config = get_default_config().lstm
    
    # Test without static features
    logger.info("\n1a. Testing temporal-only model...")
    model = HierarchicalSepsisLSTM(config, static_dim=0)
    logger.info(model.summary())
    
    batch_size = 32
    x_temporal = torch.randn(batch_size, 6, 12)
    
    output = model(x_temporal, return_embedding=True, return_attention=True, level=3)
    
    assert output['logits_l1'].shape == (batch_size, 1)
    assert output['logits_l2'].shape == (batch_size, 1)
    assert output['logits_l3'].shape == (batch_size, 1)
    assert output['embedding'].shape == (batch_size, 16)
    assert output['spatial_attention'].shape == (batch_size, 12)
    assert output['temporal_attention'].shape == (batch_size, 6)
    
    logger.info("✅ Temporal-only model passed all checks")
    
    # Test with static features
    logger.info("\n1b. Testing multimodal model...")
    model_mm = HierarchicalSepsisLSTM(config, static_dim=20)
    logger.info(model_mm.summary())
    
    x_static = torch.randn(batch_size, 20)
    output_mm = model_mm(x_temporal, x_static, return_embedding=True, level=3)
    
    assert output_mm['embedding'].shape == (batch_size, 16)
    logger.info("✅ Multimodal model passed all checks")
    
    return True


def test_hierarchical_loss():
    """Test HierarchicalLoss."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Hierarchical Loss Function")
    logger.info("=" * 60)
    
    from src.training.train_hierarchical import HierarchicalLoss
    
    criterion = HierarchicalLoss(
        alpha_pos=0.9,
        alpha_neg=0.1,
        gamma=2.0,
        level_weights=(1.0, 0.5, 0.25),
    )
    
    batch_size = 32
    logits_l1 = torch.randn(batch_size, 1)
    logits_l2 = torch.randn(batch_size, 1)
    logits_l3 = torch.randn(batch_size, 1)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Test all levels
    loss_dict = criterion(logits_l1, labels, logits_l2, labels, logits_l3, labels)
    
    assert 'total' in loss_dict
    assert 'l1' in loss_dict
    assert 'l2' in loss_dict
    assert 'l3' in loss_dict
    
    logger.info(f"Loss L1: {loss_dict['l1'].item():.4f}")
    logger.info(f"Loss L2: {loss_dict['l2'].item():.4f}")
    logger.info(f"Loss L3: {loss_dict['l3'].item():.4f}")
    logger.info(f"Total:   {loss_dict['total'].item():.4f}")
    
    # Test single level
    loss_dict_l1 = criterion(logits_l1, labels)
    assert 'total' in loss_dict_l1
    assert 'l1' in loss_dict_l1
    
    logger.info("✅ Hierarchical loss passed all checks")
    
    return True


def test_training_loop():
    """Test training loop with synthetic data."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Training Loop (5 epochs)")
    logger.info("=" * 60)
    
    from src.training.train_hierarchical import HierarchicalTrainer
    from src.config import get_default_config
    
    # Create synthetic data
    n_train, n_val = 1000, 200
    X_train = torch.randn(n_train, 6, 12)
    y_train = (torch.rand(n_train) > 0.75).long()
    X_val = torch.randn(n_val, 6, 12)
    y_val = (torch.rand(n_val) > 0.75).long()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Configure for quick test
    config = get_default_config()
    config.training.max_epochs = 5
    config.training.early_stopping_patience = 3
    config.training.use_wandb = False
    
    # Train
    trainer = HierarchicalTrainer(config, static_dim=0, device='cpu')
    results = trainer.train(train_loader, val_loader, use_hierarchy=True)
    
    logger.info(f"\nTraining Results:")
    logger.info(f"  Best Val AUROC: {results['best_val_auroc']:.4f}")
    logger.info(f"  Best Epoch: {results['best_epoch']}")
    logger.info(f"  Total Epochs: {results['total_epochs']}")
    
    # Test evaluation
    test_metrics = trainer.evaluate(val_loader, use_hierarchy=True)
    logger.info(f"\nTest Metrics:")
    logger.info(f"  AUROC L1: {test_metrics['test_auroc_l1']:.4f}")
    logger.info(f"  AUROC L2: {test_metrics['test_auroc_l2']:.4f}")
    logger.info(f"  AUROC L3: {test_metrics['test_auroc_l3']:.4f}")
    
    # Test embedding extraction
    embeddings, labels = trainer.extract_embeddings(val_loader)
    assert embeddings.shape == (n_val, 16)
    assert labels.shape == (n_val,)
    
    logger.info(f"\nEmbeddings: {embeddings.shape}")
    logger.info(f"  Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    logger.info("✅ Training loop passed all checks")
    
    return True


def test_spatial_attention():
    """Test spatial attention mechanism."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Spatial Attention Mechanism")
    logger.info("=" * 60)
    
    from src.models.hierarchical_lstm import SpatialAttention
    
    attention = SpatialAttention(n_features=12, attention_dim=32)
    
    batch_size = 16
    seq_len = 6
    n_features = 12
    
    x = torch.randn(batch_size, seq_len, n_features)
    x_weighted, weights = attention(x)
    
    assert x_weighted.shape == (batch_size, seq_len, n_features)
    assert weights.shape == (batch_size, seq_len, n_features)
    
    # Check that weights sum to 1 over features
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones(batch_size, seq_len), atol=1e-5)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {x_weighted.shape}")
    logger.info(f"Attention weights shape: {weights.shape}")
    logger.info(f"Weights sum check: {weights_sum[0, 0]:.6f} (should be 1.0)")
    
    # Check that attention modifies input
    assert not torch.allclose(x, x_weighted)
    
    logger.info("✅ Spatial attention passed all checks")
    
    return True


def test_quantum_kernel_integration():
    """Test quantum kernel with hierarchical embeddings."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Quantum Kernel Integration")
    logger.info("=" * 60)
    
    from src.models.quantum_kernel import QuantumKernelSepsis
    from src.config import get_default_config
    
    # Create synthetic embeddings
    n_train, n_val, n_test = 400, 80, 120
    train_embeddings = np.random.randn(n_train, 16).astype(np.float32)
    train_labels = np.random.randint(0, 2, n_train).astype(np.int8)
    val_embeddings = np.random.randn(n_val, 16).astype(np.float32)
    val_labels = np.random.randint(0, 2, n_val).astype(np.int8)
    test_embeddings = np.random.randn(n_test, 16).astype(np.float32)
    test_labels = np.random.randint(0, 2, n_test).astype(np.int8)
    
    # Save synthetic embeddings
    embeddings_path = Path("data/processed/_test_hierarchical_embeddings.npz")
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        embeddings_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )
    
    # Test quantum kernel pipeline
    config = get_default_config()
    quantum = QuantumKernelSepsis(config.quantum, random_state=42)
    
    # Load and subsample
    data = quantum.load_embeddings(str(embeddings_path))
    X_train_sub, y_train_sub, _ = quantum.balanced_subsample(
        data['train_embeddings'],
        data['train_labels'],
        max_samples=200,
    )
    
    # PCA
    X_train_pca = quantum.fit_pca(X_train_sub)
    X_test_pca = quantum.transform_pca(data['test_embeddings'])
    
    logger.info(f"Original dim: 16")
    logger.info(f"PCA dim: {X_train_pca.shape[1]}")
    logger.info(f"Explained variance: {quantum.pca.explained_variance_ratio_.sum():.4f}")
    
    # Train (RBF kernel, no Qiskit)
    train_metrics = quantum.fit_rbf(X_train_pca, y_train_sub, tune_hyperparameters=False)
    
    logger.info(f"\nTraining Metrics:")
    logger.info(f"  AUROC: {train_metrics['train_auroc']:.4f}")
    logger.info(f"  Support vectors: {train_metrics['train_support_vectors']}")
    
    # Evaluate
    test_metrics = quantum.evaluate(X_test_pca, data['test_labels'], prefix='test_')
    
    logger.info(f"\nTest Metrics:")
    logger.info(f"  AUROC: {test_metrics['test_auroc']:.4f}")
    logger.info(f"  AUPRC: {test_metrics['test_auprc']:.4f}")
    
    # Cleanup
    embeddings_path.unlink()
    
    logger.info("✅ Quantum kernel integration passed all checks")
    
    return True


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("FOU-INSPIRED MODEL TESTS")
    logger.info("=" * 70)
    
    tests = [
        ("Hierarchical Model", test_hierarchical_model),
        ("Hierarchical Loss", test_hierarchical_loss),
        ("Spatial Attention", test_spatial_attention),
        ("Training Loop", test_training_loop),
        ("Quantum Integration", test_quantum_kernel_integration),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ {name} FAILED: {e}", exc_info=True)
            results[name] = f"❌ FAILED: {str(e)}"
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for name, result in results.items():
        logger.info(f"{name:30s}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Ready to run on real data.")
        return 0
    else:
        logger.error("\n⚠️  Some tests failed. Fix issues before running on real data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
