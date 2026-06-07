"""
Train FOU LSTM Model
====================
Training script for FOU detection using BiLSTM + Attention.

Multi-class classification (4 classes):
- 0: No FOU
- 1: Infectious FOU
- 2: Non-infectious FOU
- 3: Undiagnosed FOU

Usage:
    python scripts/train_fou_lstm.py --data data/processed/fou_features.h5 --epochs 100
"""

import sys
import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_default_fou_config
from src.models.lstm_fou import FouLSTM
from src.models.losses import MultiClassFocalLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str):
    """Load FOU features from HDF5."""
    logger.info(f"Loading data from {data_path}...")

    with h5py.File(data_path, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Train labels: {np.bincount(y_train)}")
        logger.info(f"Val labels: {np.bincount(y_val)}")
        logger.info(f"Test labels: {np.bincount(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits, probs, _, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, macro_f1


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, probs, _, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()

            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, macro_f1, weighted_f1, all_preds, all_labels, np.array(all_probs)


def extract_embeddings(model, dataloader, device):
    """Extract embeddings for quantum kernel."""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            _, _, embedding, _ = model(X_batch, return_embedding=True)
            embeddings.append(embedding.cpu().numpy())
            labels.append(y_batch.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description="Train FOU LSTM")
    parser.add_argument('--data', type=str, required=True, help='Path to FOU features HDF5')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/fou', help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='data/processed/fou', help='Output directory')
    args = parser.parse_args()

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = get_default_fou_config()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Model
    model = FouLSTM(
        input_size=config.lstm.input_size,
        seq_len=config.lstm.seq_len,
        hidden_dim=config.lstm.hidden_dim,
        n_layers=config.lstm.n_layers,
        bidirectional=config.lstm.bidirectional,
        dropout=config.lstm.dropout,
        attention_dim=config.lstm.attention_dim,
        fc1_dim=config.lstm.fc1_dim,
        embedding_dim=config.lstm.embedding_dim,
        n_classes=config.lstm.n_classes
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = MultiClassFocalLoss(
        alpha=config.training.focal_alpha,
        gamma=config.training.focal_gamma
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.scheduler_t_max
    )

    # Training loop
    best_macro_f1 = 0.0
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_macro_f1, val_weighted_f1, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step()

        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_macro_f1:.4f}, Weighted F1: {val_weighted_f1:.4f}")

        # Save best model
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            patience_counter = 0

            checkpoint_path = checkpoint_dir / "fou_lstm_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_macro_f1,
                'config': config
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (Macro F1: {val_macro_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(checkpoint_dir / "fou_lstm_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    logger.info("\nFinal test evaluation...")
    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )

    logger.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Macro F1: {test_macro_f1:.4f}, Weighted F1: {test_weighted_f1:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    class_names = ["No FOU", "Infectious", "Non-infectious", "Undiagnosed"]
    # Get unique labels in test set
    unique_labels = sorted(np.unique(test_labels))
    labels_present = [class_names[i] for i in unique_labels]
    print(classification_report(test_labels, test_preds, labels=unique_labels, target_names=labels_present, zero_division=0))

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds, labels=unique_labels)
    print(cm)

    # Extract embeddings
    logger.info("\nExtracting embeddings for quantum kernel...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    val_embeddings, val_labels = extract_embeddings(model, val_loader, device)
    test_embeddings, test_labels_emb = extract_embeddings(model, test_loader, device)

    # Save embeddings
    embeddings_path = output_dir / "fou_lstm_embeddings.npz"
    np.savez(
        embeddings_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels_emb
    )
    logger.info(f"Embeddings saved to {embeddings_path}")

    logger.info("\n✓ FOU LSTM training complete")


if __name__ == "__main__":
    main()
