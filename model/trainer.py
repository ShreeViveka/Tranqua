"""
trainer.py — Model Training Script
====================================
Trains the GRU + Fusion model on the Combined Data.csv mental health dataset.

What this does:
  1. Loads Combined Data.csv (your Kaggle dataset)
  2. Builds vocabulary from all statements
  3. Trains the TextGRU on labelled statements
  4. Saves the trained model to model/saved_model.pt
  5. Saves vocabulary to model/vocab.pkl

After training, the predictor.py uses the saved model to
make daily predictions on new diary entries.

Run this ONCE (takes ~10-20 minutes on CPU):
  python model/trainer.py --data path/to/Combined\ Data.csv

Then the model is ready for daily use.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import (
    Vocabulary, tokenize, tokens_to_ids, text_to_ids,
    LABEL2IDX, IDX2LABEL, LABELS,
    DEFAULT_CONFIG, save_config, load_config,
    VOCAB_PATH, CONFIG_PATH
)
from model import FusionModel, build_model, save_model, DEVICE

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'training.log'))
    ]
)
log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model.pt')


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset for the Combined Data.csv.
    Each item is a (token_ids, dummy_numeric_features, label_idx) tuple.

    Note: During training on the public dataset, we don't have real
    numerical features — we use zeros. The model learns to rely on text.
    During real use, both text AND numerical features are available,
    and the fusion gate learns to combine them appropriately.
    """

    def __init__(self, texts: list[str], labels: list[int],
                 vocab: Vocabulary, max_len: int = 128, num_features: int = 10):
        self.texts       = texts
        self.labels      = labels
        self.vocab       = vocab
        self.max_len     = max_len
        self.num_features= num_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text    = self.texts[idx]
        label   = self.labels[idx]
        tokens  = tokenize(text)
        ids     = tokens_to_ids(tokens, self.vocab.word2idx, self.max_len)

        # During training on public dataset: use zeros for numerical features
        # During real prediction: real features are passed in from MySQL
        num_feats = np.zeros(self.num_features, dtype=np.float32)

        return {
            'token_ids'   : torch.tensor(ids,       dtype=torch.long),
            'num_features': torch.tensor(num_feats, dtype=torch.float),
            'label'       : torch.tensor(label,     dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> tuple[list, list]:
    """Load and clean the Combined Data.csv."""
    log.info(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate columns
    if 'statement' not in df.columns or 'status' not in df.columns:
        raise ValueError(f"Expected 'statement' and 'status' columns. Got: {df.columns.tolist()}")

    # Clean
    df = df.dropna(subset=['statement', 'status'])
    df = df.drop_duplicates(subset=['statement'])
    df = df[df['status'].isin(LABELS)]
    df = df.reset_index(drop=True)

    texts  = df['statement'].tolist()
    labels = [LABEL2IDX[s] for s in df['status']]

    log.info(f"Dataset loaded: {len(texts):,} samples")
    log.info("Class distribution:")
    for label, idx in LABEL2IDX.items():
        count = labels.count(idx)
        pct   = count / len(labels) * 100
        log.info(f"  {label:<25} {count:>6,} ({pct:.1f}%)")

    return texts, labels


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, criterion, scheduler=None):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for batch in dataloader:
        token_ids    = batch['token_ids'].to(DEVICE)
        num_features = batch['num_features'].to(DEVICE)
        labels       = batch['label'].to(DEVICE)

        optimizer.zero_grad()

        output = model(token_ids, num_features)
        loss   = criterion(output['logits'], labels)
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds       = output['probs'].argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def eval_epoch(model, dataloader, criterion):
    """Run evaluation on validation set."""
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            token_ids    = batch['token_ids'].to(DEVICE)
            num_features = batch['num_features'].to(DEVICE)
            labels       = batch['label'].to(DEVICE)

            output = model(token_ids, num_features)
            loss   = criterion(output['logits'], labels)

            total_loss += loss.item()
            preds       = output['probs'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return total_loss / len(dataloader), accuracy, all_preds, all_labels


def train(csv_path: str, config: dict = None):
    """Full training pipeline."""

    if config is None:
        config = DEFAULT_CONFIG.copy()

    log.info("=" * 60)
    log.info("  Mental Health Tracker — Model Training")
    log.info("=" * 60)
    log.info(f"  Device      : {DEVICE}")
    log.info(f"  Epochs      : {config['epochs']}")
    log.info(f"  Batch size  : {config['batch_size']}")
    log.info(f"  Learning rate: {config['learning_rate']}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    texts, labels = load_dataset(csv_path)

    # ── 2. Build vocabulary ───────────────────────────────────────────────────
    vocab = Vocabulary(min_freq=2, max_size=config['vocab_size'])
    vocab.build(texts)
    vocab.save(VOCAB_PATH)

    # Update config with actual vocab size
    config['vocab_size'] = len(vocab)
    save_config(config, CONFIG_PATH)

    # ── 3. Split data ─────────────────────────────────────────────────────────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size    = 0.15,
        random_state = 42,
        stratify     = labels    # keep class ratios balanced
    )
    log.info(f"Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    # ── 4. Create datasets & dataloaders ─────────────────────────────────────
    train_ds = MentalHealthDataset(train_texts, train_labels, vocab,
                                   config['max_seq_len'], config['num_features'])
    val_ds   = MentalHealthDataset(val_texts,   val_labels,   vocab,
                                   config['max_seq_len'], config['num_features'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    # ── 5. Build model ────────────────────────────────────────────────────────
    model = build_model(config)
    sizes = model.get_model_size()
    log.info(f"Model parameters: {sizes['total_params']:,}")

    # ── 6. Loss function with class weights (handles imbalanced dataset) ──────
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes      = np.array(range(len(LABELS))),
        y            = np.array(train_labels)
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # ── 7. Optimizer + scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config['learning_rate'],
        weight_decay = 0.01
    )
    # Cosine annealing: gradually reduces learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    # ── 8. Training loop ──────────────────────────────────────────────────────
    best_val_acc  = 0
    patience_left = config['early_stop_patience']
    history       = []

    log.info("\nStarting training...\n")

    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)

        # Validate
        val_loss, val_acc, val_preds, val_labels_list = eval_epoch(
            model, val_loader, criterion
        )

        scheduler.step()

        history.append({
            'epoch'     : epoch,
            'train_loss': round(train_loss, 4),
            'train_acc' : round(train_acc,  4),
            'val_loss'  : round(val_loss,   4),
            'val_acc'   : round(val_acc,    4),
        })

        log.info(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            patience_left = config['early_stop_patience']
            save_model(model, MODEL_PATH)
            log.info(f"  ✅ New best! Val Acc: {val_acc:.4f} — model saved.")
        else:
            patience_left -= 1
            log.info(f"  No improvement. Patience: {patience_left}/{config['early_stop_patience']}")
            if patience_left == 0:
                log.info("Early stopping triggered.")
                break

    # ── 9. Final evaluation ───────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    log.info(f"{'='*60}\n")

    # Classification report
    log.info("Classification Report (on validation set):")
    report = classification_report(
        val_labels_list, val_preds,
        target_names = LABELS,
        digits       = 4
    )
    log.info(f"\n{report}")

    # Save training history
    history_path = os.path.join(os.path.dirname(__file__), 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved to {history_path}")

    return best_val_acc, history


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Mental Health GRU model')
    parser.add_argument('--data',   type=str, required=True,
                        help='Path to Combined Data.csv')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch',  type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr',     type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['epochs']        = args.epochs
    config['batch_size']    = args.batch
    config['learning_rate'] = args.lr

    best_acc, history = train(args.data, config)
    print(f"\nFinal best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vocab saved to: {VOCAB_PATH}")
    print("\nNext step: Run python model/predictor.py to test predictions!")
