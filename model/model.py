"""
model.py — GRU + Fusion Model Architecture
===========================================
Three components working together:

  1. TextGRU       — processes diary entry text sequences
  2. NumericalMLP  — processes laptop usage & health numbers
  3. FusionModel   — combines both and makes final prediction

Architecture diagram:

  Diary text tokens                Numerical features
  [128 token IDs]                  [screen_time, social_media,
        ↓                           work_time, idle_time,
  Embedding layer                   keystrokes, breaks,
  (vocab_size × embed_dim)          late_night, active_time,
        ↓                           mouse_dist, plugged_in]
  GRU (2 layers, hidden=256)              ↓
        ↓                          MLP (Dense layers)
  Text vector [256]                       ↓
        ↓                          Numeric vector [64]
        └──────────┬───────────────┘
                   ↓
            Concatenate [320]
                   ↓
          Attention weighting
                   ↓
          Fusion Dense [128]
                   ↓
          Output [7 classes]
          (softmax probabilities)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

# ── Device setup ──────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — TEXT BRANCH (GRU)
# ════════════════════════════════════════════════════════════════════════════

class TextGRU(nn.Module):
    """
    Processes diary text as a sequence of word embeddings through a GRU.

    Why GRU over LSTM?
    - Fewer parameters → trains faster on small personal data
    - Performs equally well for short sequences (diary entries)
    - Better for on-device federated learning (lighter model)

    Why 2 layers?
    - Layer 1: learns basic word patterns ("feeling anxious", "can't sleep")
    - Layer 2: learns higher-level emotional patterns across the entry
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_size: int, num_layers: int, dropout: float):
        super().__init__()

        # Embedding: converts token IDs to dense vectors
        # padding_idx=0 means PAD tokens contribute zero gradient
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = 0
        )

        # GRU: reads the sequence left to right AND right to left (bidirectional)
        # bidirectional=True means it reads the diary entry forwards AND backwards
        # This helps because "I am not feeling good" needs both directions to understand
        self.gru = nn.GRU(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,       # input shape: (batch, seq_len, embed_dim)
            bidirectional = True,        # doubles the output size
            dropout       = dropout if num_layers > 1 else 0
        )

        # After bidirectional GRU, output is hidden_size * 2
        self.output_dim = hidden_size * 2

        # Dropout for regularization (prevents overfitting on small data)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len) — integer token IDs
        returns  : (batch_size, hidden_size*2) — text representation vector
        """
        # Step 1: Convert token IDs → embeddings
        # Shape: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(token_ids))

        # Step 2: Pass through GRU
        # gru_out: (batch, seq_len, hidden*2) — output at every position
        # hidden : (num_layers*2, batch, hidden) — final hidden states
        gru_out, hidden = self.gru(embedded)

        # Step 3: Use attention over all positions instead of just last hidden
        # This lets the model focus on the most emotionally significant parts
        # of the diary entry, not just the ending
        text_vector = self._attention_pool(gru_out)

        # Step 4: Normalize + dropout
        return self.layer_norm(self.dropout(text_vector))

    def _attention_pool(self, gru_out: torch.Tensor) -> torch.Tensor:
        """
        Simple self-attention: learn which positions in the diary are most
        important for predicting mental state.

        E.g., "Today was okay but at night I felt really hopeless" —
        the model should attend more to "hopeless" than "okay".
        """
        # Score each position: (batch, seq_len, hidden*2) → (batch, seq_len, 1)
        scores  = torch.tanh(gru_out)
        weights = F.softmax(scores.mean(dim=-1, keepdim=True), dim=1)

        # Weighted sum: (batch, seq_len, hidden*2) × (batch, seq_len, 1)
        attended = (gru_out * weights).sum(dim=1)
        return attended


# ════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — NUMERICAL BRANCH (MLP)
# ════════════════════════════════════════════════════════════════════════════

class NumericalMLP(nn.Module):
    """
    Processes the laptop usage & health features through a small MLP.

    Input features (10 total):
      [0]  total_screen_time_mins  (normalized 0-1)
      [1]  social_media_mins       (normalized 0-1)
      [2]  work_app_mins           (normalized 0-1)
      [3]  entertainment_mins      (normalized 0-1)
      [4]  idle_time_mins          (normalized 0-1)
      [5]  keystrokes_count        (normalized 0-1)
      [6]  break_count             (normalized 0-1)
      [7]  late_night_usage_mins   (normalized 0-1)
      [8]  active_time_mins        (normalized 0-1)
      [9]  mouse_distance_px       (normalized 0-1)

    All values are normalized to [0, 1] before being passed in.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3):
        super().__init__()

        self.output_dim = hidden_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size * 2),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch_size, num_features) — normalized health/usage numbers
        returns : (batch_size, hidden_size)  — numerical representation vector
        """
        return self.network(features)


# ════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — FUSION MODEL
# ════════════════════════════════════════════════════════════════════════════

class FusionModel(nn.Module):
    """
    Combines the text vector from GRU and numerical vector from MLP.

    Uses a learned attention gate to decide HOW MUCH to trust each branch.
    - If the diary entry is very expressive: text weight → high
    - If the diary entry is short but usage data is extreme: numeric weight → high

    This is the key innovation — the model learns per-user which signals
    are more predictive of THEIR mental state.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # ── Text branch ──────────────────────────────────────────────────────
        self.text_branch = TextGRU(
            vocab_size  = config['vocab_size'],
            embed_dim   = config['embed_dim'],
            hidden_size = config['gru_hidden'],
            num_layers  = config['gru_layers'],
            dropout     = config['gru_dropout'],
        )

        # ── Numerical branch ─────────────────────────────────────────────────
        self.num_branch = NumericalMLP(
            input_size  = config['num_features'],
            hidden_size = config['num_hidden'],
            dropout     = config['gru_dropout'],
        )

        # ── Fusion attention gate ─────────────────────────────────────────────
        # Learns to weight text vs numerical based on context
        combined_dim = self.text_branch.output_dim + self.num_branch.output_dim

        self.attention_gate = nn.Sequential(
            nn.Linear(combined_dim, 2),   # 2 weights: text and numeric
            nn.Softmax(dim=-1)
        )

        # ── Fusion classifier ─────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, config['fusion_hidden']),
            nn.ReLU(),
            nn.Dropout(config['gru_dropout']),
            nn.LayerNorm(config['fusion_hidden']),

            nn.Linear(config['fusion_hidden'], config['fusion_hidden'] // 2),
            nn.ReLU(),
            nn.Dropout(config['gru_dropout']),
        )

        self.classifier = nn.Linear(config['fusion_hidden'] // 2, config['num_classes'])

    def forward(self, token_ids: torch.Tensor,
                num_features: torch.Tensor) -> dict:
        """
        token_ids   : (batch, seq_len)     — diary entry as token IDs
        num_features: (batch, num_features) — normalized laptop usage stats

        Returns dict with:
          'logits'       : raw scores for each class
          'probs'        : softmax probabilities
          'text_weight'  : how much the text influenced this prediction
          'num_weight'   : how much the numerical data influenced this prediction
        """
        # ── Get branch representations ────────────────────────────────────────
        text_vec = self.text_branch(token_ids)     # (batch, gru_hidden*2)
        num_vec  = self.num_branch(num_features)   # (batch, num_hidden)

        # ── Concatenate ───────────────────────────────────────────────────────
        combined = torch.cat([text_vec, num_vec], dim=-1)   # (batch, combined_dim)

        # ── Attention gate: how much to trust each branch ─────────────────────
        gate_weights = self.attention_gate(combined)        # (batch, 2)
        text_w = gate_weights[:, 0:1]                       # (batch, 1)
        num_w  = gate_weights[:, 1:2]                       # (batch, 1)

        # Scale branch vectors by their attention weights
        text_vec_scaled = text_vec * text_w.expand_as(
            text_vec[:, :text_vec.shape[1]]
        )
        num_vec_scaled = num_vec * num_w.expand_as(num_vec)

        # Recombine with weights applied
        weighted = torch.cat([text_vec_scaled, num_vec_scaled], dim=-1)

        # ── Fusion + classification ────────────────────────────────────────────
        fused  = self.fusion(weighted)
        logits = self.classifier(fused)
        probs  = F.softmax(logits, dim=-1)

        return {
            'logits'      : logits,
            'probs'       : probs,
            'text_weight' : text_w.squeeze(-1).mean().item(),
            'num_weight'  : num_w.squeeze(-1).mean().item(),
        }

    def predict_single(self, token_ids: list[int],
                       num_features: list[float]) -> dict:
        """
        Convenience method for predicting a single diary entry.
        Returns human-readable prediction dict.
        """
        from preprocessor import IDX2LABEL, LABEL_COLORS, LABEL_EMOJI

        self.eval()
        with torch.no_grad():
            ids_tensor  = torch.tensor([token_ids],    dtype=torch.long).to(DEVICE)
            feat_tensor = torch.tensor([num_features], dtype=torch.float).to(DEVICE)

            output = self.forward(ids_tensor, feat_tensor)
            probs  = output['probs'][0].cpu().numpy()

        pred_idx   = probs.argmax()
        pred_label = IDX2LABEL[pred_idx]
        confidence = float(probs[pred_idx])

        scores = {IDX2LABEL[i]: float(probs[i]) for i in range(len(probs))}

        return {
            'predicted_state': pred_label,
            'confidence'     : round(confidence, 4),
            'scores'         : scores,
            'text_weight'    : round(output['text_weight'], 4),
            'num_weight'     : round(output['num_weight'],  4),
            'color'          : LABEL_COLORS.get(pred_label, '#888'),
            'emoji'          : LABEL_EMOJI.get(pred_label, '🧠'),
        }

    def get_model_size(self) -> dict:
        """Return model parameter counts."""
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text   = sum(p.numel() for p in self.text_branch.parameters())
        num    = sum(p.numel() for p in self.num_branch.parameters())
        fusion = total - text - num
        return {
            'total_params'  : total,
            'trainable'     : train,
            'text_branch'   : text,
            'numeric_branch': num,
            'fusion'        : fusion,
        }


# ════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════════

def build_model(config: dict) -> FusionModel:
    """Build and return the model on the appropriate device."""
    model = FusionModel(config).to(DEVICE)
    sizes = model.get_model_size()
    log.info(f"[Model] Built FusionModel | "
             f"Total params: {sizes['total_params']:,} | "
             f"Text: {sizes['text_branch']:,} | "
             f"Numeric: {sizes['numeric_branch']:,} | "
             f"Device: {DEVICE}")
    return model


def save_model(model: FusionModel, path: str):
    """Save model weights to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config'          : model.config,
    }, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    log.info(f"[Model] Saved to {path} ({size_mb:.1f} MB)")


def load_model(path: str) -> FusionModel:
    """Load model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run trainer.py first.")

    checkpoint = torch.load(path, map_location=DEVICE)
    model      = FusionModel(checkpoint['config']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    log.info(f"[Model] Loaded from {path}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessor import DEFAULT_CONFIG

    print(f"Device: {DEVICE}")
    print("Building model...")

    model = build_model(DEFAULT_CONFIG)
    sizes = model.get_model_size()

    print(f"\nModel Architecture:")
    print(f"  Total parameters : {sizes['total_params']:,}")
    print(f"  Text branch (GRU): {sizes['text_branch']:,}")
    print(f"  Numeric branch   : {sizes['numeric_branch']:,}")
    print(f"  Fusion layers    : {sizes['fusion']:,}")

    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    batch_size  = 4
    seq_len     = 128
    num_feats   = 10

    dummy_tokens = torch.randint(0, 1000, (batch_size, seq_len)).to(DEVICE)
    dummy_nums   = torch.rand(batch_size, num_feats).to(DEVICE)

    output = model(dummy_tokens, dummy_nums)
    print(f"  Input tokens shape : {dummy_tokens.shape}")
    print(f"  Input numeric shape: {dummy_nums.shape}")
    print(f"  Output probs shape : {output['probs'].shape}")
    print(f"  Text weight        : {output['text_weight']:.4f}")
    print(f"  Numeric weight     : {output['num_weight']:.4f}")
    print(f"\n[OK] Model architecture is correct!")
