"""
preprocessor.py — Text Preprocessing Pipeline
==============================================
Cleans and tokenizes diary entries before feeding into the GRU model.
Also builds and saves the vocabulary from the training dataset.

Used by:
  - trainer.py  (to build vocab from Combined Data.csv)
  - predictor.py (to preprocess new diary entries at runtime)
"""

import re
import os
import json
import pickle
import logging
from collections import Counter

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__))
VOCAB_PATH = os.path.join(MODEL_DIR, 'vocab.pkl')
CONFIG_PATH= os.path.join(MODEL_DIR, 'model_config.json')

# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = '<PAD>'   # index 0 — padding shorter sequences
UNK_TOKEN = '<UNK>'   # index 1 — unknown words not in vocab

# ── Label mapping ─────────────────────────────────────────────────────────────
LABELS = ['Normal', 'Depression', 'Suicidal', 'Anxiety',
          'Stress', 'Bipolar', 'Personality disorder']

LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}

# Colour coding for UI display
LABEL_COLORS = {
    'Normal'              : '#1D9E75',
    'Anxiety'             : '#E24B4A',
    'Stress'              : '#F4A261',
    'Depression'          : '#5B6EAE',
    'Bipolar'             : '#9B59B6',
    'Suicidal'            : '#C0392B',
    'Personality disorder': '#E67E22',
}

# Emoji for UI
LABEL_EMOJI = {
    'Normal'              : '😊',
    'Anxiety'             : '😰',
    'Stress'              : '😤',
    'Depression'          : '😔',
    'Bipolar'             : '🔄',
    'Suicidal'            : '🆘',
    'Personality disorder': '🌀',
}


# ════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean raw text for tokenization.
    Keeps contractions and emotional punctuation intact for GRU.
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+',  '', text)   # remove URLs
    text = re.sub(r'<.*?>',            '', text)   # remove HTML tags
    text = re.sub(r'@\w+',            '', text)    # remove @mentions
    text = re.sub(r'#(\w+)',    r'\1', text)        # keep hashtag word
    text = re.sub(r"'",         "'",   text)        # normalize apostrophes
    text = re.sub(r'[^a-z\s\'\-]', ' ', text)      # keep letters + ' and -
    text = re.sub(r'\s+',       ' ',   text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens."""
    return clean_text(text).split()


def tokens_to_ids(tokens: list[str], vocab: dict,
                  max_len: int = 128) -> list[int]:
    """
    Convert tokens to integer IDs using vocabulary.
    Pads or truncates to max_len.
    """
    unk_id = vocab.get(UNK_TOKEN, 1)
    ids    = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]

    # Pad to max_len
    ids += [0] * (max_len - len(ids))
    return ids


def text_to_ids(text: str, vocab: dict, max_len: int = 128) -> list[int]:
    """Full pipeline: raw text → token IDs."""
    return tokens_to_ids(tokenize(text), vocab, max_len)


# ════════════════════════════════════════════════════════════════════════════
# VOCABULARY BUILDER
# ════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Builds and stores the word-to-index mapping from training data.
    Saved to vocab.pkl so the predictor can use it without retraining.
    """

    def __init__(self, min_freq: int = 2, max_size: int = 30000):
        self.min_freq  = min_freq    # ignore words that appear < min_freq times
        self.max_size  = max_size    # cap vocabulary size
        self.word2idx  = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word  = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.word_freq : Counter = Counter()

    def build(self, texts: list[str]):
        """Build vocab from a list of raw text strings."""
        print(f"Building vocabulary from {len(texts):,} texts...")

        # Count word frequencies
        for text in texts:
            for token in tokenize(text):
                self.word_freq[token] += 1

        # Add words that meet minimum frequency, sorted by frequency
        common_words = [
            word for word, freq in self.word_freq.most_common(self.max_size)
            if freq >= self.min_freq
        ]

        for word in common_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

        print(f"Vocabulary size: {len(self.word2idx):,} words "
              f"(min_freq={self.min_freq}, max_size={self.max_size})")

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str = VOCAB_PATH):
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx' : self.word2idx,
                'idx2word' : self.idx2word,
                'word_freq': dict(self.word_freq)
            }, f)
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: str = VOCAB_PATH) -> 'Vocabulary':
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Vocabulary not found at {path}. Run trainer.py first."
            )
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab          = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq= Counter(data.get('word_freq', {}))
        print(f"Vocabulary loaded: {len(vocab):,} words")
        return vocab


# ════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG — saved alongside vocab
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Text model
    'vocab_size'    : 30000,
    'embed_dim'     : 128,     # FastText-style embedding dimension
    'gru_hidden'    : 256,     # GRU hidden state size
    'gru_layers'    : 2,       # stacked GRU layers
    'gru_dropout'   : 0.3,
    'max_seq_len'   : 128,     # max tokens per diary entry

    # Numerical model
    'num_features'  : 10,      # number of health/usage features
    'num_hidden'    : 64,      # MLP hidden size for numerical branch

    # Fusion
    'fusion_hidden' : 128,     # size after concatenating text + numeric vectors
    'num_classes'   : 7,       # Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, PD

    # Training
    'learning_rate' : 0.001,
    'batch_size'    : 32,
    'epochs'        : 20,
    'early_stop_patience': 4,

    # Labels
    'labels'        : LABELS,
}


def save_config(config: dict = DEFAULT_CONFIG, path: str = CONFIG_PATH):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")


def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    # Quick test
    sample = "I have been feeling really anxious lately and cannot sleep well at night."
    tokens = tokenize(sample)
    print(f"Tokens: {tokens}")
    print(f"Labels: {LABEL2IDX}")
    print(f"Config keys: {list(DEFAULT_CONFIG.keys())}")
