"""Feature and augmentation helpers separated from MA5770 experiments."""

from __future__ import annotations

from typing import Iterable, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def tfidf_features(
    texts: Iterable[str],
    max_features: int = 30000,
    stop_words: str = "english",
    ngram_range=(1, 2),
    min_df: int = 2,
):
    """Build normalized TF-IDF matrix as used in the text experiments."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X = vectorizer.fit_transform(list(texts))
    return normalize(X, norm="l2", axis=1), vectorizer


def pad_or_trim(x, fixed_len: int = 8000) -> np.ndarray:
    """Pad or trim a 1D waveform to a fixed length."""
    x = np.asarray(x)
    if len(x) < fixed_len:
        return np.pad(x, (0, fixed_len - len(x)))
    return x[:fixed_len]


def extract_logmel_features(
    waveforms,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """Extract flattened log-mel features. Requires librosa installed."""
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("extract_logmel_features requires librosa. Install with: pip install librosa") from exc

    feats: List[np.ndarray] = []
    for x in waveforms:
        mel = librosa.feature.melspectrogram(
            y=np.asarray(x, dtype=np.float32),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        feats.append(logmel.flatten())
    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return normalize(np.array(feats, dtype=np.float32), norm="l2", axis=1)


def extract_image_features(imgs) -> np.ndarray:
    """Flatten image tensors and L2-normalize rows."""
    X = np.asarray(imgs, dtype=np.float32).reshape(len(imgs), -1)
    return normalize(X, norm="l2", axis=1)


def corrupt_text(text: str, drop_prob: float = 0.15, seed: int = 0) -> str:
    """Create a reproducible text near-duplicate by randomly dropping words."""
    rng = np.random.default_rng(seed)
    words = text.split()
    kept = [w for w in words if rng.random() > drop_prob]
    return " ".join(kept) if kept else text


def augment_audio(x, noise_std: float = 0.01, shift: int = 400, seed: int = 0) -> np.ndarray:
    """Create a reproducible audio near-duplicate by shifting and adding noise."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    y = np.roll(x, shift)
    y = y + rng.normal(0, noise_std, size=y.shape).astype(np.float32)
    max_abs = np.max(np.abs(y)) + 1e-12
    return y / max_abs


def augment_image(
    img,
    shift_x: int = 2,
    shift_y: int = 2,
    noise_std: float = 0.02,
    brightness: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """Create a reproducible image near-duplicate by shifting, brightening, and adding noise."""
    rng = np.random.default_rng(seed)
    y = np.asarray(img, dtype=np.float32).copy()
    y = np.roll(y, shift=shift_y, axis=0)
    y = np.roll(y, shift=shift_x, axis=1)
    y = y + brightness + rng.normal(0, noise_std, size=y.shape).astype(np.float32)
    return np.clip(y, 0.0, 1.0)
