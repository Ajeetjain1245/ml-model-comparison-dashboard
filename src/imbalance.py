"""
Robust, Zero-Dependency Imbalance & Resampling Engine (SMOTE, Over-sampling, Under-sampling).
"""
from typing import Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Minority Over-sampling Technique (SMOTE).
    Generates synthetic instances along the feature space line segments
    joining minority class neighbors. Pure NumPy/Scikit-learn implementation.
    """
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    X_resampled = [X]
    y_resampled = [y]

    for cls, count in zip(unique_classes, counts):
        n_samples_to_generate = max_count - count
        if n_samples_to_generate <= 0:
            continue

        X_cls = X[y == cls]
        n_cls_samples = len(X_cls)

        if n_cls_samples < 2:
            # If only 1 sample, fallback to simple duplication
            idx = rng.choice(n_cls_samples, size=n_samples_to_generate, replace=True)
            X_resampled.append(X_cls[idx])
            y_resampled.append(np.full(n_samples_to_generate, cls))
            continue

        # Adjust k if class has fewer samples than k_neighbors + 1
        k = min(k_neighbors, n_cls_samples - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        nn.fit(X_cls)
        # Find neighbors (excluding self at index 0)
        neighbors_indices = nn.kneighbors(X_cls, return_distance=False)[:, 1:]

        # Randomly choose base samples and corresponding neighbors
        base_indices = rng.choice(n_cls_samples, size=n_samples_to_generate, replace=True)
        neighbor_pick = rng.integers(0, k, size=n_samples_to_generate)
        chosen_neighbors = neighbors_indices[base_indices, neighbor_pick]

        # Generate synthetic samples: base + gap * (neighbor - base)
        base_samples = X_cls[base_indices]
        target_neighbors = X_cls[chosen_neighbors]
        gaps = rng.uniform(0, 1, size=(n_samples_to_generate, 1))

        synthetic_samples = base_samples + gaps * (target_neighbors - base_samples)

        X_resampled.append(synthetic_samples)
        y_resampled.append(np.full(n_samples_to_generate, cls))

    X_out = np.vstack(X_resampled)
    y_out = np.concatenate(y_resampled)

    # Shuffle output
    shuffle_idx = rng.permutation(len(y_out))
    return X_out[shuffle_idx], y_out[shuffle_idx]


def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly duplicates minority class samples until all classes match the majority count."""
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    X_resampled = [X]
    y_resampled = [y]

    for cls, count in zip(unique_classes, counts):
        n_needed = max_count - count
        if n_needed > 0:
            X_cls = X[y == cls]
            idx = rng.choice(len(X_cls), size=n_needed, replace=True)
            X_resampled.append(X_cls[idx])
            y_resampled.append(np.full(n_needed, cls))

    X_out = np.vstack(X_resampled)
    y_out = np.concatenate(y_resampled)
    shuffle_idx = rng.permutation(len(y_out))
    return X_out[shuffle_idx], y_out[shuffle_idx]


def random_undersample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly downsamples majority class samples until all classes match the minority count."""
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)

    X_resampled = []
    y_resampled = []

    for cls in unique_classes:
        X_cls = X[y == cls]
        idx = rng.choice(len(X_cls), size=min_count, replace=False)
        X_resampled.append(X_cls[idx])
        y_resampled.append(np.full(min_count, cls))

    X_out = np.vstack(X_resampled)
    y_out = np.concatenate(y_resampled)
    shuffle_idx = rng.permutation(len(y_out))
    return X_out[shuffle_idx], y_out[shuffle_idx]


def apply_imbalance_handling(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "None",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the chosen imbalance handling technique to feature matrix X and target y.
    Supported strategies:
      - 'None'
      - 'SMOTE (Synthetic Minority Over-sampling)'
      - 'Random Over-sampling'
      - 'Random Under-sampling'
      - 'Balanced Class Weights' (handled algorithmically in models)
    """
    if strategy == "SMOTE (Synthetic Minority Over-sampling)":
        return smote_resample(X, y, random_state=random_state)
    elif strategy == "Random Over-sampling":
        return random_oversample(X, y, random_state=random_state)
    elif strategy == "Random Under-sampling":
        return random_undersample(X, y, random_state=random_state)
    else:
        return X, y
