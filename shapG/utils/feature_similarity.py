"""Feature similarity calculation with type awareness."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
from scipy.spatial.distance import hamming, jaccard
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Optional, Tuple


def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association."""
    # Create contingency table
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    # Cramér's V with bias correction
    phi2 = chi2 / n
    phi2_corrected = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)

    if min(r_corrected - 1, k_corrected - 1) == 0:
        return 0

    return np.sqrt(phi2_corrected / min(r_corrected - 1, k_corrected - 1))


def calculate_mixed_similarity_matrix(X, feature_types=None, feature_ranges=None):
    """
    Calculate similarity matrix using appropriate metrics for each feature type.

    Parameters:
    - X: Feature matrix (numpy array or DataFrame)
    - feature_types: Dict mapping feature indices to types ('numerical', 'categorical', 'binary')
    - feature_ranges: Dict with 'num', 'cat', 'bin' keys showing index ranges

    Returns:
    - similarity_matrix: Symmetric similarity matrix
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.to_numpy()
        n_features = X.shape[1]
    else:
        X_array = X
        n_features = X.shape[1]

    # Identify feature types if not provided
    if feature_types is None:
        feature_types = {}

        if feature_ranges is not None:
            # Use feature ranges to identify types
            for i in range(n_features):
                if "num" in feature_ranges:
                    start, end = feature_ranges["num"]
                    if start <= i < end:
                        feature_types[i] = "numerical"
                        continue
                if "cat" in feature_ranges:
                    start, end = feature_ranges["cat"]
                    if start <= i < end:
                        feature_types[i] = "categorical"
                        continue
                if "bin" in feature_ranges:
                    start, end = feature_ranges["bin"]
                    if start <= i < end:
                        feature_types[i] = "binary"
                        continue
                # Default to numerical if not found
                feature_types[i] = "numerical"
        else:
            # Auto-detect based on unique values
            for i in range(n_features):
                unique_vals = np.unique(X_array[:, i])
                if len(unique_vals) == 2:
                    feature_types[i] = "binary"
                elif len(unique_vals) < 10:  # Heuristic for categorical
                    feature_types[i] = "categorical"
                else:
                    feature_types[i] = "numerical"

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_features, n_features))
    np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity is 1

    # Calculate pairwise similarities
    for i in range(n_features):
        for j in range(i + 1, n_features):
            type_i = feature_types.get(i, "numerical")
            type_j = feature_types.get(j, "numerical")

            x_i = X_array[:, i]
            x_j = X_array[:, j]

            # Choose similarity metric based on feature types
            if type_i == "numerical" and type_j == "numerical":
                # Use Pearson correlation for numerical features
                try:
                    corr, _ = pearsonr(x_i, x_j)
                    sim = abs(corr)  # Use absolute correlation
                except:
                    sim = 0.0

            elif type_i == "binary" and type_j == "binary":
                # Use Jaccard similarity for binary features
                try:
                    # Convert to binary if needed
                    x_i_bin = (x_i > 0).astype(int)
                    x_j_bin = (x_j > 0).astype(int)
                    # Jaccard = intersection / union
                    intersection = np.sum((x_i_bin == 1) & (x_j_bin == 1))
                    union = np.sum((x_i_bin == 1) | (x_j_bin == 1))
                    sim = intersection / union if union > 0 else 0.0
                except:
                    sim = 0.0

            elif type_i == "categorical" and type_j == "categorical":
                # Use normalized mutual information or Cramér's V
                try:
                    # Cramér's V for categorical association
                    sim = cramers_v(x_i, x_j)
                except:
                    # Fallback to normalized mutual information
                    try:
                        mi = mutual_info_score(x_i, x_j)
                        # Normalize by geometric mean of entropies
                        h_i = -np.sum(
                            [
                                p * np.log(p + 1e-10)
                                for p in np.bincount(x_i.astype(int)) / len(x_i)
                                if p > 0
                            ]
                        )
                        h_j = -np.sum(
                            [
                                p * np.log(p + 1e-10)
                                for p in np.bincount(x_j.astype(int)) / len(x_j)
                                if p > 0
                            ]
                        )
                        sim = mi / np.sqrt(h_i * h_j) if h_i > 0 and h_j > 0 else 0.0
                    except:
                        sim = 0.0

            else:
                # Mixed types - use normalized mutual information
                try:
                    # Discretize numerical features if needed
                    if type_i == "numerical":
                        x_i_discrete = pd.qcut(
                            x_i, q=5, labels=False, duplicates="drop"
                        )
                    else:
                        x_i_discrete = x_i

                    if type_j == "numerical":
                        x_j_discrete = pd.qcut(
                            x_j, q=5, labels=False, duplicates="drop"
                        )
                    else:
                        x_j_discrete = x_j

                    mi = mutual_info_score(x_i_discrete, x_j_discrete)
                    # Simple normalization
                    max_mi = min(
                        np.log(len(np.unique(x_i_discrete))),
                        np.log(len(np.unique(x_j_discrete))),
                    )
                    sim = mi / max_mi if max_mi > 0 else 0.0
                except:
                    sim = 0.0

            # Ensure similarity is in [0, 1]
            sim = np.clip(sim, 0, 1)

            # Fill symmetric matrix
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix, feature_types


def get_feature_ranking_mixed(X, y, feature_types=None, feature_ranges=None):
    """
    Rank features by their association with target using appropriate metrics.

    Parameters:
    - X: Feature matrix
    - y: Target variable
    - feature_types: Dict mapping feature indices to types
    - feature_ranges: Dict with feature range information

    Returns:
    - Ordered list of feature indices (least important first for edge removal)
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.to_numpy()
        n_features = X.shape[1]
    else:
        X_array = X
        n_features = X.shape[1]

    # Identify feature types if not provided
    if feature_types is None:
        feature_types = {}
        for i in range(n_features):
            unique_vals = np.unique(X_array[:, i])
            if len(unique_vals) == 2:
                feature_types[i] = "binary"
            elif len(unique_vals) < 10:
                feature_types[i] = "categorical"
            else:
                feature_types[i] = "numerical"

    # Calculate importance scores
    importance_scores = {}

    # Determine target type
    unique_y = np.unique(y)
    if len(unique_y) == 2:
        y_type = "binary"
    elif len(unique_y) < 10:
        y_type = "categorical"
    else:
        y_type = "numerical"

    for i in range(n_features):
        feature_type = feature_types.get(i, "numerical")
        x_i = X_array[:, i]

        try:
            if feature_type == "numerical" and y_type in ["numerical", "binary"]:
                # Use correlation
                corr, p_val = pearsonr(x_i, y)
                importance_scores[i] = abs(corr)

            elif feature_type == "binary" and y_type == "binary":
                # Use phi coefficient (Pearson for binary)
                corr, _ = pearsonr(x_i, y)
                importance_scores[i] = abs(corr)

            elif feature_type == "categorical" and y_type in ["categorical", "binary"]:
                # Use Cramér's V
                importance_scores[i] = cramers_v(x_i, y)

            else:
                # Mixed or other cases - use mutual information
                if feature_type == "numerical":
                    x_discrete = pd.qcut(x_i, q=5, labels=False, duplicates="drop")
                else:
                    x_discrete = x_i

                if y_type == "numerical":
                    y_discrete = pd.qcut(y, q=5, labels=False, duplicates="drop")
                else:
                    y_discrete = y

                mi = mutual_info_score(x_discrete, y_discrete)
                max_mi = min(
                    np.log(len(np.unique(x_discrete))),
                    np.log(len(np.unique(y_discrete))),
                )
                importance_scores[i] = mi / max_mi if max_mi > 0 else 0.0
        except:
            importance_scores[i] = 0.0

    # Sort by importance (ascending - least important first for removal)
    sorted_indices = sorted(
        importance_scores.keys(), key=lambda k: importance_scores[k]
    )

    return sorted_indices
