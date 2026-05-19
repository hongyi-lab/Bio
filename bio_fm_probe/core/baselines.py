"""Modality-specific baseline feature builders for the (model, dataset, task) audit.

Every builder takes the dataset's raw inputs and a feature dim K, returns:
  (X: (n_samples, K) float32, info: dict)

These are used to fit the "raw input → labels" baseline that the model's
representations are graded against (and against which the SAE−PCA ablation
gap is measured).

Three builders for the three modalities the toolkit supports today:
  - log1p_pca:   scRNA — total-count normalize + log1p + PCA
  - kmer_pca:    DNA   — HashingVectorizer over k-mers + PCA
  - onehot_aa_pca: protein — one-hot AA, position-mean, then PCA
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def log1p_pca(adata, n_components: int = 50) -> Tuple[np.ndarray, Dict]:
    """scRNA baseline: total-count normalize to 1e4 + log1p + PCA.

    Operates on adata.X (raw counts or already-log-normed; auto-detects negatives).
    Returns (X (n_cells, n_components), info).
    """
    import scanpy as sc

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X).astype(np.float32)

    if (X < 0).any():
        # already log/scaled; use as-is
        Xn = X
    else:
        # raw counts → CPM-like + log1p
        sf = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        Xn = np.log1p(X / sf * 1e4)

    from sklearn.decomposition import PCA
    n_c = min(n_components, Xn.shape[1], Xn.shape[0] - 1)
    pca = PCA(n_components=n_c, random_state=0)
    X_pca = pca.fit_transform(Xn).astype(np.float32)
    info = {
        "kind": "log1p_pca",
        "n_components": int(pca.n_components_),
        "evr_sum": float(pca.explained_variance_ratio_.sum()),
        "n_genes": int(Xn.shape[1]),
    }
    return X_pca, info


def _seq_to_kmers(seq: str, k: int) -> List[str]:
    seq = seq.upper()
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def kmer_pca(
    seqs: List[str], k_mer: int = 6, n_components: int = 50,
    n_features_hash: int = 4096,
) -> Tuple[np.ndarray, Dict]:
    """DNA baseline: HashingVectorizer over k-mers + PCA.

    Avoids needing an explicit vocabulary (which would be 4^6 = 4096 for k=6,
    larger for bigger k). HashingVectorizer collides hashes into n_features
    buckets but for PCA-baseline purposes this is fine.
    """
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import HashingVectorizer

    docs = [" ".join(_seq_to_kmers(s, k_mer)) for s in seqs]
    vec = HashingVectorizer(
        n_features=n_features_hash,
        alternate_sign=False, norm=None, dtype=np.float32,
    )
    Xc = vec.fit_transform(docs)
    Xc = Xc.astype(np.float32)
    n_c = min(n_components, Xc.shape[1], Xc.shape[0] - 1)
    pca = PCA(n_components=n_c, random_state=0)
    # PCA on sparse needs densification; do batches if too big. For a baseline
    # of ~10k samples × 4096 features this fits in ~150 MB — fine.
    X_dense = Xc.toarray()
    X_pca = pca.fit_transform(X_dense).astype(np.float32)
    info = {
        "kind": "kmer_pca",
        "k_mer": k_mer,
        "n_features_hash": n_features_hash,
        "n_components": int(pca.n_components_),
        "evr_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return X_pca, info


# 20 standard AAs + X for unknown
_AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
_AA_TO_IDX = {a: i for i, a in enumerate(_AA_LIST)}


def _aa_compose(seq: str) -> np.ndarray:
    """AA composition vector (length 20) — position-mean of one-hot."""
    v = np.zeros(len(_AA_LIST), dtype=np.float32)
    n = 0
    for aa in seq.upper():
        idx = _AA_TO_IDX.get(aa)
        if idx is not None:
            v[idx] += 1
            n += 1
    if n > 0:
        v /= n
    return v


def onehot_aa_pca(
    seqs: List[str], n_components: int = 20,
) -> Tuple[np.ndarray, Dict]:
    """Protein baseline: AA composition (position-mean of one-hot) + PCA.

    AA composition is 20-D so PCA is essentially identity for n_components=20;
    that's fine — composition itself is the baseline. For bigger n_components,
    we additionally include di-peptide composition (20*20 = 400 dim) and PCA.
    """
    from sklearn.decomposition import PCA

    comp = np.stack([_aa_compose(s) for s in seqs], axis=0)
    if n_components <= len(_AA_LIST):
        # PCA on 20-D composition is essentially a unitary transform
        n_c = min(n_components, comp.shape[1], comp.shape[0] - 1)
        pca = PCA(n_components=n_c, random_state=0)
        X = pca.fit_transform(comp).astype(np.float32)
        info = {
            "kind": "onehot_aa_pca",
            "feature_set": "aa_composition",
            "n_components": int(pca.n_components_),
            "evr_sum": float(pca.explained_variance_ratio_.sum()),
        }
        return X, info

    # Need >20 dims: extend with di-peptide composition
    n_aa = len(_AA_LIST)
    di = np.zeros((len(seqs), n_aa * n_aa), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = s.upper()
        n_pairs = 0
        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            ai, bi = _AA_TO_IDX.get(a), _AA_TO_IDX.get(b)
            if ai is None or bi is None:
                continue
            di[r, ai * n_aa + bi] += 1
            n_pairs += 1
        if n_pairs > 0:
            di[r] /= n_pairs
    feats = np.concatenate([comp, di], axis=1)
    n_c = min(n_components, feats.shape[1], feats.shape[0] - 1)
    pca = PCA(n_components=n_c, random_state=0)
    X = pca.fit_transform(feats).astype(np.float32)
    info = {
        "kind": "onehot_aa_pca",
        "feature_set": "aa_composition + dipeptide_composition",
        "n_components": int(pca.n_components_),
        "evr_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return X, info


def build_baseline(sample, n_components: int = 50) -> Tuple[np.ndarray, Dict]:
    """Dispatch on sample.baseline_kind."""
    if sample.baseline_kind == "log1p_pca":
        return log1p_pca(sample.inputs, n_components=n_components)
    if sample.baseline_kind == "kmer_pca":
        return kmer_pca(sample.inputs, n_components=n_components)
    if sample.baseline_kind == "onehot_aa_pca":
        return onehot_aa_pca(sample.inputs, n_components=n_components)
    raise ValueError(f"unknown baseline_kind: {sample.baseline_kind!r}")
