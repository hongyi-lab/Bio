"""Top-9 PCA on layer-12 cell-level activations, interpreted biologically.

Per-cell activation = mean of layer-12 token activations over the cell's
non-pad, non-CLS gene tokens (same pooling phase2 found best).

Then PCA-9 on (n_cells, d_model), and for each PC compute:
  - Pearson |r| against every continuous covariate
  - eta^2 against every categorical covariate (one-way ANOVA effect size)

Covariates:
  cell type (louvain)              categorical
  n_counts                         continuous  (obs.n_counts)
  n_genes                          continuous  (obs.n_genes)
  percent_mito                     continuous  (obs.percent_mito)
  S_score, G2M_score               continuous  (sc.tl.score_genes_cell_cycle, Tirosh 2016)
  housekeeping_score               continuous  (HK_genes.txt)
  HALLMARK / KEGG / REACTOME / GO  continuous  (one score per .grp in gene_sets/)

Outputs in results/phase3/pca_biology/:
  pca_scores.npz                  PC scores (n_cells, 9) + covariate matrix
  pc_var_explained.png            scree plot
  pc_covariate_heatmap.png        9 PCs x N covariates, |corr| / eta
  pc1_pc2_grid.png                PC1 vs PC2 scatter, one panel per covariate
  top_pc_pairs_celltype.png       (PC1,2) (PC3,4) (PC5,6) (PC7,8) colored by cell type
  pca_biology.json                full table: per-PC variance + top covariates
  SUMMARY.md                      digest

Usage:
  python src/phase3_pca_biology.py                       # default: layer_12
  python src/phase3_pca_biology.py --layer layer_03      # any cached layer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = ROOT / "results" / "phase3"
GENE_SETS_DIR = ROOT / "gene_sets"
OUT_DIR = PHASE3_DIR / "pca_biology"

# Tirosh 2016 cell-cycle gene lists (S and G2M); scanpy ships them in the
# tutorial but not the package. Hard-coded for self-containment.
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2",
    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "BRIP1", "E2F8",
]
G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]


# ---------------------------------------------------------------------------
# Aggregation: tokens -> per-cell mean
# ---------------------------------------------------------------------------
def cells_from_tokens(token_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-pool gene-token activations to per-cell vectors.

    Returns:
      cell_acts (n_cells, d_model) float32
      tokens_per_cell (n_cells,) int64
    """
    print(f"[pca] loading {token_npz} ({token_npz.stat().st_size / 1e9:.2f} GB)", flush=True)
    npz = np.load(token_npz, mmap_mode="r")
    acts = np.asarray(npz["acts"])             # (N, d)
    cell_idx = np.asarray(npz["cell_idx"]).astype(np.int64)  # (N,)
    n_cells = int(cell_idx.max() + 1)
    d = acts.shape[1]
    print(f"[pca]   {acts.shape[0]:,} tokens, {n_cells} cells, d={d}", flush=True)

    sums = np.zeros((n_cells, d), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    # Stream in chunks to control RAM peak
    chunk = 500_000
    for s in range(0, acts.shape[0], chunk):
        e = min(s + chunk, acts.shape[0])
        np.add.at(sums, cell_idx[s:e], acts[s:e].astype(np.float64))
        np.add.at(counts, cell_idx[s:e], 1)
    counts_safe = np.clip(counts, 1, None)
    cell_acts = (sums / counts_safe[:, None]).astype(np.float32)
    return cell_acts, counts


# ---------------------------------------------------------------------------
# Gene-set loading
# ---------------------------------------------------------------------------
def load_grp(path: Path) -> List[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        out.append(line.split("\t")[0])  # tolerate optional 2-col format
    return out


def load_hk(path: Path) -> List[str]:
    # HK_genes.txt format is typically gene symbol per line (possibly tab-sep).
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(line.split("\t")[0])
    return out


# ---------------------------------------------------------------------------
# Covariate construction
# ---------------------------------------------------------------------------
def build_covariates(
    adata,
    gene_sets_dir: Path,
) -> pd.DataFrame:
    """Produce a wide DataFrame of covariates indexed like adata.obs.

    For gene scoring we need log1p-normalized non-negative expression. pbmc3k
    stores that in `.raw`; we materialize a tiny working AnnData for scoring.
    """
    print("[pca] building covariates ...", flush=True)
    # Prefer .raw (log1p) if available; otherwise assume .X is non-negative.
    if adata.raw is not None and adata.X.min() < 0:
        import anndata as ad
        score_ad = ad.AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
        )
    else:
        score_ad = adata.copy()
    score_ad.var_names_make_unique()
    available = set(score_ad.var_names)
    print(f"[pca]   {len(available)} genes available for scoring")

    cov = pd.DataFrame(index=score_ad.obs.index)

    # Continuous QC covariates
    for col in ("n_counts", "n_genes", "percent_mito"):
        if col in score_ad.obs.columns:
            cov[col] = score_ad.obs[col].astype(float).values

    # Cell-type (categorical; we keep it separate, not in the corr matrix)
    if "louvain" in score_ad.obs.columns:
        cov["__celltype__"] = score_ad.obs["louvain"].astype(str).values

    # Cell-cycle (Tirosh 2016 lists)
    s_present = [g for g in S_GENES if g in available]
    g2m_present = [g for g in G2M_GENES if g in available]
    print(f"[pca]   cell-cycle: S {len(s_present)}/{len(S_GENES)}, "
          f"G2M {len(g2m_present)}/{len(G2M_GENES)}")
    if len(s_present) >= 5 and len(g2m_present) >= 5:
        sc.tl.score_genes_cell_cycle(score_ad, s_genes=s_present, g2m_genes=g2m_present,
                                     random_state=0)
        cov["S_score"] = score_ad.obs["S_score"].astype(float).values
        cov["G2M_score"] = score_ad.obs["G2M_score"].astype(float).values
        cov["cell_cycle_S_minus_G2M"] = cov["S_score"] - cov["G2M_score"]

    # Housekeeping
    hk_path = gene_sets_dir / "HK_genes.txt"
    if hk_path.exists():
        hk_genes = [g for g in load_hk(hk_path) if g in available]
        if len(hk_genes) >= 5:
            sc.tl.score_genes(score_ad, gene_list=hk_genes, score_name="HK_score",
                              random_state=0)
            cov["HK_score"] = score_ad.obs["HK_score"].astype(float).values
            print(f"[pca]   HK: {len(hk_genes)} genes used")

    # All .grp gene sets
    for grp in sorted(gene_sets_dir.glob("*.grp")):
        gene_list = [g for g in load_grp(grp) if g in available]
        name = grp.stem
        if len(gene_list) < 5:
            print(f"[pca]   {name}: only {len(gene_list)} genes overlap — skipping")
            continue
        score_name = f"{name}_score"
        sc.tl.score_genes(score_ad, gene_list=gene_list, score_name=score_name,
                          random_state=0)
        cov[score_name] = score_ad.obs[score_name].astype(float).values
        print(f"[pca]   {name}: {len(gene_list)} genes used")

    return cov


# ---------------------------------------------------------------------------
# Correlations / effect sizes
# ---------------------------------------------------------------------------
def pearson_r_matrix(scores: np.ndarray, cov_df: pd.DataFrame) -> pd.DataFrame:
    """|Pearson r| between every PC column and every continuous covariate."""
    cont = cov_df.select_dtypes(include=[np.number])
    mat = np.zeros((scores.shape[1], cont.shape[1]))
    s_centered = scores - scores.mean(axis=0)
    s_std = s_centered.std(axis=0, ddof=0)
    for j, col in enumerate(cont.columns):
        v = cont[col].values.astype(float)
        v_centered = v - v.mean()
        v_std = v.std(ddof=0)
        if v_std < 1e-12:
            continue
        mat[:, j] = (s_centered.T @ v_centered) / (
            len(v) * s_std * v_std + 1e-12
        )
    return pd.DataFrame(mat, columns=list(cont.columns),
                        index=[f"PC{i + 1}" for i in range(scores.shape[1])])


def eta_squared(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """eta^2 per PC (one-way ANOVA effect size: between-class SS / total SS)."""
    classes = np.unique(labels)
    out = np.zeros(scores.shape[1])
    for j in range(scores.shape[1]):
        x = scores[:, j]
        grand = x.mean()
        ss_total = ((x - grand) ** 2).sum()
        ss_between = 0.0
        for c in classes:
            xs = x[labels == c]
            if len(xs) == 0:
                continue
            ss_between += len(xs) * (xs.mean() - grand) ** 2
        out[j] = ss_between / max(ss_total, 1e-12)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_var_explained(pca: PCA, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    ax.bar(xs, pca.explained_variance_ratio_, color="C0", alpha=0.7,
           label="per-PC")
    ax2 = ax.twinx()
    ax2.plot(xs, np.cumsum(pca.explained_variance_ratio_), color="C3",
             marker="o", label="cumulative")
    ax.set_xticks(xs)
    ax.set_xlabel("PC")
    ax.set_ylabel("variance explained")
    ax2.set_ylabel("cumulative")
    ax.set_title(f"scree — top {len(xs)} PCs of layer-12 cell activations")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_heatmap(r_df: pd.DataFrame, eta_celltype: np.ndarray | None,
                 out_png: Path) -> None:
    # combine continuous |r| and (optional) categorical sqrt(eta^2)
    cols = list(r_df.columns)
    M = np.abs(r_df.values)
    if eta_celltype is not None:
        cols = ["celltype (sqrt(eta^2))"] + cols
        M = np.hstack([np.sqrt(eta_celltype)[:, None], M])
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(cols)), 4.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=min(1.0, M.max() * 1.05))
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(r_df.index)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.6 else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax, label="|corr| (or sqrt(eta^2) for cell type)")
    ax.set_title("PC vs covariate strength")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_pc12_grid(scores: np.ndarray, cov_df: pd.DataFrame,
                   celltype: np.ndarray | None, out_png: Path) -> None:
    cont_cols = list(cov_df.select_dtypes(include=[np.number]).columns)
    panels = ["__celltype__"] + cont_cols if celltype is not None else cont_cols
    n = len(panels)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows),
                            squeeze=False)
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    pc1, pc2 = scores[:, 0], scores[:, 1]
    for ax, name in zip(axs.flat, panels):
        if name == "__celltype__":
            classes = np.unique(celltype)
            for i, c in enumerate(classes):
                m = celltype == c
                ax.scatter(pc1[m], pc2[m], s=6, alpha=0.6, label=str(c),
                           color=plt.cm.tab10(i % 10))
            ax.legend(fontsize=6, markerscale=2, loc="best")
            ax.set_title("cell type (louvain)")
        else:
            vals = cov_df[name].values.astype(float)
            sc_ = ax.scatter(pc1, pc2, c=vals, s=6, alpha=0.7, cmap="viridis")
            plt.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(name, fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("layer-12 PCA — PC1 vs PC2, colored by covariates", y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_top_pc_pairs_celltype(scores: np.ndarray, celltype: np.ndarray | None,
                               out_png: Path) -> None:
    if celltype is None:
        return
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    classes = np.unique(celltype)
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (a, b) in zip(axs.flat, pairs):
        for i, c in enumerate(classes):
            m = celltype == c
            ax.scatter(scores[m, a], scores[m, b], s=8, alpha=0.6, label=str(c),
                       color=plt.cm.tab10(i % 10))
        ax.set_xlabel(f"PC{a + 1}"); ax.set_ylabel(f"PC{b + 1}")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, markerscale=2, loc="best")
    fig.suptitle("layer-12 PCA — PC pairs colored by cell type")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="layer_12")
    p.add_argument("--n_components", type=int, default=9)
    p.add_argument("--data", default="data/pbmc3k.h5ad")
    p.add_argument("--gene_sets_dir", default=str(GENE_SETS_DIR))
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token_npz = PHASE3_DIR / args.layer / "token_activations.npz"
    if not token_npz.exists():
        raise SystemExit(f"missing {token_npz}; run phase3_sae.py first")
    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    gene_sets_dir = Path(args.gene_sets_dir) if Path(args.gene_sets_dir).is_absolute() else ROOT / args.gene_sets_dir

    t0 = time.time()

    # 1) Cell-level activations via mean-pool
    cell_acts, tokens_per_cell = cells_from_tokens(token_npz)
    print(f"[pca] cell-level activations: {cell_acts.shape}", flush=True)

    # 2) PCA
    pca = PCA(n_components=args.n_components, random_state=0)
    scores = pca.fit_transform(cell_acts)  # (n_cells, 9)
    var_ratio = pca.explained_variance_ratio_
    print(f"[pca] var explained per PC: {[f'{v:.4f}' for v in var_ratio]}", flush=True)

    # 3) Load adata + build covariates
    adata = sc.read_h5ad(data_path)
    if adata.n_obs != cell_acts.shape[0]:
        raise SystemExit(f"cell count mismatch: adata={adata.n_obs} vs acts={cell_acts.shape[0]}")
    cov_df = build_covariates(adata, gene_sets_dir)

    # 4) Correlations
    celltype = cov_df.pop("__celltype__").values if "__celltype__" in cov_df.columns else None
    r_df = pearson_r_matrix(scores, cov_df)
    eta_ct = eta_squared(scores, celltype) if celltype is not None else None

    print("\n=== top covariate per PC ===")
    cont_cols = list(r_df.columns)
    for i in range(args.n_components):
        row = r_df.iloc[i].abs()
        top3 = row.sort_values(ascending=False).head(3)
        ct = (f"  celltype eta^2 = {eta_ct[i]:.3f}" if eta_ct is not None else "")
        print(f"  PC{i + 1} (var={var_ratio[i]:.4f}){ct}: "
              + ", ".join(f"{name} |r|={val:.3f}" for name, val in top3.items()))

    # 5) Plots
    plot_var_explained(pca, OUT_DIR / "pc_var_explained.png")
    plot_heatmap(r_df, eta_ct, OUT_DIR / "pc_covariate_heatmap.png")
    plot_pc12_grid(scores, cov_df, celltype, OUT_DIR / "pc1_pc2_grid.png")
    plot_top_pc_pairs_celltype(scores, celltype, OUT_DIR / "top_pc_pairs_celltype.png")

    # 6) Save scores + json
    np.savez(
        OUT_DIR / "pca_scores.npz",
        scores=scores,
        var_explained=var_ratio,
        components=pca.components_,
        tokens_per_cell=tokens_per_cell,
        covariate_names=np.array(cont_cols),
        covariate_values=cov_df[cont_cols].values.astype(np.float32),
        celltype=np.array(celltype) if celltype is not None else np.array([]),
    )

    digest = {
        "layer": args.layer,
        "n_cells": int(cell_acts.shape[0]),
        "d_model": int(cell_acts.shape[1]),
        "tokens_per_cell_mean": float(tokens_per_cell.mean()),
        "tokens_per_cell_min": int(tokens_per_cell.min()),
        "tokens_per_cell_max": int(tokens_per_cell.max()),
        "var_explained_per_PC": var_ratio.tolist(),
        "var_explained_cumulative": np.cumsum(var_ratio).tolist(),
        "celltype_eta_squared_per_PC": (eta_ct.tolist() if eta_ct is not None else None),
        "abs_pearson_r_PC_vs_covariate": r_df.abs().to_dict(),
        "covariate_names": cont_cols,
        "wall_time_s": time.time() - t0,
    }
    (OUT_DIR / "pca_biology.json").write_text(json.dumps(digest, indent=2))

    # 7) SUMMARY.md
    md = []
    md.append(f"# Phase 3 — layer `{args.layer}` PCA biology\n")
    md.append(f"- n_cells = {cell_acts.shape[0]}, d_model = {cell_acts.shape[1]}, n_PCs = {args.n_components}")
    md.append(f"- tokens per cell: mean = {tokens_per_cell.mean():.0f}, "
              f"min = {tokens_per_cell.min()}, max = {tokens_per_cell.max()}\n")
    md.append("## Variance explained by top PCs\n")
    md.append("| PC | var_ratio | cumulative |")
    md.append("|---|---|---|")
    cum = 0.0
    for i, v in enumerate(var_ratio):
        cum += v
        md.append(f"| PC{i + 1} | {v:.4f} | {cum:.4f} |")
    md.append("\n## Top 3 covariates per PC (|Pearson r|, plus cell-type eta^2)\n")
    md.append("| PC | celltype eta^2 | top1 | top2 | top3 |")
    md.append("|---|---|---|---|---|")
    for i in range(args.n_components):
        row = r_df.iloc[i].abs().sort_values(ascending=False).head(3)
        ct = (f"{eta_ct[i]:.3f}" if eta_ct is not None else "n/a")
        entries = [f"{n} ({v:.3f})" for n, v in row.items()]
        while len(entries) < 3:
            entries.append("")
        md.append(f"| PC{i + 1} | {ct} | {entries[0]} | {entries[1]} | {entries[2]} |")
    md.append("\n## Reading guide\n")
    md.append("- `celltype eta^2`: fraction of PC variance explained by louvain cluster identity.")
    md.append("  High = PC encodes cell-type structure (expected for PC1/2 of any single-cell embedding).")
    md.append("- `|Pearson r|`: linear association of a PC with a continuous covariate score.")
    md.append("- A PC with high pathway |r| but low celltype eta^2 means that PC tracks a *biological process*")
    md.append("  (e.g., ribosome activity, cell cycle) independently of cell identity — that's the kind of")
    md.append("  axis we want to see if scGPT is computing pathway-level concepts.")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))
    print(f"\n[pca] wrote {OUT_DIR}")
    print(f"[pca] wall time {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
