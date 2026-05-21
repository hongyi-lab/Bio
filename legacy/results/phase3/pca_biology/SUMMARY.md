# Phase 3 — layer `layer_12` PCA biology

- n_cells = 2638, d_model = 512, n_PCs = 9
- tokens per cell: mean = 779, min = 201, max = 1199

## Variance explained by top PCs

| PC | var_ratio | cumulative |
|---|---|---|
| PC1 | 0.3957 | 0.3957 |
| PC2 | 0.2350 | 0.6307 |
| PC3 | 0.0711 | 0.7018 |
| PC4 | 0.0556 | 0.7574 |
| PC5 | 0.0397 | 0.7971 |
| PC6 | 0.0338 | 0.8308 |
| PC7 | 0.0208 | 0.8517 |
| PC8 | 0.0169 | 0.8685 |
| PC9 | 0.0146 | 0.8832 |

## Top 3 covariates per PC (|Pearson r|, plus cell-type eta^2)

| PC | celltype eta^2 | top1 | top2 | top3 |
|---|---|---|---|---|
| PC1 | 0.869 | KEGG_RIBOSOME_score (0.377) | HALLMARK_MTORC1_SIGNALING_score (0.288) | n_genes (0.236) |
| PC2 | 0.142 | n_genes (0.874) | n_counts (0.788) | HALLMARK_MYC_TARGETS_V1_score (0.680) |
| PC3 | 0.558 | KEGG_PROTEASOME_score (0.140) | percent_mito (0.129) | HALLMARK_MTORC1_SIGNALING_score (0.110) |
| PC4 | 0.082 | KEGG_RIBOSOME_score (0.054) | n_genes (0.044) | n_counts (0.023) |
| PC5 | 0.215 | KEGG_RIBOSOME_score (0.323) | HALLMARK_MYC_TARGETS_V1_score (0.218) | HALLMARK_UNFOLDED_PROTEIN_RESPONSE_score (0.166) |
| PC6 | 0.191 | KEGG_RIBOSOME_score (0.327) | HALLMARK_MYC_TARGETS_V1_score (0.273) | n_counts (0.169) |
| PC7 | 0.041 | percent_mito (0.116) | KEGG_RIBOSOME_score (0.084) | n_counts (0.076) |
| PC8 | 0.109 | KEGG_PROTEASOME_score (0.175) | HALLMARK_MYC_TARGETS_V1_score (0.158) | percent_mito (0.105) |
| PC9 | 0.243 | KEGG_RIBOSOME_score (0.221) | HALLMARK_MYC_TARGETS_V1_score (0.155) | HALLMARK_UNFOLDED_PROTEIN_RESPONSE_score (0.101) |

## Reading guide

- `celltype eta^2`: fraction of PC variance explained by louvain cluster identity.
  High = PC encodes cell-type structure (expected for PC1/2 of any single-cell embedding).
- `|Pearson r|`: linear association of a PC with a continuous covariate score.
- A PC with high pathway |r| but low celltype eta^2 means that PC tracks a *biological process*
  (e.g., ribosome activity, cell cycle) independently of cell identity — that's the kind of
  axis we want to see if scGPT is computing pathway-level concepts.