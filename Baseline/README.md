# Baseline methods

This directory contains 12 baseline methods for spatial domain identification and multi-slice integration, along with supporting alignment algorithms and helper modules. The existing source code and directory structure have been preserved. No baseline methods were run, and no algorithm code was modified as part of preparing this document.

## Method list

| Method | Directory / main interface | Description |
| --- | --- | --- |
| CellCharter | `Cellcharter/cellcharter_DLPFC.ipynb` | Provides a DLPFC example using scVI embeddings, spatial neighborhood aggregation, and mclust clustering. Requires the external `cellcharter` package; its source is not included here. |
| DeepST | `deepst/DeepST.py` (`run`) | Combines gene expression, spatial neighborhoods, and optional histology features to learn representations for spatial domain identification and batch integration. |
| GraphST | `GraphST/GraphST.py` (`GraphST`) | Learns spatial representations with a graph encoder and contrastive learning. Also includes reference single-cell-to-spot mapping functionality. |
| MENDER | `MENDER/MENDER.py` (`MENDER`, `MENDER_single`) | Constructs multi-scale neighborhood representations from cell-type labels and spatial coordinates for spatial domain identification. |
| SEDR | `SEDR/SEDR_model.py` (`Sedr`) | Combines expression reconstruction and a variational graph autoencoder to learn representations, with optional deep embedded clustering and imputation support. |
| SpaBatch | `SpaBatch/train.py` (`train_model`) | Learns representations from PCA features and spatial graphs using a variational graph autoencoder, masked reconstruction, and cross-batch triplet training. |
| SpaCross | `SpaCross/Pipeline.py` (`SC_pipeline`, `SC_BC_pipeline`) | Provides graph-based representation learning pipelines with clustering-guided sampling, plus utilities for clustering and spot alignment. |
| SPIRAL | `spiral/main.py` (`SPIRAL_integration`); `spiral/CoordAlignment.py` (`CoordAlignment`) | Provides spatial data integration with adversarial learning and a separate coordinate alignment interface. Uses the bundled `pytorch_revgrad` helper. |
| STAGATE | `STAGATE_pyG/Train_STAGATE.py` (`train_STAGATE`) | Learns spatial embeddings using a graph attention autoencoder. Includes spatial graph construction and clustering utilities. |
| STAligner | `STAligner/train_STAligner.py` (`train_STAligner`, `train_STAligner_subgraph`) | Integrates spatial slices using graph attention embeddings and mutual-nearest-neighbor triplet training. Includes ICP alignment utilities. |
| STG3Net | `STG3Net/STNet.py` (`G3net`) | Learns multi-slice spatial representations with graph-based reconstruction, triplet learning, and an adversarial slice discriminator. Includes clustering and imputation modes. |
| STitch3D | `STitch3D/model.py` (`Model`); `STitch3D/utils.py` | Provides slice alignment, preprocessing, and graph attention modeling with reference expression signatures for integrated representations and cell-type deconvolution. |

## Supporting algorithms and modules

The `paste/` directory provides PASTE, an alignment algorithm required by other baseline methods. Its main interfaces, `pairwise_align` and `center_align`, are implemented in `paste/PASTE.py` and align spatial transcriptomics slices using fused Gromov-Wasserstein optimal transport. It requires `POT` (imported as `ot`) and is not counted as a baseline method.

The `pytorch_revgrad/` directory is a gradient-reversal helper used by SPIRAL, rather than a separate baseline method.

## Usage notes

The project-level `requirements.txt` lists shared dependencies, but it is not a verified environment specification for every baseline. Some methods require additional third-party packages, such as `cellcharter`. Dependencies were not installed or checked by executing the methods when preparing this document.

Most methods expose Python classes or functions rather than a unified command-line entry point. Inputs differ between methods: many use AnnData objects with spatial coordinates in `adata.obsm['spatial']`, while some also require PCA features, precomputed graphs, batch or slice labels, or reference expression signatures. SPIRAL's integration interface accepts feature, edge, and metadata files. Refer to each interface for its exact parameters and preprocessing requirements.

Import paths are not uniform. Modules using relative imports can be imported through `Baseline.*` from the project root, `C:\Code\iSpa3D-1`. MENDER, SpaBatch, and STitch3D also use top-level imports such as `MENDER.utils`, `SpaBatch.model`, and `STitch3D.networks`, so the `Baseline/` directory must be on Python's module search path for those imports to resolve. SPIRAL likewise requires `Baseline/` on the search path to find `pytorch_revgrad`. DeepST uses bare local imports such as `utils_func`, `model`, and `trainer`, so its own `Baseline/deepst/` directory must be available on the search path.

Several clustering utilities use `rpy2` and the R package `mclust`, which requires a separate R installation. Some files and the CellCharter notebook contain machine-specific R, data, or output paths; review these before use. The CellCharter notebook also imports `STG3Net.Metrics`, which requires `Baseline/` on the search path.
