# Baseline methods

This directory provides 12 baseline methods, each with a Jupyter notebook demonstrating analysis of the DLPFC dataset. The descriptions below follow the workflows in these notebooks, including data preparation, model training, clustering, and evaluation. PASTE and `pytorch_revgrad` are supporting components and are not counted as baseline methods.

## DLPFC examples

| Method | Example notebook | Workflow in the example |
| --- | --- | --- |
| CellCharter | [cellcharter_DLPFC.ipynb](Cellcharter/cellcharter_DLPFC.ipynb) | Loads and combines Visium slices, trains scVI with `sample` as the batch key, constructs spatial neighbors within each slice, and aggregates three neighborhood layers with CellCharter. Clusters `X_cellcharter` with mclust and evaluates the spatial domains. Includes separate blocks for all three slice groups. |
| DeepST | [DeepST_integration.ipynb](deepst/DeepST_integration.ipynb) | Loads Visium slices and histology images, extracts image crops, augments expression, constructs spatial graphs, and trains DeepST with `task="Integration"`. Clusters `DeepST_embed` and visualizes the refined labels in `DeepST_refine_domain`. Contains three DLPFC group blocks, followed by a separate MERFISH example. |
| GraphST | [GraphST_DLPFC.ipynb](GraphST/GraphST_DLPFC.ipynb) | Reads prepared expression, label, and coordinate files together with PASTE-aligned 3D coordinates. Trains GraphST on the combined AnnData object, clusters the `emb` representation with mclust, and computes overall and per-slice metrics. The selected slices are `151673`–`151676`. |
| MENDER | [MENDER_DLPFC.ipynb](MENDER/MENDER_DLPFC.ipynb) | Loads Visium slices, normalizes expression, applies PCA and Harmony, and uses Leiden clusters as cell states in `ct`. Builds three-scale neighborhood representations with `nn_mode='ring'`, clusters the resulting spatial contexts, and saves `msm.adata_MENDER` with domain labels in `MENDER`. Includes separate blocks for all three groups. |
| SEDR | [SEDR_DLPFC.ipynb](SEDR/SEDR_DLPFC.ipynb) | Constructs per-slice spatial graphs with `k=12`, combines slices, selects 2,000 highly variable genes, and computes 200 PCA components. Trains SEDR with DEC, applies Harmony to the latent representation, and clusters `SEDR.Harmony` with mclust. Includes separate blocks for all three groups. |
| SpaBatch | [Run_DLPFC_151507.ipynb](SpaBatch/Run_DLPFC_151507.ipynb) | Builds per-slice KNN graphs with `k_cutoff=8`, combines expression and graphs, selects 2,000 highly variable genes, and computes 200 PCA components. Trains SpaBatch with DEC and clusters `SpaBatch` with mclust. Despite the filename, the current slice list is `151669`–`151672`, with five clusters. |
| SpaCross | [Visium10x_DLPFC_Multi.ipynb](SpaCross/Visium10x_DLPFC_Multi.ipynb) | Preprocesses each Visium slice with 5,000 highly variable genes, aligns slices with ICP, constructs a 3D KNN graph, and computes 200 PCA components. Runs `SC_pipeline` through `net.trian()`, stores `latent` and `recon`, and clusters the latent representation with mclust. Uses `151673`–`151676` and reports per-slice ARI, ACC, and DIS. |
| SPIRAL | [Spiral_DLPFC.ipynb](spiral/Spiral_DLPFC.ipynb) | Reads prepared expression, KNN edge, label, and coordinate files for `151669`–`151672`, then trains `SPIRAL_integration`. Exports embeddings and corrected expression, removes the noise dimensions from the embedding, and clusters the remaining `spiral` representation with mclust. Input-generation code is provided in commented cells. |
| STAGATE | [STAGATE_DLPFC.ipynb](STAGATE_pyG/STAGATE_DLPFC.ipynb) | Constructs a spatial network for each slice with `rad_cutoff=150`, combines the slices and their networks, selects 3,000 highly variable genes, normalizes and log-transforms expression, and trains STAGATE. Clusters `STAGATE` with mclust and visualizes spatial domains and UMAP. The current group is `151673`–`151676`. |
| STAligner | [STAligner_DLPFC.ipynb](STAligner/STAligner_DLPFC.ipynb) | Constructs per-slice spatial networks with `rad_cutoff=150`, selects 5,000 highly variable genes per slice, and combines the networks into a block-diagonal graph. Trains STAligner with `knn_neigh=50`, clusters `STAligner` with mclust, and evaluates the integrated domains. Includes separate blocks for the three slice groups. |
| STG3Net | [Run_DLPFC.ipynb](STG3Net/Run_DLPFC.ipynb) | Loads configuration from YAML, builds and combines per-slice spatial graphs, selects highly variable genes, and computes 200 PCA components. Trains `G3net`, stores `latent` and `recon`, clusters `latent` with mclust, and evaluates the domains. The current slice list is `151507`–`151510`, with seven clusters. |
| STitch3D | [STitch3D_DLPFC.ipynb](STitch3D/STitch3D_DLPFC.ipynb) | Loads four Visium slices and an additional single-cell reference, aligns the slices, and preprocesses them with slice distances `[10., 300., 10.]`. Trains STitch3D to obtain integrated representations and cell-type deconvolution results, then fits a seven-component Gaussian mixture to `latent` and stores reordered domain labels in `Cluster`. Uses `151507`–`151510`. |

## Data preparation

The notebooks use the following DLPFC slice groups. Many examples use seven domains for the first and third groups and five for the second; MENDER also includes clustering by resolution. Check the settings in the selected notebook block when switching groups.

| Slice group | Slice IDs | Common domain count |
| --- | --- | --- |
| Group 1 | `151507`, `151508`, `151509`, `151510` | 7 |
| Group 2 | `151669`, `151670`, `151671`, `151672` | 5 |
| Group 3 | `151673`, `151674`, `151675`, `151676` | 7 |

Most examples load Visium data from `C:\Code\Data\DLPFC`. Each slice folder needs its expression matrix and `spatial/` resources; many loaders explicitly read `filtered_feature_bc_matrix.h5`. 

## Running a notebook

1. Choose a notebook from the table and select the DLPFC slice group to analyze. For notebooks with repeated donor blocks, run the imports and helper definitions, then the relevant block. In `DeepST_integration.ipynb`, the later MERFISH cells are a separate example; `MBA_MBP.ipynb` is a mouse-brain example.
2. Update data paths, slice lists, cluster settings, and output paths. Many examples write results to `C:\Code\Answer\DLPFC`; ensure the selected output directory exists and use a distinct filename for each group.
3. Make local modules available to the notebook kernel. Most examples use imports such as `import STAligner` or `from STG3Net import Metrics`, so add this repository's `Baseline/` directory to the Python search path before those imports:

   ```python
   import sys
   sys.path.insert(0, r"C:\Code\iSpa3D-1\Baseline")
   ```

The project-level `requirements.txt` lists shared packages. CellCharter additionally imports the external `cellcharter` package. Examples using mclust require R, `rpy2`, and the R package `mclust`. Review the hard-coded `R_HOME`, `R_USER`, and CUDA settings in both notebooks and imported modules for the local environment. SPIRAL's embedding-export cells explicitly call `.cuda()`.

## Results and evaluation

The notebooks demonstrate spatial domain plots, UMAP visualizations, and method-specific exports. Many calculate overall and per-slice ARI, NMI, HOM, COM, CHAOS, PAS, and LISI-based scores.