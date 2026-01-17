# iSpa3D: An interpretable deep learning framework for 3D spatial domain reconstruction and domain-specific gene discovery in multi-slice spatial transcriptomics

## Overview

iSpa3D is an interpretable deep learning framework designed to achieve high-fidelity 3D spatial domain reconstruction by integrating multi-slice Spatially Resolved Transcriptomics (SRT) data while simultaneously identifying domain-specific genes. By leveraging a novel **Cluster-Level Contrastive Learning (CLCL) module** and a built-in interpretability module powered by attribution analysis, iSpa3D empowers domain-specific gene discovery, ranging from the identification of tissue architectures and dynamic spatiotemporal patterns to the elucidation of molecular drivers underlying 3D tissue structures.

![iSpa3D.png](iSpa3D.png)

## Data

All public datasets used in this paper are available at [Zenodo](https://zenodo.org/uploads/18012911).

## Installation

1. Clone repository:

```bash
git clone https://github.com/wenwenmin/iSpa3D.git
```

2. Create virtual environment:

```bash
conda create --name ispa3d python=3.10
```

3. Activate virtual environment:

```bash
conda activate ispa3d
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
iSpa3D/
├── Config/           # Experiment configurations (YAML)
├── iSp3D/            # Core modules for model and utilities
├── Baseline/         # Baseline methods for comparison
├── Tutorial/         # Jupyter notebooks for tutorials
└── README.md         # Documentation
```

## Running the Code

Once your environment is set up, follow these steps to get started with iSpa3D using the DLPFC dataset as an example:

### 1. Download the Data

iSpa3D uses the **DLPFC dataset** for this tutorial. Download the dataset from [Zenodo](https://zenodo.org/uploads/18012911).

**Key Path Definitions:**

* **`data_root`**: This is the absolute path to the dataset folder containing the raw SRT data.
* **`proj_list`**: A list of spatial tissue slices to process (e.g., `['151673', '151674', '151675', '151676']`).

**Required File Structure:**

```text
/data/DLPFC/                    <-- Root folder (data_root)
├── 151673/                      <-- Slice 1 data (proj_list item)
│   ├── filtered_feature_bc_matrix/
│   ├── spatial/
│   └── 151673_truth.txt         <-- Ground truth labels
├── 151674/
├── 151675/
└── 151676/
```

### 2. Prepare Configuration File

A configuration file in YAML format controls all aspects of the analysis. We provide example configurations for various datasets in the `Config/` directory.

**Create or modify your configuration (e.g., `Config_DLPFC.yaml`):**

```yaml
data:
  dataset: DLPFC
  k_cutoff: 12                   # Number of nearest neighbors
  top_genes: 2000                # Number of highly variable genes

model:
  class_num: 7                   # Number of spatial domains
  latent_dim: 16                 # Dimension of latent representation
  # ... other model parameters

train:
  epochs: 1000
  lr: 0.001                      # Learning rate
  # ... other training parameters

```

### 3. Execution

Run the tutorial script to process your data step by step:

```bash
jupyter notebook Tutorial/Run_DLPFC.ipynb
```

**Key steps in the tutorial:**

1. **Data Loading & Preprocessing**
   - Load multi-slice Visium data
   - Filter genes and normalize expression
   - Perform PCA dimensionality reduction

2. **Model Training**
   - Initialize G3net model
   - Train with contrastive learning
   - Obtain 3D spatial domain predictions

3. **Clustering & Evaluation**
   - Cluster spots into spatial domains
   - Compute evaluation metrics (ARI, NMI, etc.)
   - Visualize results on spatial coordinates

4. **Attribution Analysis**
   - Train a classifier on latent representations
   - Compute attribution scores for each gene
   - Identify domain-specific marker genes

5. **Visualization**
   - Plot spatial domains on tissue slices
   - Generate UMAP visualizations
   - Create marker gene expression heatmaps

---

## Example Usage with DLPFC

```python
import iSp3D as MODEL
from iSp3D.Classifier import iSpaNetClassifier
from iSp3D.Attribution import compute_and_summary_by_cluster
import torch

# Load configuration
with open('Config/Config_DLPFC.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load and preprocess data
adata, graph_dict, pca = get_data(proj_list, config['data'])

# Initialize model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = MODEL.G3net(adata, graph_dict=graph_dict, device=device, 
                  config=config, num_cluster=config['model']['class_num'])

# Train model
net.train()

# Get latent representations and predictions
enc_rep, recon = net.process()
adata = net.clustering(adata, num_cluster=config['model']['class_num'], 
                       used_obsm='latent', key_added_pred='mclust')

# Fine-tune with classifier for attribution analysis
classifier = iSpaNetClassifier(in_features=config['model']['latent_dim'], 
                               n_classes=config['model']['class_num'])
classifier.prepare_data(g3net_model=net, adata=adata, target_labels=cluster_labels)
classifier.train(epochs=500, early_stop_patience=50)

# Identify domain-specific marker genes
res_df = compute_and_summary_by_cluster(net=net, cluster_key='mclust', 
                                        classifier=classifier)
```

---

> **Note**: For multi-GPU training or larger datasets, adjust the configuration file parameters (`num_cluster`, `latent_dim`, etc.) to optimize performance for your hardware.

## Contact details

If you have any questions, please contact huangxu_bzcd@stu.ynu.edu.cn and minwenwen@yun.edu.cn.


