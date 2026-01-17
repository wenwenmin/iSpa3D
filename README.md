#  iSpa3D: An interpretable deep learning framework for 3D spatial domain reconstruction and domain-specific gene discovery in multi-slice spatial transcriptomics
## Introduction
With the rapid accumulation of multi-slice Spatially Resolved Transcriptomics (SRT) data, there is a pressing need for computational frameworks that can reconstruct 3D tissue architectures. However, a critical analytical gap exists: existing methods either build 3D structural models without providing molecular interpretation, or they identify spatially variable genes (SVGs) only in 2D contexts, failing to link them to the complex 3D structures they are meant to characterize. To address this, we present **iSpa3D**, an interpretable deep learning framework that unifies high-fidelity 3D spatial domain reconstruction with domain-specific gene discovery. iSpa3D employs a novel **Cluster-Level Contrastive Learning (CLCL) module** for robust multi-slice integration and features a built-in interpretability module to identify the molecular drivers of the reconstructed 3D domains. We applied iSpa3D to eight diverse datasets across multiple species and platforms (10x Visium, ST, Stereo-seq, Slide-seqV2, MERFISH, and STARmap PLUS). Comprehensive validation demonstrates that iSpa3D consistently ranks among the top-tier of state-of-the-art methods in 3D spatial domain identification while effectively correcting batch effects. It successfully captures conserved tissue architectures, tracks dynamic spatiotemporal changes in developing embryos, and robustly reconstructs 3D models in pathological contexts. Crucially, its integrated **interpretability module** successfully identifies key domain-specific SVGs, providing direct molecular validation for the reconstructed 3D structures. Overall, iSpa3D offers a robust and interpretable end-to-end framework that bridges the critical gap between 3D tissue architecture and its underlying molecular drivers.

![iSpa3D.png](iSpa3D.png)

## Data
- All public datasets used in this paper are available at [Zenodo](https://zenodo.org/uploads/18012911)

## Setup
-   `pip install -r requirement.txt`
- NVIDIA GPU (a single Nvidia GeForce RTX 3090)

## Get Started
We provided codes for reproducing the experiments of the paper "iSpa3D: An interpretable deep learning framework for 3D spatial domain reconstruction and domain-specific gene discovery in multi-slice spatial transcriptomics", and comprehensive tutorials for using iSpa3D.
- Please see `Tutorial`.


