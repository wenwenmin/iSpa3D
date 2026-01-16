import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, GradientShap, Occlusion
import warnings
from tqdm import tqdm
from .Utils import find_neighboring_domains

def normalize_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + eps
    return arr / norms

def _try_int(x):
    try:
        return int(x)
    except:
        return x

def _infer_feature_names(adata, methods_attr: dict) -> np.ndarray:
    try:
        for method_dict in methods_attr.values():
            if method_dict:
                sample_key = next(iter(method_dict.keys()))
                d = method_dict[sample_key].shape[1]
                break
        else:
            return None
        
        if d is not None and hasattr(adata, 'var_names') and d == adata.n_vars:
            return np.array(list(map(str, adata.var_names)))
    except:
        pass
    return None


def _pca_attr_to_gene(attr_np: np.ndarray, pca) -> np.ndarray:
    if pca is None or not hasattr(pca, 'components_'):
        return attr_np
    comp = np.asarray(pca.components_)
    if getattr(pca, 'whiten', False) and hasattr(pca, 'explained_variance_'):
        s = np.asarray(pca.explained_variance_)
        if s.ndim == 1 and s.shape[0] == comp.shape[0]:
            scale = np.sqrt(s).reshape(1, -1)
            return (attr_np * scale) @ comp
    return attr_np @ comp


def compute_cluster_attributions(net, cluster_key, cluster_value, *, baseline_vec: torch.Tensor, batch_size: int = 64, classifier):
    mask = (net.adata.obs[cluster_key].astype(str).values == str(cluster_value))
    target_nodes = np.where(mask)[0]
    unique_clusters = sorted(map(str, net.adata.obs[cluster_key].unique()), key=_try_int)
    target_class_idx = unique_clusters.index(str(cluster_value))

    target_nodes_tensor = torch.from_numpy(target_nodes).long().to(net.device)
    full_edge_index = net._get_full_edge_index()
    
    x_full = net.X.to(net.device)
    x_full.requires_grad_(True)

    net.autoencoder.eval()
    classifier.model.eval()
    
    model = net._ClassifierScoreWrapper(
        net.autoencoder, 
        full_edge_index, 
        net.device, 
        classifier.model, 
        target_class_idx
    ).to(net.device)

    ig = IntegratedGradients(model)
    oc = Occlusion(model)
    gs = GradientShap(model)

    with torch.no_grad():
        baselines = baseline_vec.to(net.device).detach()
        if baselines.dim() == 1:
            baselines = baselines.unsqueeze(0)
        oc_baseline = baselines.mean(dim=0, keepdim=True)

    N = x_full.shape[0]
    B = baselines.shape[0]
    Fdim = x_full.shape[1]

    with torch.enable_grad():
        if B == 1:
            ig_result = ig.attribute(x_full, baselines=baselines.expand(N, -1), n_steps=50, internal_batch_size=batch_size)
        else:
            x_multi_baseline = x_full.unsqueeze(1).expand(N, B, Fdim).reshape(N * B, Fdim)
            baseline_multi = baselines.unsqueeze(0).expand(N, B, Fdim).reshape(N * B, Fdim)
            ig_expanded = ig.attribute(x_multi_baseline, baselines=baseline_multi, n_steps=50, internal_batch_size=batch_size)
            ig_result = ig_expanded.view(N, B, -1).mean(dim=1)

    with torch.enable_grad():
        oc_result = oc.attribute(x_full, sliding_window_shapes=(1,), strides=(1,), baselines=oc_baseline.expand_as(x_full), perturbations_per_eval=batch_size)
    
    with torch.enable_grad():
        gs_result = gs.attribute(x_full, baselines=baselines)

    ig_result = ig_result[target_nodes_tensor].detach().cpu().numpy()
    oc_result = oc_result[target_nodes_tensor].detach().cpu().numpy()
    gs_result = gs_result[target_nodes_tensor].detach().cpu().numpy()

    del x_full, target_nodes_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ig_result, oc_result, gs_result


def compute_and_summary_by_cluster(net, cluster_key, classifier, batch_size=64, top_k=200, pca=None, normalize=True, target_clusters=None,
                                   spatial_radius=150, spatial_ratio=0.5, max_baselines=3, selection_mode='top_k', adaptive_lambda=1.5):
    """
    Compute attributions for each cluster and select domain-specific SVGs.
    
    Args:
        net: Trained G3net model
        cluster_key: Column name in adata.obs containing cluster labels
        classifier: Trained classifier model
        batch_size: Batch size for attribution computation
        top_k: Number of genes to select (for 'top_k' mode)
        pca: PCA object for transforming attributions back to gene space
        normalize: Whether to apply L2 normalization to attributions
        target_clusters: Specific clusters to analyze (None for all)
        spatial_radius: Radius for finding neighboring domains
        spatial_ratio: Ratio threshold for neighbor identification
        max_baselines: Maximum number of baseline samples to prevent OOM
        selection_mode: Gene selection strategy ('top_k' or 'adaptive')
        adaptive_lambda: Lambda parameter for adaptive threshold (default 1.5)
    
    Returns:
        final_dfs: DataFrame per cluster with sorted genes and scores
        top_dict: Selected gene list per cluster
        per_method_dfs: Per-method differential attribution scores
    """
    present_vals = net.adata.obs[cluster_key].astype(str).values
    present_clusters = sorted(pd.unique(present_vals))

    if target_clusters is None:
        target_list = present_clusters[:]
    else:
        if isinstance(target_clusters, (str, int)):
            target_candidates = [str(target_clusters)]
        else:
            target_candidates = [str(c) for c in target_clusters]
        keyset = set(present_clusters)
        valid = [c for c in target_candidates if c in keyset]
        invalid = [c for c in target_candidates if c not in keyset]
        if invalid:
            warnings.warn(f"Ignored invalid clusters: {invalid}")
        if not valid:
            raise ValueError("No valid target clusters found")
        target_list = valid

    cluster_means = {}
    for c in present_clusters:
        idx = np.where(present_vals == str(c))[0]
        if len(idx) > 0:
            idx_tensor = torch.from_numpy(idx).long()
            with torch.no_grad():
                mean_val = net.X[idx_tensor].mean(dim=0, keepdim=True).cpu()
            cluster_means[str(c)] = mean_val

    methods_attr = {'ig': {}, 'oc': {}, 'gs': {}}
    neighbor_map = {}

    print(f"Computing attributions for {len(target_list)} clusters...")
    for c in tqdm(target_list, desc="Cluster attribution"):
        neighbor_keys = find_neighboring_domains(
            net.adata, cluster_key, c, radius=spatial_radius, ratio=spatial_ratio
        )
        
        # Limit number of baselines to prevent GPU OOM
        if len(neighbor_keys) > max_baselines:
            indices = np.linspace(0, len(neighbor_keys) - 1, max_baselines, dtype=int)
            neighbor_keys = [neighbor_keys[i] for i in indices]
            warnings.warn(f"Cluster {c}: baselines limited to {max_baselines}")
        
        neighbor_map[str(c)] = neighbor_keys
        base_list = [cluster_means.get(str(k)) for k in neighbor_keys 
                     if cluster_means.get(str(k)) is not None]
        
        if not base_list:
            warnings.warn(f"No valid baselines for cluster {c}, skipped")
            continue
        
        baselines_batch = torch.cat(base_list, dim=0)
        ig_result, oc_result, gs_result = compute_cluster_attributions(
            net, cluster_key, c, baseline_vec=baselines_batch,
            batch_size=batch_size, classifier=classifier
        )

        if pca is not None and hasattr(pca, 'components_'):
            comp_n = pca.components_.shape[0]
            if ig_result.shape[1] == comp_n:
                ig_result = _pca_attr_to_gene(ig_result, pca)
            if oc_result.shape[1] == comp_n:
                oc_result = _pca_attr_to_gene(oc_result, pca)
            if gs_result.shape[1] == comp_n:
                gs_result = _pca_attr_to_gene(gs_result, pca)

        if normalize:
            ig_result = normalize_l2(ig_result)
            oc_result = normalize_l2(oc_result)
            gs_result = normalize_l2(gs_result)

        methods_attr['ig'][str(c)] = ig_result.astype(np.float32)
        methods_attr['oc'][str(c)] = oc_result.astype(np.float32)
        methods_attr['gs'][str(c)] = gs_result.astype(np.float32)

    feat_names = _infer_feature_names(net.adata, methods_attr)
    final_dfs, top_dict, per_method_dfs = analyze_diff_from_attr(
        methods_attr, neighbor_map=neighbor_map, top_k=top_k, feat_names=feat_names,
        selection_mode=selection_mode, adaptive_lambda=adaptive_lambda
    )
    return final_dfs, top_dict, per_method_dfs


def analyze_diff_from_attr(methods_attr, neighbor_map, top_k=200, feat_names=None, selection_mode='top_k', adaptive_lambda=1.5):
    """
    Analyze differential attributions across methods and select SVGs.
    
    Args:
        methods_attr: Dictionary of attribution results per method
        neighbor_map: Dictionary mapping cluster to its neighbor clusters
        top_k: Number of top genes to select (used when selection_mode='top_k')
        feat_names: Feature/gene names
        selection_mode: Gene selection strategy:
            - 'top_k': Select fixed top-K genes by consensus score
            - 'adaptive': Select genes with log(score) > μ̂ + λσ̂ (STAMarker-style)
        adaptive_lambda: Lambda parameter for adaptive threshold (default 1.5)
    
    Returns:
        final_dfs: DataFrame per cluster with sorted genes and scores
        top_dict: Selected gene list per cluster
        per_method_dfs: Per-method differential scores
    """
    methods = list(methods_attr.keys())
    clusters = set()
    for m in methods:
        clusters |= set(map(str, methods_attr[m].keys()))
    clusters = sorted(list(clusters))

    sample_method = methods[0]
    first_cluster = next(iter(methods_attr[sample_method].keys()))
    d = methods_attr[sample_method][first_cluster].shape[1]
    
    if feat_names is not None:
        if isinstance(feat_names, (list, tuple)) and len(feat_names) != d:
            warnings.warn("feat_names length mismatch; falling back to numeric indices")
            feat_names_use = np.arange(d)
        else:
            feat_names_use = np.array(feat_names)
    else:
        feat_names_use = np.arange(d)

    per_method_dfs = {m: {} for m in methods}

    for m in methods:
        all_attr = methods_attr[m]
        for c in clusters:
            if c not in all_attr:
                continue
            A_c = np.asarray(all_attr[c])
            mean_c = np.mean(A_c, axis=0)
            
            neigh = [x for x in neighbor_map.get(c, []) if x in all_attr and x != c]
            if not neigh:
                neigh = [k for k in clusters if k != c and k in all_attr]
            
            if not neigh:
                mean_other = np.zeros_like(mean_c)
            else:
                pooled_other = np.vstack([np.asarray(all_attr[k]) for k in neigh])
                mean_other = np.mean(pooled_other, axis=0)
            
            diff = mean_c - mean_other
            df = pd.DataFrame({
                'feature': feat_names_use,
                'mean_in': mean_c,
                'mean_other': mean_other,
                'diff': diff
            }).set_index('feature')
            per_method_dfs[m][c] = df

    final_dfs, top_dict = {}, {}
    for c in clusters:
        available = [m for m in methods if c in per_method_dfs[m]]
        if not available:
            continue
        
        diffs = np.stack([per_method_dfs[m][c]['diff'].values for m in available], axis=0)
        n_feat = diffs.shape[1]
        ranks = np.argsort(np.argsort(-diffs, axis=1), axis=1)
        final_score = np.sum(n_feat - ranks, axis=0)
        
        base = per_method_dfs[available[0]][c].copy()
        base['final_score'] = final_score
        base_sorted = base.sort_values('final_score', ascending=False)
        final_dfs[c] = base_sorted
        
        # Gene selection based on mode
        if selection_mode == 'adaptive':
            # STAMarker-style adaptive threshold: log(score) > μ̂ + λσ̂
            scores = final_score.copy()
            # Add small constant to avoid log(0)
            scores_safe = scores + 1e-10
            log_scores = np.log(scores_safe)
            
            mu_hat = np.mean(log_scores)
            sigma_hat = np.std(log_scores)
            threshold = mu_hat + adaptive_lambda * sigma_hat
            
            selected_mask = log_scores > threshold
            selected_indices = np.where(selected_mask)[0]
            
            # Sort selected genes by original score
            selected_scores = scores[selected_indices]
            sorted_order = np.argsort(-selected_scores)
            selected_features = feat_names_use[selected_indices[sorted_order]]
            
            top_dict[c] = list(selected_features)
        else:
            # Default: top-k selection
            top_dict[c] = list(base_sorted.index[:top_k])

    return final_dfs, top_dict, per_method_dfs