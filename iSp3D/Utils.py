
import os
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib
from sklearn.cluster import KMeans
import scanpy as sc

# Set R environment variables
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.2'
os.environ['R_USER'] = r'C:\Users\CLEARLOVE\.conda\envs\STG\Lib\site-packages\rpy2'

def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=2023):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res - increment

def louvain_py(adata, num_cluster, used_obsm='latent', key_added_pred='G3STNET', random_seed=2023):
    sc.pp.neighbors(adata, use_rep=used_obsm)
    res = res_search_fixed_clus_louvain(adata, num_cluster, increment=0.01, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added_pred] = adata.obs['louvain']
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

def Kmeans_cluster(adata, num_cluster, used_obsm='latent', key_added_pred="G3STNET", random_seed=2024):
    np.random.seed(random_seed)
    cluster_model = KMeans(n_clusters=num_cluster, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(adata.obsm[used_obsm])
    adata.obs[key_added_pred] = cluster_labels
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='latent', key_added_pred='G3STNET',
             random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added_pred] = mclust_res
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

def create_dictionary_gnn(adata, use_rep, use_label, batch_name, k = 50,  verbose = 1, mask_rate=0.5):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    cells = []
    remains_cells = []
    for i in batch_list.unique():
        cells.append(cell_names[batch_list == i])
        remains_cells.append(cell_names[batch_list != i])

    mnns = dict()
    u_unique_set = None
    n_unique_set = None
    for idx, b_name in enumerate(batch_list.unique()):
        key_name = b_name + "_" + "rest" 
        mnns[key_name] = {}

        new = list(cells[idx])
        ref = list(remains_cells[idx])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        gt1 = adata[new].obs[use_label]
        gt2 = adata[ref].obs[use_label]
        names1 = new
        names2 = ref
        match, u_unique_set, n_unique_set = GNN(ds1, ds2, gt1, gt2, names1, names2, u_unique_set, n_unique_set, knn=k, mask_rate=mask_rate)
        if (verbose > 0):
            print('Processing datasets {0} have {1} nodes or edges'.format(b_name, len(match)))

        if len(match) > 0:
            G = nx.Graph() 
            G.add_edges_from(match) 
            node_names = np.array(G.nodes) 
            anchors = list(node_names) 
            adj = nx.adjacency_matrix(G) 
            tmp = np.split(adj.indices, adj.indptr[1:-1]) 
            for i in range(0, len(anchors)):
                key = anchors[i] 
                i = tmp[i] 
                names = list(node_names[i]) 
                mnns[key_name][key]= names 
    return(mnns)

def GNN(target_slice_ds, rest_slice_ds, gt1, gt2, names1, names2, u_unique_set=None, n_unique_set=None, knn=20, approx=False, mask_rate=0.5):
    if u_unique_set is None:
        u_unique_set = set()
    if n_unique_set is None:
        n_unique_set = set()

    similarity = torch.matmul(torch.tensor(target_slice_ds), torch.transpose(torch.tensor(rest_slice_ds), 1, 0))
    _, I_knn = similarity.topk(k=knn, dim=1, largest=True, sorted=False)

    mask = torch.rand(I_knn.shape) < mask_rate
    I_knn[mask] = -1
    match_lst = []
    for i in range(I_knn.shape[0]):
        gt = gt1[i]
        gt_tmp = set(gt2[gt2 == gt].index) 
        for j in I_knn[i]:
            if j == -1:
                continue 
            if names2[j] not in gt_tmp:
                continue
            item = (names1[i], names2[j])
            ex_item = (names2[j], names1[i])

            if ex_item in u_unique_set:
                n_unique_set.add(ex_item)
                continue
            if item not in u_unique_set:
                u_unique_set.add(item)
                match_lst.append(item)
    return match_lst, u_unique_set, n_unique_set

def compute_gene_spatial_pvalue_vectorized(adata, gene, cluster_key, target_cluster,
                                           n_permutations=1000, random_seed=42):
    if gene not in adata.var_names:
        return np.nan
    
    rng = np.random.default_rng(random_seed)
    
    mask = adata.obs[cluster_key].astype(str) == str(target_cluster)
    n_in_cluster = mask.sum()
    
    if n_in_cluster == 0:
        return 1.0
    
    gene_expr = adata[:, gene].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray().flatten()
    else:
        gene_expr = np.asarray(gene_expr).flatten()
        
    n_total = len(gene_expr)
    
    if n_in_cluster > n_total:
        return 1.0

    observed_mean = gene_expr[mask].mean()
    
    indices = np.arange(n_total)
    shuffled_indices_all = rng.permuted(np.tile(indices, (n_permutations, 1)), axis=1)
    perm_indices = shuffled_indices_all[:, :n_in_cluster]
    perm_expr = gene_expr[perm_indices]
    perm_means = perm_expr.mean(axis=1)
    
    n_greater = np.sum(perm_means >= observed_mean)
    n_less = np.sum(perm_means <= observed_mean)
    
    p_value = 2 * min((n_greater + 1) / (n_permutations + 1), (n_less + 1) / (n_permutations + 1))
    
    return min(p_value, 1.0)

def count_nbr(target_cluster, df, radius):
    target_df = df[df['pred'] == str(target_cluster)]
    num_nbr = []
    
    for index, row in target_df.iterrows():
        x_pos = row['x']
        y_pos = row['y']
        tmp_nbr = df[((df['x'] - x_pos)**2 + (df['y'] - y_pos)**2) <= (radius**2)]
        num_nbr.append(tmp_nbr.shape[0])
    
    return np.mean(num_nbr)

def search_radius(target_cluster, df, start, end, num_min=8, num_max=15, max_run=100):
    run = 0
    num_low = count_nbr(target_cluster, df, start)
    num_high = count_nbr(target_cluster, df, end)
    
    if num_min <= num_low <= num_max:
        return start
    elif num_min <= num_high <= num_max:
        return end
    elif num_low > num_max:
        return None
    elif num_high < num_min:
        return None
    
    while (num_low < num_min) and (num_high > num_min):
        run += 1
        if run > max_run:
            return None
        
        mid = (start + end) / 2
        num_mid = count_nbr(target_cluster, df, mid)
        
        if num_min <= num_mid <= num_max:
            return mid
        
        if num_mid < num_min:
            start = mid
            num_low = num_mid
        elif num_mid > num_max:
            end = mid
            num_high = num_mid

def find_neighboring_domains(adata, cluster_key, target_cluster, radius=300, ratio=0.5):
    coords = adata.obsm['spatial']
    df = pd.DataFrame({
        'cell_id': adata.obs_names,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'pred': adata.obs[cluster_key].astype(str),
        'batch': adata.obs['batch_name'].astype(str) if 'batch_name' in adata.obs else 'unknown'
    })
    df.index = df['cell_id']
    
    cluster_num = df['pred'].value_counts().to_dict()
    target_df = df[df['pred'] == str(target_cluster)]

    slice_neighbor_results = {}
    all_batches = target_df['batch'].unique()
    
    for batch in all_batches:
        target_df_batch = target_df[target_df['batch'] == batch]
        df_same_batch = df[df['batch'] == batch]
        nbr_num_batch = {}
        
        for index, row in target_df_batch.iterrows():
            tmp_nbr = df_same_batch[((df_same_batch['x'] - row['x'])**2 + 
                                     (df_same_batch['y'] - row['y'])**2) <= (radius**2)]
            for p in tmp_nbr['pred']:
                if p != str(target_cluster):
                    nbr_num_batch[p] = nbr_num_batch.get(p, 0) + 1

        nbr_filtered_batch = [k for k, v in nbr_num_batch.items() 
                             if v > (ratio * cluster_num.get(k, 0))]
        slice_neighbor_results[batch] = nbr_filtered_batch

    neighbor_votes = {}
    for batch, neighbors in slice_neighbor_results.items():
        for nbr in neighbors:
            neighbor_votes[nbr] = neighbor_votes.get(nbr, 0) + 1
    
    # Get neighbors in majority of slices (> 50%)
    num_slices = len(all_batches)
    majority_threshold = num_slices / 2
    neighbor_list = [nbr for nbr, votes in neighbor_votes.items() 
                     if votes > majority_threshold]
    
    # Sort by vote count
    neighbor_list.sort(key=lambda x: -neighbor_votes[x])
    
    if len(neighbor_list) == 0 and neighbor_votes:
        neighbor_list = [max(neighbor_votes, key=neighbor_votes.get)]
    
    print(f"Cluster {target_cluster}: neighbors = {neighbor_list}")
    return neighbor_list

def compute_expression_ratio_and_fc(expr_slice, target_mask, neighbor_mask):
    target_expressing = np.sum((expr_slice > 0) & target_mask)
    neighbor_expressing = np.sum((expr_slice > 0) & neighbor_mask)
    
    target_total = target_mask.sum()
    neighbor_total = neighbor_mask.sum()
    
    target_ratio = target_expressing / target_total if target_total > 0 else 0
    neighbor_ratio = neighbor_expressing / neighbor_total if neighbor_total > 0 else 0
    
    target_mean_expr = expr_slice[target_mask].mean() if target_total > 0 else 0
    neighbor_mean_expr = expr_slice[neighbor_mask].mean() if neighbor_total > 0 else 0
    
    epsilon = 1e-10
    fold_change = (target_mean_expr + epsilon) / (neighbor_mean_expr + epsilon)
    
    return target_ratio, neighbor_ratio, fold_change

def process_gene(gene, gene_to_idx, target_mask, neighbor_masks, neighbor_domains, X_recon, pvalue_adata, cluster_key, 
                 target_cluster_id, expression_threshold=0.8, ratio_threshold=1.0, fc_threshold=1.5, neighbor_size_ratio=0.5,):
    gene_idx = gene_to_idx.get(gene)
    if gene_idx is None:
        return None

    pvalue = compute_gene_spatial_pvalue_vectorized(pvalue_adata, gene, cluster_key=cluster_key, target_cluster=target_cluster_id)

    expr_slice = X_recon[:, gene_idx]
    
    target_expressing_spots = np.sum((expr_slice > 0) & target_mask)
    target_total_spots = target_mask.sum()
    target_expression_ratio = target_expressing_spots / target_total_spots if target_total_spots > 0 else 0
    
    passes_neighbor_criteria = True
    neighbor_ratios = []
    fold_changes = []
    eligible_neighbors = 0
    
    for neighbor_domain in neighbor_domains:
        neighbor_mask = neighbor_masks[neighbor_domain]
        neighbor_total = neighbor_mask.sum()
        
        if neighbor_total < neighbor_size_ratio * target_total_spots:
            continue
        
        eligible_neighbors += 1
        target_ratio, neighbor_ratio, fc = compute_expression_ratio_and_fc(
            expr_slice, target_mask, neighbor_mask
        )
        
        neighbor_ratios.append(neighbor_ratio)
        fold_changes.append(fc)
        
        ratio_check = (target_ratio / neighbor_ratio) > ratio_threshold if neighbor_ratio > 0 else True
        fc_check = fc > fc_threshold
        
        if not (ratio_check and fc_check):
            passes_neighbor_criteria = False
    
    if eligible_neighbors == 0:
        passes_neighbor_criteria = False
    
    avg_neighbor_ratio = np.mean(neighbor_ratios) if neighbor_ratios else 0
    avg_fold_change = np.mean(fold_changes) if fold_changes else 0
    min_fold_change = np.min(fold_changes) if fold_changes else 0
    
    result = {
        'gene': gene,
        'p_value': pvalue,
        'target_expression_ratio': target_expression_ratio,
        'avg_neighbor_ratio': avg_neighbor_ratio,
        'avg_fold_change': avg_fold_change,
        'min_fold_change': min_fold_change,
        'eligible_neighbors': eligible_neighbors,
        'total_neighbors': len(neighbor_domains),
        'passes_expression_threshold': target_expression_ratio > expression_threshold,
        'passes_neighbor_criteria': passes_neighbor_criteria,
    }
    
    return result

def acquire_pairs(X, Y, k=30, metric='angular'):
    """Find mutual nearest neighbors between two datasets using AnnoyIndex."""
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i, sorted_mat[i]] = True
    _ = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i], i] = True
    mnn_mat = np.logical_and(_, mnn_mat).astype(int)
    return mnn_mat

def best_fit_transform(A, B):
    assert A.shape == B.shape
    m = A.shape[1]
    
    # Translate to centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # Homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def transform(point_cloud, T):
    point_cloud_align = np.ones((point_cloud.shape[0], 3))
    point_cloud_align[:, 0:2] = np.copy(point_cloud)
    point_cloud_align = np.dot(T, point_cloud_align.T).T
    return point_cloud_align[:, :2]

def align_spatial_slices(adata, proj_list, n_sample=5000, random_seed=1234):
    import matplotlib.pyplot as plt
    
    n_slice = len(proj_list)
    adata_st_list = [adata[adata.obs['batch_name'] == proj_name].copy() for proj_name in proj_list]
    
    # Ensure var_names consistency
    for i_slice in range(n_slice):
        adata_st_list[i_slice].var_names = adata.var_names
    
    # Initialize spatial_regi
    for i_slice in range(n_slice):
        if 'spatial' in adata_st_list[i_slice].obsm:
            adata_st_list[i_slice].obsm["spatial_regi"] = adata_st_list[i_slice].obsm["spatial"].copy()
        elif 'X' in adata_st_list[i_slice].obs.columns and 'Y' in adata_st_list[i_slice].obs.columns:
            adata_st_list[i_slice].obsm["spatial_regi"] = adata_st_list[i_slice].obs[['X', 'Y']].values
        else:
            print(f"Warning: slice {proj_list[i_slice]} has no spatial coords, skipping")
            continue
    
    n_sample_actual = min(n_sample, min([ad.shape[0] for ad in adata_st_list]))
    print(f"Using {n_sample_actual} samples for alignment")
    np.random.seed(random_seed)
    
    # Progressive alignment
    for i_slice in range(n_slice - 1):
        print(f"\nAligning slice {proj_list[i_slice]} and {proj_list[i_slice+1]}")
        
        loc0 = adata_st_list[i_slice].obsm["spatial_regi"]
        loc1 = adata_st_list[i_slice + 1].obsm["spatial_regi"]
        
        latent_0 = adata[adata_st_list[i_slice].obs.index].obsm['latent']
        latent_1 = adata[adata_st_list[i_slice + 1].obs.index].obsm['latent']
        
        # Sampling
        n_sample_0 = min(n_sample_actual, latent_0.shape[0])
        n_sample_1 = min(n_sample_actual, latent_1.shape[0])
        ss_0 = np.random.choice(latent_0.shape[0], size=n_sample_0, replace=False)
        ss_1 = np.random.choice(latent_1.shape[0], size=n_sample_1, replace=False)
        
        loc0_sample = loc0[ss_0, :]
        loc1_sample = loc1[ss_1, :]
        latent_0_sample = latent_0[ss_0, :]
        latent_1_sample = latent_1[ss_1, :]
        
        # Find MNN pairs
        mnn_mat = acquire_pairs(latent_0_sample, latent_1_sample, k=1, metric='euclidean')
        idx_0, idx_1 = [], []
        for i in range(mnn_mat.shape[0]):
            if np.sum(mnn_mat[i, :]) > 0:
                nns = np.where(mnn_mat[i, :] == 1)[0]
                for j in list(nns):
                    idx_0.append(i)
                    idx_1.append(j)
        
        if len(idx_0) == 0:
            print(f"Warning: no MNN pairs found, skipping alignment")
            continue
        
        print(f"Found {len(idx_0)} MNN pairs")
        
        loc0_pair = loc0_sample[idx_0, :]
        loc1_pair = loc1_sample[idx_1, :]
        
        # Calculate transformation
        T, _, _ = best_fit_transform(loc1_pair, loc0_pair)
        
        # Apply transformation
        loc1_new = transform(loc1, T)
        adata_st_list[i_slice + 1].obsm["spatial_regi"] = loc1_new
    
    # Update main adata with aligned coordinates
    for i_slice, proj_name in enumerate(proj_list):
        idx = adata.obs['batch_name'] == proj_name
        adata.obsm['spatial_regi'] = np.zeros_like(adata.obsm['spatial']) if 'spatial_regi' not in adata.obsm else adata.obsm['spatial_regi']
        adata.obsm['spatial_regi'][idx] = adata_st_list[i_slice].obsm["spatial_regi"]
    
    print("\nSpatial alignment completed")
    return adata