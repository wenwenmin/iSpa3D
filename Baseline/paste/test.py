import numpy as np
import scanpy as sc
import pandas as pd
import paste as pst
import ot
import torch
from paste.helper import match_spots_using_spatial_heuristic
import os

# 确保输出目录存在
dirs = r"C:\Code\Data\MERFISH/"

# 定义样本组
proj_list = ['0.04', '0.09', '0.14', '0.19', '0.24']
print(f"处理样本组: {proj_list}")

# 生成文件标识
flags = ""
for i in proj_list:
    flags = flags + "_" + i

# 加载该组所有样本数据
slices = []

for proj_name in proj_list:
    # 加载h5ad文件
    adata_tmp = sc.read_h5ad(os.path.join(dirs, f"hypothalamus-{proj_name}.h5ad"))
    adata_tmp.var_names_make_unique()

    # 如果坐标有负值，进行调整确保都为正值
    if adata_tmp.obsm["spatial"].min() < 0:
        adata_tmp.obsm["spatial"] = adata_tmp.obsm["spatial"] + np.abs(adata_tmp.obsm["spatial"].min()) + 100

    slices.append(adata_tmp)

# 依次对齐样本
pis = []
for i in range(len(slices) - 1):
    # 使用空间启发式方法创建初始映射
    pi0 = match_spots_using_spatial_heuristic(slices[i].obsm['spatial'],
                                              slices[i + 1].obsm['spatial'],
                                              use_ot=True)

    # 对齐样本对
    pi = pst.pairwise_align(slices[i], slices[i + 1],
                            alpha=0.1,
                            G_init=pi0,
                            norm=False,
                            backend=ot.backend.TorchBackend(),
                            use_gpu=True)
    pis.append(pi)

# 整合所有样本到共同坐标系
new_slices = pst.stack_slices_pairwise(slices, pis)

# 计算新的坐标
all_spatial = []
all_cells = []
all_slice_ids = []

for i, s in enumerate(new_slices):
    current_spatial = s.obsm['spatial']
    all_spatial.append(current_spatial)
    all_cells.append(s.obs_names)
    all_slice_ids.append(np.repeat(proj_list[i], len(s)))

new_coord = np.vstack(all_spatial)
cells = np.concatenate(all_cells)
slice_ids = np.concatenate(all_slice_ids)

# 创建并保存坐标数据框
new_coord_df = pd.DataFrame(
    data={
        'x': new_coord[:, 0],
        'y': new_coord[:, 1],
        'slice': ['-' + str(id) for id in slice_ids]
    },
    index=cells
)
new_coord_df.to_csv(dirs + "paste_new_coord" + flags + ".csv")

print(f"完成样本组 {proj_list} 的整合，结果已保存到 {dirs}paste_new_coord{flags}.csv")