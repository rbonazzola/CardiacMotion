# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from scipy import sparse as sp

# %%
df = pd.read_csv("/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt", header=None)
df.columns = ["partition"]

# %%
partitions = {
  "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
  "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
  "left_ventricle" : ("LV", "AVP", "MVP"),
  "right_ventricle" : ("RV", "PVP", "TVP"),
  "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
  "aorta" : ("aorta",)
}

# %%
subpart_names = ["LV", "AVP", "LA", "MVP", "RV", "PVP", "PV1", "PV2", "PV3", "PV4", "PV5", "RA", "TVP", "PV6", "PV7", "aorta"]

# %%
for subpart_name in subpart_names:
    df[subpart_name] = df.partition == subpart_name

# %%
for partition in partitions:    
    indices[partition] = df[list(partitions[partition])].apply(any, axis=1)

# %%
col_ind = indices["left_ventricle"].index[indices["left_ventricle"]].to_list()
row_ind = list(range(len(col_ind)))

subsetting_mtx = sp.csc.csc_matrix(
  (np.ones(len(col_ind)), (row_ind, col_ind)), 
  shape=(len(col_ind), df.shape[0])
)

# %%
subsetting_mtx

# %%
[sum(v) for k,v in indices.items()]
