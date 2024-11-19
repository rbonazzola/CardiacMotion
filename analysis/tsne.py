import sklearn
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import os, sys
import mlflow

from tqdm import tqdm

import ipywidgets as widgets
from ipywidgets import interact
from functools import partial

MLFLOW_TRACKING_URI = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
z_filename = lambda exp_id, run_id: f"{MLFLOW_TRACKING_URI}/{exp_id}/{run_id}/artifacts/latent_vector.csv"

df = mlflow.search_runs(experiment_ids=[str(i) for i in range(3, 9)])
df = df[(df["metrics.val_rec_ratio_to_time_mean"] < 0.8) & (df["params.dataset_n_timeframes"] == '10')]

exp_id = "4"

for i, row in tqdm(df.sort_values(["experiment_id", "metrics.val_rec_ratio_to_time_mean"]).iterrows()):        
    
    # print(row.experiment_id, row.run_id, row["metrics.val_rec_ratio_to_time_mean"])
    
    if row.experiment_id != "4":
        continue 
       
    exp_id, run_id = row.experiment_id, row.run_id
    zfn = z_filename(exp_id, run_id)
    
    if not os.path.exists(zfn):
        continue
        # print(zfn)
        
    z_df = pd.read_csv(zfn) # .head(1000)
    z_df = z_df.set_index("ID")
    
    z_static = z_df.iloc[:,:8]
    z_dynamic = z_df.iloc[:,8:]
    
    # for dim in [2,3]:
    for dim in [2]:
        for suffix, z in {"static": z_static, "dynamic": z_dynamic}.items(): # , "all": z_df}.items():
            
            print(dim)
            print(suffix)
            tsne = TSNE(n_components=dim, learning_rate='auto', init='pca', )
            t = tsne.fit_transform(z)
            tvalues_df = pd.DataFrame(t)
            
            if dim == 2:
                tvalues_df.columns = ["tsne-2d-one", "tsne-2d-two"]
            elif dim == 3:
                tvalues_df.columns = ["tsne-3d-one", "tsne-3d-two", "tsne-3d-three"]
       
            tvalues_df = tvalues_df.set_index(z.index)
        
            t_filename = f"{MLFLOW_TRACKING_URI}/{exp_id}/{run_id}/artifacts/tsne_{dim}d_z_{suffix}.csv"
            tvalues_df.to_csv(t_filename)
