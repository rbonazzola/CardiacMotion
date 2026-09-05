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
import os
import pandas as pd 
import mlflow

HOME = os.environ["HOME"]
CARDIAC_GWAS_REPO = f"{HOME}/01_repos/CardiacGWAS"
CARDIAC_COMA_REPO = f"{HOME}/01_repos/CardiacCOMA"
MLRUNS_DIR = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/"

mlflow.set_tracking_uri(MLRUNS_DIR)
df = mlflow.search_runs(experiment_ids=[str(i) for i in range(3, 9)])
df = df[(df["metrics.val_rec_ratio_to_time_mean"] < 0.7) & (df["params.dataset_n_timeframes"] == '10')]
df["partition"] = df.experiment_id.apply(lambda expid: mlflow.get_experiment(expid).name)
df = df.set_index("run_id")
df = df[df.status == "FINISHED"]

z_df_merged = []

for runid, row in df.iterrows():

    expid = df.loc[runid].experiment_id
    latent_vector_file = f"{MLRUNS_DIR}/{expid}/{runid}/artifacts/latent_vector.csv"
    print(latent_vector_file)
    z_df = pd.read_csv(latent_vector_file)
    z_df = z_df.set_index("ID")
    z_df.columns = [ f"{expid}_{runid}_{z}" for z in z_df.columns ]
    z_df = z_df.sort_index()
    z_df.index = z_df.index.astype(int)
    z_df_merged.append(z_df)

z_df_merged = pd.concat(z_df_merged, axis=1)

z_df_merged.to_csv(
    f"/home/rodrigo/01_repos/CardiacMotionGWAS/output/latent_vector_all_chambers.csv", 
    index=True, index_label="ID", float_format='%.4f'
)

print(z_df_merged.shape)

# %%
