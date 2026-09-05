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
from subprocess import check_output
import shlex
os.chdir(check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii'))

# %%
import vtk
import meshio
import pandas as pd
import pickle as pkl

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

import scipy
from scipy.stats import spearmanr

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from utils.CardioMesh.CardiacMesh import Cardiac3DMesh as Mesh

# %%
EPI, ENDO = 1, 2

AHA_FILENAME = "utils/CardioMesh/data/LV_4396_vertices_with_aha_segments.vtk"
lv_aha_mesh = meshio.read(AHA_FILENAME)
lv_aha_labels = lv_aha_mesh.point_data['subpartID'].astype(int)

EPIENDO_FILENAME = "utils/CardioMesh/data/LV_4396_vertices_with_epi_endo.vtk"
epiendo_mesh = meshio.read(EPIENDO_FILENAME)
epi_endo_labels = epiendo_mesh.point_data['subpartID'].astype(int)


# %% [markdown]
# ### Wall thickness

# %% [markdown]
# For each point in the epicardial surface, find closest point in the endocardial surface.

# %%
@interact
def count_vertices(aha_index=widgets.IntSlider(min=1, max=17)):
    print (sum((epi_endo_labels == EPI) & (lv_aha_labels == aha_index)))    


# %%
endo_indices = (epi_endo_labels == ENDO)
epi_aha_indices = {i: (epi_endo_labels == EPI) & (lv_aha_labels == i) for i in range(1,18)}

# %%
from tqdm import tqdm

# %%
from scipy import sparse as sp

subpart_df = pd.read_csv("/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt", header=None)
lv_indices = subpart_df == "LV"

col_ind = lv_indices.index[lv_indices[0]].to_list()
row_ind = list(range(len(col_ind)))
    
subsetting_mtx = sp.csc_matrix(
    (np.ones(len(col_ind)), (row_ind, col_ind)), 
    shape=(len(col_ind), subpart_df.shape[0])
)


# %%
def compute_thickness_per_aha(id):
    
    mean_d_per_segment = []

    for t in range(50):
    
        mean_d_per_segment.append([])
        
        timeframe = str(t+1).zfill(3)
        
        point_cloud_file = f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{id}/models/FHM_res_0.1_time{timeframe}.npy"
        
        if not os.path.exists(point_cloud_file):
            print(f"{point_cloud_file} does not exist.")
            mean_d_per_segment[t] = [0] * 17
            
        point_cloud =  np.load(point_cloud_file)        
        lv_mesh = subsetting_mtx * point_cloud
        
        # fhm_mesh = Mesh(
        #     filename=f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{id}/models/FHM_res_0.1_time{timeframe}.npy",
        #     faces_filename="/home/user/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
        #     subpart_id_filename="/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
        # )
        # 
        # lv_mesh = fhm_mesh["LV"]
        
        for segment in range(1, 18):
        
            epi_aha_mesh = lv_mesh[epi_aha_indices[segment]]
            endo_mesh = lv_mesh[endo_indices] # .unsqueeze(1).shape
            
            epi_aha_mesh_reshaped = epi_aha_mesh.reshape(epi_aha_mesh.shape[0], 1, 3)
            endo_mesh_reshaped = endo_mesh.reshape(1, endo_mesh.shape[0], 3)
            
            distance_pairs = ((epi_aha_mesh_reshaped - endo_mesh_reshaped)**2).sum(2)
            endo_closest = distance_pairs.argmin(axis=1)
            
            mean_d = np.sqrt(np.array(
                [ distance_pairs[i, endo_closest[i]] for i in range(distance_pairs.shape[0]) ]
            )).mean()
            
            mean_d_per_segment[t].append(mean_d)        
            # print(mean_d.round(3))
            
    return mean_d_per_segment


# %%
all_ids = sorted(os.listdir("/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/"))

# %%
# %%timeit
kk = np.array(compute_thickness("1000215"))

# %%
import multiprocessing

def worker_function(args):
    
    start, end = args
    
    for i in range(start, end):
        id = ids[i]
        NPY_FILE = f"notebooks/thicknesses/{id}_thickness_per_aha.npy"
        # if os.path.exists(NPY_FILE):
        #     continue
        thickness = compute_thickness_per_aha(id)
        np.save(NPY_FILE, thickness)
        
    # return dd


def parallel_for_loop(num_cores, total_iterations):
    
    chunk_size = total_iterations // num_cores
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Split the loop into chunks and assign them to different processes
    # results = 
    indices = [(i, i + chunk_size) for i in range(0, total_iterations, chunk_size)]
    # print(indices)
    pool.map(worker_function, indices)
    
    pool.close()
    pool.join()

    # return results
    # Combine results from different processes
    # final_result = sum(results)
    
    # return final_result



# %% jupyter={"outputs_hidden": true}
TOTAL_ITERATIONS = 60000
ids = [ 
    id for id in all_ids[:TOTAL_ITERATIONS] 
    if not os.path.exists(f"notebooks/thicknesses/{id}_thickness_per_aha.npy") 
]

print(len(ids))

NUM_CORES = 200 # multiprocessing.cpu_count()  # Use all available CPU cores
parallel_for_loop(NUM_CORES, TOTAL_ITERATIONS)
# result = parallel_for_loop(NUM_CORES, TOTAL_ITERATIONS)
# print("Final result:", result)

# %%
import matplotlib.pyplot as plt


# %%
@interact
def plot_thickness(id=widgets.Select(options=ids[:10]), aha_index=widgets.IntSlider(min=1, max=17, description="AHA segment")):
    t = np.load(f"notebooks/thicknesses/{id}_thickness_per_aha.npy")
    plt.scatter(x=range(50), y=t[:, aha_index-1])


# %%

# count vertices per AHA segment 
# pd.Series(mesh.point_data['subpartID'].astype(int)).value_counts().sort_index()

# %%
wall_thickness = pkl.load(open("data/transforms/cached/wall_thickness.pkl", "rb"))

# %%
wall_thickness["wall_thickness_epicardial"].shape

# %%
normalized_wt = []
for i, wt in enumerate(wall_thickness["wall_thickness_epicardial"]):
    cbrt_vol = wall_thickness["convex_hull_volumes"][i]**(1./3)
    normalized_wt.append(wt/cbrt_vol)
normalized_wt = np.array(normalized_wt)

# %%
output_dir = "output"
experiments = [x for x in os.listdir(output_dir) if os.path.exists(os.path.join(output_dir, x, ".finished"))]

w = widgets.Dropdown(
    value="2020-09-11_02-13-41",
    options=experiments,
    description='Experiment:',
    disabled=False,
)

display(w)

# %%
import ExperimentClass
experiment = ExperimentClass.ComaExperiment(os.path.join(output_dir, w.value))
experiment.load_model()

# %%
z = experiment.load_z()

# reorder
ids = np.array([int(x) for x in wall_thickness["ids"]])
# z = z.set_index("ID").loc[ids,:].drop("subset", 1)
# z = np.array(z) # Pandas DataFrame to Numpy array

# %%
dic = pkl.load(open("data/transforms/cached/2ch_segmentation__LV__ED__non_scaled__dic.pkl", "rb"))

# %%
# Get indices of each subpartition (LV endo and epi)
LVRV = Mesh("template/template.vtk").extractSubpart([1,2])

# Booleans indicating vertices that belong to each surface
endo_j = (LVRV.subpartID == 1)
epi_j = (LVRV.subpartID == 2)

# %%
faces = np.hstack([[3] + list(x) for x in LVRV.triangles])
LVRV_pv = pv.PolyData(dic["mean"], faces)

# %%
id = 2
# prepending a 3 before each triangle (PyVista format for faces)
faces_epi = np.hstack([[3] + list(x) for x in lv_epi.triangles])
lv_epi_pv = pv.PolyData(LVRV.points[epi_j], faces_epi)

faces_endo = np.hstack([[3] + list(x) for x in lv_endo.triangles])
lv_endo_pv = pv.PolyData(endo[id], faces_endo)

# lv_epi_pv.plot(scalars=wall_thickness[id])
# lv_epi_pv.rotate_z(180)
# lv_endo_pv.rotate_z(180)

# %%
corr = spearmanr(z, normalized_wt, axis=0)


# %%
def plot_mesh(mesh, faces, angle=0):
    
  surf = pv.PolyData(mesh, faces)
    
  surf.rotate_z(angle)
  plotter = pv.Plotter(notebook=True)
  # kargs = {"point_size": 2, "render_points_as_spheres": True}
    
  # surf.plot() #, **kargs)
  plotter.add_mesh(surf, show_edges=True)
  plotter.show(interactive=True)

  plotter.enable()


# %%
def f(id, angle):
  
  lv_epi_pv = pv.PolyData(epi[id], faces_epi)
  lv_endo_pv = pv.PolyData(endo[id], faces_endo)

  lv_epi_pv.rotate_z(angle)
  lv_endo_pv.rotate_z(angle)
  
  plotter = pv.Plotter(notebook=False)
          
  # plotter.add_mesh(lv_epi_pv, opacity=0.8, scalars=wall_thickness['wall_thickness_epicardial'][id])
  plotter.add_mesh(lv_epi_pv, opacity=1, scalars=corr.correlation[8:,(1,)])
  plotter.add_mesh(lv_endo_pv, opacity=1)    
  plotter.show(interactive=True)
  plotter.enable()

interact(f, 
  id = widgets.SelectionSlider(options=range(200)),
  angle = widgets.SelectionSlider(options=range(360))
)


# %%
def f():    
    plotter = pv.Plotter(notebook=True)          
    # plotter.add_mesh(LVRV_pv, opacity=0.8)#, scalars=wall_thickness['wall_thickness_epicardial'][id])
    plotter.add_mesh(lv_epi_pv, opacity=1, scalars=corr.correlation[8:,(1,)])
    plotter.add_mesh(lv_endo_pv, opacity=1)    
    plotter.show(interactive=True)
    plotter.enable()

f()
