import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "../../mlruns"
MLFLOW_URI = "mlruns"

mlflow.set_tracking_uri(MLFLOW_URI)
assert os.path.exists(MLFLOW_URI), f"{MLFLOW_URI} does not exist"

client = MlflowClient()

missing_parameter = "dataset_static_representative"
default_value = "temporal_mean"

experiment_ids = [ str(k) for k in range(3, 8) ]

# Obtener todas las runs del experimento
runs = client.search_runs(experiment_ids=experiment_ids, filter_string="", max_results=10000)
print(runs)
for run in runs:
    run_id = run.info.run_id
    params = run.data.params
    
    if missing_parameter not in params:
        print(f"Adding {missing_parameter} to run {run_id} with value {default_value}")
        client.log_param(run_id, missing_parameter, default_value)
    else:
        print(f"Run {run_id} already has '{missing_parameter}', skipping...")

print("Patch completed.")
