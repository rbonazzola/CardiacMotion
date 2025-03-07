SYNTHETIC_DATASET_PARAMS = [
    "dataset_type",        
    "dataset_freq_max",
    "dataset_l_max",
    "dataset_complexity",
    "dataset_complexity_s",
    "dataset_complexity_c",
    "dataset_max_static_amplitude",
    "dataset_max_dynamic_amplitude"
]

SYNTHETIC_DATASET_PARAMS_MLFLOW = [ f"params.{k}" for k in SYNTHETIC_DATASET_PARAMS ]