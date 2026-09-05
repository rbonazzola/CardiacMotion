import os
import re
import mlflow

# directorio con los archivos
src_dir = "../CardiacMotionGWAS/results/gwas/Unsupervised_spatiotemporal/summaries"

# regex para capturar el run id entre los "__"
pattern = re.compile(r"GWAS__z\d+_([a-f0-9]{32})__regionwise_summary\.tsv")

resultados = []
for fname in os.listdir(src_dir):
    m = pattern.match(fname)
    if not m:
      continue  # no coincide con el patrón
    run_id = m.group(1)
    print(f"Archivo: {fname}")
    print(f"Run ID: {run_id}")

    # cargar info de MLflow
    try:
        run = mlflow.get_run(run_id)
        params = run.data.params
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        experiment_name = experiment.name

        resultados.append({
            "archivo": fname,
            "run_id": run_id,
            "experimento": experiment_name,
            **params
        })
                                                
    except Exception as e:
        print(f"  Error al obtener run {run_id}: {e}")

print("-" * 40)
import pandas as pd
df = pd.DataFrame(resultados)
df.to_csv("params.csv", index=False)
