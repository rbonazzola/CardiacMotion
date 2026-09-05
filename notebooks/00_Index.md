# Notebooks Index

## data/
Preparación y preprocesamiento de datos.

| Archivo | Descripción |
|---|---|
| `prepare_cardiac_datasets.ipynb` | Carga y prepara los datasets de mallas cardíacas reales (UK Biobank) |
| `compute_temporal_mean.ipynb` | Calcula la forma promedio temporal de cada sujeto |
| `compress_meshes_into_pcs.ipynb` | Proyecta mallas en componentes principales (PCA) tras aplicar transformaciones de Procrustes |
| `generate_synthetic_shapes.ipynb` | Genera poblaciones de mallas sintéticas para experimentos controlados |
| `MeshDecimation.ipynb` | Decimación de mallas (reducción de resolución) |
| `MeshPartitioning.ipynb` | Particionado de mallas en subregiones (e.g. ventrículo izquierdo, biventrículo) |

> Los archivos `.py` homónimos son versiones en script de los notebooks correspondientes.

---

## training/
Entrenamiento y carga de modelos.

| Archivo | Descripción |
|---|---|
| `train_network.ipynb` | Entrenamiento del autoencoder en datos sintéticos |
| `train_network-cardiac_data.ipynb` | Entrenamiento del autoencoder en datos cardíacos reales |
| `load_weights.ipynb` | Carga de pesos desde un run de MLflow y reconstrucción del modelo |

---

## analysis/
Análisis de resultados y variables latentes.

| Archivo | Descripción |
|---|---|
| `analysis.ipynb` | Análisis general de runs de MLflow: métricas, reconstrucciones |
| `collect_z.ipynb` | Extrae y guarda las representaciones latentes `z` de todos los sujetos a partir de runs de MLflow |
| `examine_hidden_variables.ipynb` | Exploración interactiva del espacio latente |
| `encoder_performance_analysis.ipynb` | Evaluación del encoder en datos sintéticos |
| `performance_analysis.ipynb` | Análisis de performance general del autoencoder |
| `performance_analysis-cardio.ipynb` | Análisis de performance en datos cardíacos reales |
| `wall_thickness.ipynb` | Análisis de grosor de pared ventricular a partir de las mallas |

> `performance_analysis-cardio.py` es la versión en script del notebook homónimo.

---

## dev/
Notebooks de desarrollo y pruebas. No son parte del flujo principal.

| Archivo | Descripción |
|---|---|
| `development.ipynb` | Sandbox general de desarrollo |
| `network_development.ipynb` | Prototipado de arquitecturas de red |
| `testing.ipynb` | Pruebas varias |
| `test_dataset.ipynb` | Verificación de datasets y dataloaders |
