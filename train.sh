: ${PARTITION:=biventricle}
: ${EXPNAME:=Biventricle}
: ${POLYDEGREE:="6 6 6 6"}
: ${NCHANNELS:="32 32 32 32"}
: ${RED_FACTORS:="3 3 4 4"}
: ${NTIMEFRAMES:=50}
: ${LATENTDIM_C:=8}
: ${LATENTDIM_S:=8}
: ${WKL:=1e-3}
: ${LR:=1e-4}
: ${PRECISION:=32}
: ${BATCH_SIZE:=8}
: ${PATIENCE:=3}

COMMAND="python main_autoencoder_cardiac.py \
  -c ${HOME}/01_repos/CardiacMotionRL/config_files/config_folded_c_and_s.yaml \
  --mlflow_experiment \"${EXPNAME}\" \
  --learning_rate ${LR} \
  --precision ${PRECISION} \
  --batch_size ${BATCH_SIZE} \
  --partition ${PARTITION} \
  --n_timeframes ${NTIMEFRAMES} \
  --polynomial_degree ${POLYDEGREE} \
  --reduction_factors ${RED_FACTORS} \
  --n_channels ${NCHANNELS} \
  --latent_dim_c ${LATENTDIM_C} \
  --latent_dim_s ${LATENTDIM_S} \
  --w_kl ${WKL} \
  --patience ${PATIENCE} \
  --max-epochs 10000 \
  --static_representative end_diastole"
  # --dry-run"

echo $COMMAND
$COMMAND
