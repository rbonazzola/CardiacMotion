import os, sys; 

sys.path.append(PKG_DIR := os.path.dirname(os.path.realpath(__file__)))

from models.Model3D import (
  Encoder3DMesh, 
  Decoder3DMesh)

from models.Model4D import (
  AutoencoderTemporalSequence,
  EncoderTemporalSequence,
  DecoderTemporalSequence,
  DecoderContent,
  DecoderStyle,
  ENCODER_ARGS,
  DECODER_C_ARGS,
  DECODER_S_ARGS)

from models.TemporalAggregators import (
  TemporalAggregator, 
  FCN_Aggregator)

MLFLOW_URI = os.getenv("MLFLOW_URI", f"{os.path.dirname(PKG_DIR)}/mlruns")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])

logger = logging.getLogger()