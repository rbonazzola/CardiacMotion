import os
from typing import Literal

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))

HeartPartitionType = Literal[*VALID_PARTITIONS]

# Define directories based on the package location
DATA_DIR   = os.path.join(PACKAGE_DIR, "data")
CACHE_DIR  = os.path.join(DATA_DIR, "cached")
MESHES_DIR = os.path.join(DATA_DIR, "meshes")