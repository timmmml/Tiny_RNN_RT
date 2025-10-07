"""Contain user-specific settings."""
from pathlib import Path
import os

MODEL_SAVE_PATH = Path('../../saved_models')
ANA_SAVE_PATH = Path('../../data/analysis')
DATA_PATH = Path('../../data')
SIM_SAVE_PATH = DATA_PATH
LOG_PATH = Path("../../logs/")
FIG_PATH = Path("../../figures/")
OBJECT_PATH = Path("../../data/")
CONFIG_PATH = None

for path in [MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, SIM_SAVE_PATH, ANA_SAVE_PATH, FIG_PATH]: #, CONFIG_PATH]:
    os.makedirs(path, exist_ok=True)