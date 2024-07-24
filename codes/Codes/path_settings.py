"""Contain user-specific settings."""
from pathlib import Path
import os

# MODEL_SAVE_PATH = Path('./files/saved_model')
MODEL_SAVE_PATH = Path('D:\\Projects\\rt-rnn\\models')
ANA_SAVE_PATH = Path('../../data/analysis')
DATA_PATH = Path('D:\\Projects\\rt-rnn\\simulated_data')
SIM_SAVE_PATH = DATA_PATH
LOG_PATH = Path("D:/Projects/rt-rnn/logs/")
FIG_PATH = Path("D:/Projects/rt-rnn/figures/")

CONFIG_PATH = None

for path in [MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, SIM_SAVE_PATH, ANA_SAVE_PATH, FIG_PATH]: #, CONFIG_PATH]:
    os.makedirs(path, exist_ok=True)