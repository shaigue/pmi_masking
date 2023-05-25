"""Contains project level configurations."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / 'experiment_config'
VOCABS_DIR = PROJECT_ROOT / 'pmi_masking_vocabs'
