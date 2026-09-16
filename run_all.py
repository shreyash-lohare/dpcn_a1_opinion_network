"""Reproduces the sanitisation, imputation, row-centring, and item-network
outputs from a fresh clone. Run from the repo root: `python run_all.py`.
"""

from src.config import PipelineConfig
from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(PipelineConfig())
