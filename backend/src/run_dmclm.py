"""
DMCLM Master End-to-End Execution Pipeline
Routes directly to run_full_pipeline in src/run_dmclm_pipeline.py.
Executes read-only ingestion, preprocessing, feature extraction, baseline importance scoring,
consolidation, and training dataset generation.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_dmclm_pipeline import run_full_pipeline


def run_full_dmclm_pipeline(*args, **kwargs):
    return run_full_pipeline()


if __name__ == "__main__":
    run_full_dmclm_pipeline()
