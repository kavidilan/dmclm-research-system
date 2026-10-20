"""
DMCLM Model Training Pipeline (Phase 10 Integration)
Trains an interpretable Random Forest model to predict cognitive memory lifecycle states:
- RETAIN
- COMPRESS
- ARCHIVE
- FORGET

Uses the 5 DMCLM cognitive features:
1. feature_recency
2. feature_frequency
3. feature_retrieval_history
4. feature_semantic_relevance
5. feature_relationship_strength

Outputs:
- models/dmclm_lifecycle_model.joblib
- models/dmclm_importance_regressor.joblib
- models/dmclm_model.joblib (compatibility alias)
- models/model_metadata.json
- results/model_metrics.json
- results/feature_importance.json
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from phase10_ml_training import (
    Phase10ModelTrainer,
    TRAINING_CSV_PATH,
    LIFECYCLE_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
    MODEL_SAVE_PATH,
    METADATA_SAVE_PATH,
    METRICS_SAVE_PATH,
    FEATURE_IMPORTANCE_PATH,
)


def train_dmclm_model() -> Dict[str, Any]:
    print("\n=======================================================")
    print("  TRAINING CUSTOM DMCLM COGNITIVE LIFECYCLE MODEL (ML)")
    print("=======================================================")

    trainer = Phase10ModelTrainer()
    clf_pipeline, reg_pipeline, metadata = trainer.train_from_csv(TRAINING_CSV_PATH)

    # Maintain compatibility alias dmclm_model.joblib
    if LIFECYCLE_MODEL_PATH.exists():
        shutil.copy2(LIFECYCLE_MODEL_PATH, MODEL_SAVE_PATH)

    print("\n=== MODEL TRAINING & EVALUATION REPORT ===")
    print(f"Model Version:         {metadata.model_version}")
    print(f"Algorithm:             {metadata.algorithm}")
    print(f"Dataset Hash:          {metadata.dataset_hash}")
    print(f"Training Samples:      {metadata.train_samples:,}")
    print(f"Validation Samples:    {metadata.val_samples:,}")
    print(f"Test Samples:          {metadata.test_samples:,}")
    print(f"Validation Accuracy:   {metadata.val_metrics.get('accuracy', 0.0):.4f}")
    print(f"Test Accuracy:         {metadata.test_metrics.get('accuracy', 0.0):.4f}")
    print(f"Test F1 (Macro):       {metadata.test_metrics.get('f1_macro', 0.0):.4f}")
    if "roc_auc" in metadata.test_metrics:
        print(f"Test ROC-AUC:          {metadata.test_metrics['roc_auc']:.4f}")

    print("\nFeature Importances (Gini):")
    for feat, imp in metadata.feature_importance.items():
        print(f"  - {feat:30}: {imp:.4f} ({imp*100:.1f}%)")

    cm = metadata.test_metrics.get("confusion_matrix", [])
    classes = metadata.test_metrics.get("classes", [])
    print(f"\nConfusion Matrix on Unseen Test Split ({classes}):")
    for row in cm:
        print(f"  {row}")

    print("\nSaved Artifacts:")
    print(f"  - Lifecycle Classifier: {LIFECYCLE_MODEL_PATH}")
    if REGRESSOR_MODEL_PATH.exists():
        print(f"  - Importance Regressor: {REGRESSOR_MODEL_PATH}")
    print(f"  - Compatibility Model:  {MODEL_SAVE_PATH}")
    print(f"  - Model Metadata:       {METADATA_SAVE_PATH}")
    print(f"  - Metrics Summary:      {METRICS_SAVE_PATH}")
    print("=======================================================\n")

    return metadata.to_dict()


if __name__ == "__main__":
    train_dmclm_model()
