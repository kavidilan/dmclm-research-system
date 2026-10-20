"""
DMCLM Comprehensive Model Evaluation Pipeline (Phase 11 Integration)
Evaluates the trained DMCLM Random Forest model against the fixed-weight baseline:
- Multi-class Accuracy, Precision, Recall, Macro/Weighted F1
- Confusion Matrix (4 classes: RETAIN, COMPRESS, ARCHIVE, FORGET)
- Regressor metrics (MAE, RMSE, R-squared, Pearson r, Spearman rho)
- Cohen's Kappa, McNemar's paired test, Wilcoxon signed-rank test
- Saves results to results/baseline_vs_ml_metrics.json and results/evaluation_report.json
"""

import json
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

from phase11_comparison import (
    Phase11BaselineMLComparator,
    TRAINING_CSV_PATH,
    LIFECYCLE_MODEL_PATH,
    COMPARISON_METRICS_PATH,
    COMPARISON_REPORT_MD_PATH,
)

REPORT_PATH = RESULTS_DIR / "evaluation_report.json"


def evaluate_dmclm() -> Dict[str, Any]:
    print("\n=======================================================")
    print("  COMPREHENSIVE DMCLM EVALUATION: ML vs BASELINE")
    print("=======================================================")

    comparator = Phase11BaselineMLComparator(
        dataset_path=TRAINING_CSV_PATH,
        classifier_path=LIFECYCLE_MODEL_PATH,
    )
    report = comparator.run_pipeline(save_artifacts=True, save_to_db=True)

    # Also save evaluation_report.json for dashboard compatibility
    rep_dict = report.to_dict()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rep_dict, f, indent=2)

    print("\n=== CLASSIFICATION PERFORMANCE COMPARISON (TEST SPLIT) ===")
    print(f"Sample Count (Unseen Test):  {report.sample_count}")
    print(f"Evaluation Split:            {report.evaluation_split}")
    print("-" * 55)
    print(f"{'Metric':<25} | {'Baseline Model':<15} | {'Trained DMCLM (ML)':<15}")
    print("-" * 55)
    b_acc = report.baseline_classification_metrics["accuracy"]
    m_acc = report.ml_classification_metrics["accuracy"]
    b_f1 = report.baseline_classification_metrics["f1_macro"]
    m_f1 = report.ml_classification_metrics["f1_macro"]
    b_prec = report.baseline_classification_metrics["precision_macro"]
    m_prec = report.ml_classification_metrics["precision_macro"]
    b_rec = report.baseline_classification_metrics["recall_macro"]
    m_rec = report.ml_classification_metrics["recall_macro"]

    print(f"{'Accuracy':<25} | {b_acc:<15.4f} | {m_acc:<15.4f}")
    print(f"{'Precision (Macro)':<25} | {b_prec:<15.4f} | {m_prec:<15.4f}")
    print(f"{'Recall (Macro)':<25} | {b_rec:<15.4f} | {m_rec:<15.4f}")
    print(f"{'F1 Score (Macro)':<25} | {b_f1:<15.4f} | {m_f1:<15.4f}")
    print("-" * 55)

    print(f"\nAgreement Rate between ML and Baseline: {report.agreement_analysis['agreement_rate']:.4f}")
    print(f"Cohen's Kappa:                         {report.agreement_analysis['cohen_kappa']:.4f}")

    print(f"\nSaved Reports:")
    print(f"  - Metrics JSON:   {COMPARISON_METRICS_PATH}")
    print(f"  - Markdown Rep:   {COMPARISON_REPORT_MD_PATH}")
    print(f"  - Legacy Report:  {REPORT_PATH}")
    print("=======================================================\n")

    return rep_dict


if __name__ == "__main__":
    evaluate_dmclm()
