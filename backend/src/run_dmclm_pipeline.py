"""
Master DMCLM Pipeline Orchestrator (Member 4 — Trained ML Integration)
Executes the end-to-end DMCLM research workflow:
1. Ingests raw multi-modal observations from DesktopResearchCollector in READ-ONLY mode.
2. Runs deterministic preprocessing: exact screenshot deduplication (SHA-256), interaction aggregation,
   noise filtering, and activity episode grouping.
3. Computes the 5 cognitive features: Recency (decay), Frequency, Retrieval History (audit log),
   Semantic Relevance (TF-IDF session centroid), and Relationship Strength (co-occurrence graph).
4. Computes baseline DMCLM Importance Scores (strictly preserved for research benchmarking).
5. Constructs genuine non-circular training targets from observed future user behavior in subsequent sessions.
6. Generates research training dataset (data/training_dataset.csv) with zero leakage and chronological splits.
7. Trains local interpretable Machine Learning models (Phase 10 Random Forest Classifier & Regressor).
8. Applies trained DMCLM ML model to generate predictions across all memory records.
9. Synthesizes genuine Long-Term Memory (LTM) nodes from consolidated episodic bursts using DMCLM predictions.
10. Evaluates retention dynamics and forgetting eligibility using DMCLM predictions.
11. Executes scientific Baseline vs Trained DMCLM comparison (Phase 11).
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dmclm_config import DEFAULT_CONFIG
    from preprocessing_pipeline import PreprocessingPipeline
    from dmclm_features import DMCLMFeaturePipeline
    from dmclm_baseline import BaselineDecisionEngine
    from phase9_training_dataset import Phase9DatasetGenerator, DATASET_CSV_PATH
    from phase10_ml_training import Phase10ModelTrainer, Phase10MLPredictor
    from phase6_consolidation_compression import Phase6Consolidator
    from phase7_adaptive_retention_forgetting import Phase7AdaptiveRetentionEngine
    from phase11_comparison import Phase11BaselineMLComparator
except ImportError:
    from src.dmclm_config import DEFAULT_CONFIG
    from src.preprocessing_pipeline import PreprocessingPipeline
    from src.dmclm_features import DMCLMFeaturePipeline
    from src.dmclm_baseline import BaselineDecisionEngine
    from src.phase9_training_dataset import Phase9DatasetGenerator, DATASET_CSV_PATH
    from src.phase10_ml_training import Phase10ModelTrainer, Phase10MLPredictor
    from src.phase6_consolidation_compression import Phase6Consolidator
    from src.phase7_adaptive_retention_forgetting import Phase7AdaptiveRetentionEngine
    from src.phase11_comparison import Phase11BaselineMLComparator

logger = logging.getLogger("DMCLM.Orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_full_pipeline():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("STARTING DMCLM COGNITIVE PIPELINE EXECUTION (TRAINED ML PIPELINE)")
    logger.info("=================================================================")

    # STAGE 1 & 2: Preprocessing & Episodic Grouping
    logger.info(">> STEP 1: Running Preprocessing & Activity Episode Grouping...")
    prep_pipeline = PreprocessingPipeline(config=DEFAULT_CONFIG.preprocessing)
    prep_results = prep_pipeline.run_pipeline()
    candidates = prep_results["candidates"]
    prep_report = prep_results["report"]

    logger.info(
        f">> Formed {len(candidates):,} normalized episodic memories from "
        f"{prep_report['raw_counts']['total_raw_observations']:,} raw Collector observations."
    )

    # STAGE 3: Five Cognitive Features Extraction
    logger.info(">> STEP 2: Computing 5 DMCLM Feature Signals (R, F, H, S, Rel)...")
    feature_pipeline = DMCLMFeaturePipeline(config=DEFAULT_CONFIG.features)
    featured_records = feature_pipeline.extract_features(candidates)
    logger.info(">> Feature signals successfully computed across all records.")

    # STAGE 4: Baseline DMCLM Scoring (Preserved strictly for comparison)
    logger.info(">> STEP 3: Evaluating Baseline Importance & Lifecycle Scores (Comparison Benchmark)...")
    decision_engine = BaselineDecisionEngine(config=DEFAULT_CONFIG)
    baseline_records = decision_engine.compute_importance_scores(featured_records)

    # STAGE 5: Training Dataset Generation from Observed Future Behavior
    logger.info(">> STEP 4: Generating Real Behavioral Training Dataset (Phase 9)...")
    dataset_generator = Phase9DatasetGenerator(config=DEFAULT_CONFIG.dataset)
    dataset_rows, consolidated_rows, manifest = dataset_generator.run_pipeline()
    logger.info(f">> Training dataset generated: {len(dataset_rows):,} rows (Target dist: {manifest.target_distribution})")

    # STAGE 6: Local Machine Learning Model Training (Phase 10)
    logger.info(">> STEP 5: Training DMCLM Machine Learning Model (Phase 10 Random Forest)...")
    ml_trainer = Phase10ModelTrainer(config=DEFAULT_CONFIG.model)
    clf_pipeline, reg_pipeline, metadata = ml_trainer.train_from_csv(DATASET_CSV_PATH)
    logger.info(
        f">> Model trained: Val Acc={metadata.val_metrics.get('accuracy', 0.0):.4f}, "
        f"Test Acc={metadata.test_metrics.get('accuracy', 0.0):.4f}, Test ROC-AUC={metadata.test_metrics.get('roc_auc', 0.0):.4f}"
    )

    # STAGE 7: Memory Consolidation using Trained DMCLM Predictions
    logger.info(">> STEP 6: Executing Memory Consolidation & Compression (Phase 6)...")
    consolidator = Phase6Consolidator(config=DEFAULT_CONFIG.consolidation)
    c_nodes, comp_metrics = consolidator.run_pipeline()
    logger.info(f">> Synthesized {len(c_nodes):,} consolidated LTM nodes ({comp_metrics.get('compression_ratio_pct', 0.0)}% reduction).")

    # STAGE 8: Adaptive Retention & Forgetting
    logger.info(">> STEP 7: Evaluating Adaptive Retention & Forgetting (Phase 7)...")
    retention_engine = Phase7AdaptiveRetentionEngine(config=DEFAULT_CONFIG.retention)
    r_nodes, r_metrics = retention_engine.run_pipeline()
    logger.info(f">> Retention evaluated across {len(r_nodes):,} memories: {r_metrics.get('state_counts', {})}")

    # STAGE 9: Scientific Baseline vs ML Model Comparison (Phase 11)
    logger.info(">> STEP 8: Running Scientific Comparison (Baseline vs Trained DMCLM)...")
    comparator = Phase11BaselineMLComparator(dataset_path=DATASET_CSV_PATH)
    comp_report = comparator.run_pipeline(save_artifacts=True, save_to_db=True)
    logger.info(
        f">> Comparison complete on unseen test split (N={comp_report.sample_count}): "
        f"Baseline Acc={comp_report.baseline_classification_metrics['accuracy']:.4f} vs "
        f"Trained DMCLM Acc={comp_report.ml_classification_metrics['accuracy']:.4f}"
    )

    elapsed = round(time.time() - start_time, 2)
    logger.info("=================================================================")
    logger.info(f"DMCLM TRAINED ML PIPELINE COMPLETED in {elapsed}s.")
    logger.info(f"Total Normalized Memories:   {len(featured_records):,}")
    logger.info(f"Training Dataset Samples:    {manifest.total_valid_examples:,}")
    logger.info(f"Trained Model Test Accuracy: {metadata.test_metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"Baseline Test Accuracy:      {comp_report.baseline_classification_metrics['accuracy']:.4f}")
    logger.info(f"Consolidated LTM Nodes:      {len(c_nodes):,}")
    logger.info("=================================================================")

    return {
        "status": "DMCLM machine learning pipeline execution completed.",
        "execution_time_sec": elapsed,
        "memory_count": len(featured_records),
        "dataset_samples": manifest.total_valid_examples,
        "ml_test_accuracy": metadata.test_metrics.get("accuracy", 0.0),
        "baseline_test_accuracy": comp_report.baseline_classification_metrics["accuracy"],
        "ltm_count": len(c_nodes),
    }


if __name__ == "__main__":
    run_full_pipeline()
