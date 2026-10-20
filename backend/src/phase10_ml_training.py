"""
Phase 10 — ML Model Training and Prediction Engine
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles & Architecture:
1. GENUINE LOCAL ML TRAINING:
   Trains interpretable, lightweight machine-learning models locally on Windows using
   scikit-learn and joblib. No pretrained models, no Hugging Face downloads, no fake predictions.
   Operates strictly on the real Phase 9 dataset partitions (data/training_dataset.csv, N=6,229).

2. ZERO DATA LEAKAGE ENFORCEMENT:
   Fits preprocessing and models strictly on the Phase 9 'train' split (N=4,360).
   The validation split (N=934) is used for hyperparameter evaluation, and the test split (N=935)
   is reserved exclusively for final evaluation.
   Strictly excludes formula scores (baseline_score, final_retention_strength) and future outcomes
   from the input feature matrix.

3. CENTRAL ROLE OF THE 5 DMCLM FEATURES:
   Recency, Frequency, Retrieval History, Semantic Relevance, and Relationship Strength
   form the core cognitive input signals, optionally augmented with validated context signals.

4. REPRODUCIBILITY & SERIALIZATION:
   Controlled with fixed random seed (default=42). Preprocessing transformations and estimators
   are packaged into a unified Pipeline and serialized to models/dmclm_lifecycle_model.joblib,
   with full provenance in models/model_metadata.json and SQLite.

5. NON-DESTRUCTIVE INTEGRATION:
   Exposes predictions via Phase10MLPredictor without modifying or overwriting Phase 5 baseline scores.
   Prepares clean ML predictions for subsequent Phase 11 baseline comparison.

6. MODEL EXPLAINABILITY:
   Calculates and persists Gini feature importances across cognitive features, mapping the
   exact relative contributions of cognitive signals to predictions.
"""

import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import (
        ModelTrainingConfig,
        DMCLMGlobalConfig,
        DEFAULT_CONFIG,
    )
    from src.models import (
        MLPredictionResult,
        ModelTrainingMetadata,
    )
    from src.dmclm_db import (
        init_dmclm_database,
        insert_or_replace_model_run,
        get_latest_model_run,
    )
except ImportError:
    from dmclm_config import (
        ModelTrainingConfig,
        DMCLMGlobalConfig,
        DEFAULT_CONFIG,
    )
    from models import (
        MLPredictionResult,
        ModelTrainingMetadata,
    )
    from dmclm_db import (
        init_dmclm_database,
        insert_or_replace_model_run,
        get_latest_model_run,
    )

# Input dataset paths
TRAINING_CSV_PATH = DATA_DIR / "training_dataset.csv"
DATASET_SCHEMA_PATH = DATA_DIR / "dataset_schema.json"
DATASET_MANIFEST_PATH = RESULTS_DIR / "dataset_manifest.json"

# Output model paths
MODEL_SAVE_PATH = MODELS_DIR / "dmclm_model.joblib"
LIFECYCLE_MODEL_PATH = MODELS_DIR / "dmclm_lifecycle_model.joblib"
REGRESSOR_MODEL_PATH = MODELS_DIR / "dmclm_importance_regressor.joblib"
METADATA_SAVE_PATH = MODELS_DIR / "model_metadata.json"
METRICS_SAVE_PATH = RESULTS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase10.MLTraining")

# Canonical feature lists
CORE_FIVE_FEATURES = [
    "feature_recency",
    "feature_frequency",
    "feature_retrieval_history",
    "feature_semantic_relevance",
    "feature_relationship_strength",
]

# Legacy key aliases mapping input memory keys to canonical Phase 9 feature names
FEATURE_KEY_ALIASES = {
    "recency": "feature_recency",
    "recency_score": "feature_recency",
    "frequency": "feature_frequency",
    "frequency_score": "feature_frequency",
    "retrieval_history": "feature_retrieval_history",
    "retrieval_history_score": "feature_retrieval_history",
    "semantic_relevance": "feature_semantic_relevance",
    "relationship_strength": "feature_relationship_strength",
}


class Phase10ModelTrainer:
    """
    Core implementation of Phase 10 — ML Model Training.
    Trains reproducible scikit-learn models on Phase 9 data with zero leakage.
    """

    def __init__(self, config: Optional[ModelTrainingConfig] = None):
        self.config = config or DEFAULT_CONFIG.model
        self.config.validate()

    def get_feature_list(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Determines the exact feature columns to use for training based on configuration.
        """
        if self.config.feature_set == "core_five":
            return list(CORE_FIVE_FEATURES)
        elif self.config.feature_set == "enriched":
            if df is not None:
                context_features = [
                    "duration_seconds", "interaction_count", "browser_event_count",
                    "screenshot_count", "has_browser_activity", "has_window_context",
                    "hour_of_day", "day_of_week", "is_weekend", "is_consolidated",
                    "cluster_member_count", "graph_degree", "graph_avg_rel_strength",
                    "graph_max_rel_strength", "is_graph_isolated"
                ]
                present_context = [c for c in context_features if c in df.columns]
                return list(CORE_FIVE_FEATURES) + present_context
            return list(CORE_FIVE_FEATURES)
        return list(CORE_FIVE_FEATURES)

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        """
        Validates that the dataset conforms to Phase 9 schema, contains required partitions,
        and enforces zero data leakage.
        """
        required_cols = {"memory_id", "split", self.config.target_column}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        splits = set(df["split"].unique())
        if not {"train", "val", "test"}.issubset(splits):
            raise ValueError(f"Dataset must contain 'train', 'val', and 'test' splits, got: {splits}")

        feature_cols = self.get_feature_list(df)
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Dataset missing configured feature columns: {missing_features}")

        # Data Leakage Protection Check
        forbidden_in_features = {
            "baseline_score", "final_retention_strength", "value_tier",
            "target_lifecycle_state", "target_importance_score", "target_consolidation_priority",
            "target_retention_state", "is_protected", "forgetting_eligible",
        }
        leakage = set(feature_cols) & forbidden_in_features
        if leakage:
            raise ValueError(f"DATA LEAKAGE DETECTED: Features must not contain target/audit columns: {leakage}")

        # Check for NaN values in feature columns
        nan_counts = df[feature_cols].isna().sum()
        if nan_counts.any():
            bad_cols = nan_counts[nan_counts > 0].to_dict()
            logger.warning(f"[PHASE 10] Handling missing values in feature columns: {bad_cols}")
            df[feature_cols] = df[feature_cols].fillna(0.0)

        dataset_hash = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
        return feature_cols, dataset_hash

    def build_classifier_pipeline(self) -> Pipeline:
        """Constructs scikit-learn classifier pipeline."""
        c = self.config
        if c.algorithm == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=c.n_estimators,
                max_depth=c.max_depth,
                min_samples_split=c.min_samples_split,
                min_samples_leaf=c.min_samples_leaf,
                class_weight=c.class_weight,
                random_state=c.random_seed,
                n_jobs=-1,
            )
        elif c.algorithm == "gradient_boosting":
            clf = GradientBoostingClassifier(
                n_estimators=c.n_estimators,
                max_depth=min(c.max_depth, 4),
                random_state=c.random_seed,
            )
        elif c.algorithm == "logistic_regression":
            clf = LogisticRegression(
                class_weight=c.class_weight,
                random_state=c.random_seed,
                max_iter=1000,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {c.algorithm}")

        steps = []
        if c.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", clf))
        return Pipeline(steps)

    def build_regressor_pipeline(self) -> Pipeline:
        """Constructs scikit-learn regressor pipeline."""
        c = self.config
        reg = RandomForestRegressor(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            min_samples_split=c.min_samples_split,
            min_samples_leaf=c.min_samples_leaf,
            random_state=c.random_seed,
            n_jobs=-1,
        )
        steps = []
        if c.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("regressor", reg))
        return Pipeline(steps)

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculates comprehensive classification evaluation metrics."""
        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        prec_macro = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        rec_macro = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        f1_mac = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        f1_wt = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)

        cm = confusion_matrix(y_true, y_pred, labels=classes).tolist() if classes else []

        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt,
            "confusion_matrix": cm,
            "classes": classes or [],
            "sample_count": int(len(y_true)),
        }

        # Calculate ROC-AUC if probabilities available
        if y_prob is not None and len(classes or []) >= 2:
            try:
                if len(classes) == 2:
                    pos_idx = 0 if classes[0] == "COMPRESS" else 1
                    y_bin = (y_true == classes[pos_idx]).astype(int)
                    roc = round(float(roc_auc_score(y_bin, y_prob[:, pos_idx])), 4)
                    metrics["roc_auc"] = roc
                else:
                    roc = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=classes)), 4)
                    metrics["roc_auc"] = roc
            except Exception:
                pass

        return metrics

    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates regression metrics (MAE, RMSE, R²)."""
        mae = round(float(mean_absolute_error(y_true, y_pred)), 4)
        rmse = round(float(root_mean_squared_error(y_true, y_pred)), 4)
        r2 = round(float(r2_score(y_true, y_pred)), 4)
        return {
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "sample_count": int(len(y_true)),
        }

    def train_and_evaluate(
        self,
        df: pd.DataFrame
    ) -> Tuple[Pipeline, Optional[Pipeline], ModelTrainingMetadata]:
        """
        Executes reproducible training, validation, and test evaluation.
        """
        start_time = time.time()
        feature_cols, dataset_hash = self.validate_dataset(df)

        logger.info(
            f"[PHASE 10] Training {self.config.algorithm} model with {len(feature_cols)} features: "
            f"{feature_cols}"
        )

        # Strictly partition subsets
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df[self.config.target_column].values

        X_val = val_df[feature_cols].values
        y_val = val_df[self.config.target_column].values

        X_test = test_df[feature_cols].values
        y_test = test_df[self.config.target_column].values

        # 1. Train Classifier Pipeline strictly on Train split
        clf_pipeline = self.build_classifier_pipeline()
        clf_pipeline.fit(X_train, y_train)

        # Extract fitted estimator and classes
        fitted_clf = clf_pipeline.named_steps["classifier"]
        classes = [str(c) for c in fitted_clf.classes_]

        # In-sample Train Metrics
        train_pred = clf_pipeline.predict(X_train)
        train_prob = clf_pipeline.predict_proba(X_train) if hasattr(clf_pipeline, "predict_proba") else None
        train_metrics = self.calculate_classification_metrics(y_train, train_pred, train_prob, classes)

        # Validation Metrics (Model selection / validation)
        val_pred = clf_pipeline.predict(X_val)
        val_prob = clf_pipeline.predict_proba(X_val) if hasattr(clf_pipeline, "predict_proba") else None
        val_metrics = self.calculate_classification_metrics(y_val, val_pred, val_prob, classes)

        # Final Test Metrics (Generalization check)
        test_pred = clf_pipeline.predict(X_test)
        test_prob = clf_pipeline.predict_proba(X_test) if hasattr(clf_pipeline, "predict_proba") else None
        test_metrics = self.calculate_classification_metrics(y_test, test_pred, test_prob, classes)

        # 2. Extract Feature Importances
        feature_importance: Dict[str, float] = {}
        if hasattr(fitted_clf, "feature_importances_"):
            importances = fitted_clf.feature_importances_
            feature_importance = {
                f: round(float(imp), 4) for f, imp in zip(feature_cols, importances)
            }
        elif hasattr(fitted_clf, "coef_"):
            # Logistic Regression coefficients
            coefs = np.abs(fitted_clf.coef_[0])
            norm_coefs = coefs / np.sum(coefs) if np.sum(coefs) > 0 else coefs
            feature_importance = {
                f: round(float(imp), 4) for f, imp in zip(feature_cols, norm_coefs)
            }
        # Sort by importance descending
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: -x[1]))

        # 3. Train Secondary Continuous Regressor on target_importance_score (if present)
        reg_pipeline: Optional[Pipeline] = None
        if self.config.train_regressor and self.config.regression_target_column in df.columns:
            logger.info("[PHASE 10] Training secondary continuous importance score regressor...")
            y_train_reg = train_df[self.config.regression_target_column].values
            y_val_reg = val_df[self.config.regression_target_column].values
            y_test_reg = test_df[self.config.regression_target_column].values

            reg_pipeline = self.build_regressor_pipeline()
            reg_pipeline.fit(X_train, y_train_reg)

            val_reg_pred = reg_pipeline.predict(X_val)
            test_reg_pred = reg_pipeline.predict(X_test)

            val_metrics["regression"] = self.calculate_regression_metrics(y_val_reg, val_reg_pred)
            test_metrics["regression"] = self.calculate_regression_metrics(y_test_reg, test_reg_pred)

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = ModelTrainingMetadata(
            model_version="v1.0",
            training_timestamp=now_iso,
            algorithm=f"{self.config.algorithm.capitalize()}",
            hyperparameters=self.config.to_dict(),
            feature_names=feature_cols,
            target_column=self.config.target_column,
            dataset_hash=dataset_hash,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            leakage_protected=True,
            model_artifact_path=str(LIFECYCLE_MODEL_PATH.relative_to(ROOT_DIR)),
        )

        elapsed = round(time.time() - start_time, 4)
        logger.info(
            f"[PHASE 10] Model training complete in {elapsed}s: "
            f"Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,} | "
            f"Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1_macro']:.4f} | "
            f"Test Acc={test_metrics['accuracy']:.4f}, Test F1={test_metrics['f1_macro']:.4f}"
        )

        return clf_pipeline, reg_pipeline, metadata

    def train_from_csv(
        self,
        dataset_path: Optional[Path] = None,
        save_artifacts: bool = True,
        save_to_db: bool = True,
    ) -> Tuple[Pipeline, Optional[Pipeline], ModelTrainingMetadata]:
        """Alias for run_pipeline for backwards compatibility with training scripts."""
        return self.run_pipeline(dataset_path=dataset_path, save_artifacts=save_artifacts, save_to_db=save_to_db)

    def run_pipeline(
        self,
        dataset_path: Optional[Path] = None,
        save_artifacts: bool = True,
        save_to_db: bool = True,
    ) -> Tuple[Pipeline, Optional[Pipeline], ModelTrainingMetadata]:
        """
        Executes full Phase 10 model training workflow from persistent dataset and saves artifacts.
        """
        csv_path = dataset_path or TRAINING_CSV_PATH
        if not csv_path.exists():
            raise FileNotFoundError(f"Phase 9 dataset not found at {csv_path}. Run Phase 9 first.")

        df = pd.read_csv(csv_path)
        logger.info(f"[PHASE 10] Loaded {len(df):,} examples from {csv_path}")

        clf_pipeline, reg_pipeline, metadata = self.train_and_evaluate(df)

        if save_artifacts:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            # 1. Save Primary Lifecycle Classifier
            joblib.dump(clf_pipeline, LIFECYCLE_MODEL_PATH)
            # Duplicate to MODEL_SAVE_PATH for backward compatibility with existing project scripts
            joblib.dump(clf_pipeline, MODEL_SAVE_PATH)
            logger.info(f"[PHASE 10] Saved classifier pipeline to {LIFECYCLE_MODEL_PATH}")

            # 2. Save Secondary Continuous Regressor (if trained)
            if reg_pipeline is not None:
                joblib.dump(reg_pipeline, REGRESSOR_MODEL_PATH)
                logger.info(f"[PHASE 10] Saved regressor pipeline to {REGRESSOR_MODEL_PATH}")

            # 3. Save Model Metadata
            with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            logger.info(f"[PHASE 10] Saved model metadata to {METADATA_SAVE_PATH}")

            # 4. Save Model Metrics
            metrics_payload = {
                "model_version": metadata.model_version,
                "training_timestamp": metadata.training_timestamp,
                "algorithm": metadata.algorithm,
                "val_metrics": metadata.val_metrics,
                "test_metrics": metadata.test_metrics,
                "feature_importance": metadata.feature_importance,
            }
            with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)
            logger.info(f"[PHASE 10] Saved metrics report to {METRICS_SAVE_PATH}")

            # 5. Save Feature Importance Report
            with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata.feature_importance, f, indent=2)
            logger.info(f"[PHASE 10] Saved feature importance to {FEATURE_IMPORTANCE_PATH}")

        if save_to_db:
            init_dmclm_database()
            insert_or_replace_model_run(metadata.to_dict())
            logger.info(f"[PHASE 10] Persisted model run metadata to SQLite.")

        return clf_pipeline, reg_pipeline, metadata


class Phase10MLPredictor:
    """
    Reusable prediction and inference engine for DMCLM Phase 10.
    Accepts real memory feature dictionaries and produces calibrated ML predictions
    without altering or overwriting baseline scores.
    """

    def __init__(
        self,
        classifier_path: Optional[Path] = None,
        regressor_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.clf_path = classifier_path or (LIFECYCLE_MODEL_PATH if LIFECYCLE_MODEL_PATH.exists() else MODEL_SAVE_PATH)
        self.reg_path = regressor_path or REGRESSOR_MODEL_PATH
        self.meta_path = metadata_path or METADATA_SAVE_PATH

        if not self.clf_path.exists():
            raise FileNotFoundError(
                f"Trained DMCLM classifier model not found at {self.clf_path}. "
                f"Please run Phase 10 model training first."
            )

        self.classifier: Pipeline = joblib.load(self.clf_path)
        self.regressor: Optional[Pipeline] = joblib.load(self.reg_path) if self.reg_path.exists() else None

        # Load metadata if available
        self.metadata: Dict[str, Any] = {}
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                pass

        self.feature_names: List[str] = self.metadata.get("feature_names", list(CORE_FIVE_FEATURES))
        self.feature_importance: Dict[str, float] = self.metadata.get("feature_importance", {})

    def extract_features(self, memory: Dict[str, Any]) -> Tuple[List[float], Dict[str, float]]:
        """
        Extracts and normalizes features from input dictionary, safely mapping
        legacy keys ('recency') to canonical Phase 9 names ('feature_recency').
        """
        feature_vals: List[float] = []
        feature_dict: Dict[str, float] = {}

        for fname in self.feature_names:
            val = None
            # Check canonical name
            if fname in memory and memory[fname] is not None:
                val = memory[fname]
            else:
                # Check legacy aliases
                for legacy_k, canon_k in FEATURE_KEY_ALIASES.items():
                    if canon_k == fname and legacy_k in memory and memory[legacy_k] is not None:
                        val = memory[legacy_k]
                        break

            if val is None:
                # Default safe imputation for missing optional features
                val = 0.0

            try:
                f_val = float(val)
                if math.isnan(f_val):
                    f_val = 0.0
            except (ValueError, TypeError):
                f_val = 0.0

            feature_vals.append(f_val)
            feature_dict[fname] = f_val

        return feature_vals, feature_dict

    def predict_single(self, memory: Dict[str, Any]) -> MLPredictionResult:
        """
        Executes inference for a single memory record.
        """
        mid = str(memory.get("memory_id") or memory.get("candidate_id") or memory.get("consolidated_memory_id") or "mem_unknown")
        feature_vals, feature_dict = self.extract_features(memory)

        X = pd.DataFrame([feature_dict], columns=self.feature_names)
        pred_label = str(self.classifier.predict(X)[0])

        # Prediction probability / confidence
        confidence = 1.0
        class_probs: Dict[str, float] = {pred_label: 1.0}
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[0]
            classes = [str(c) for c in self.classifier.named_steps["classifier"].classes_]
            class_probs = {c: round(float(p), 4) for c, p in zip(classes, probs)}
            confidence = round(float(max(probs)), 4)

        # Regressor score (if regressor loaded)
        pred_importance: Optional[float] = None
        if self.regressor is not None:
            try:
                reg_score = float(self.regressor.predict(X)[0])
                pred_importance = round(max(0.0, min(1.0, reg_score)), 4)
            except Exception:
                pass

        # Explainability: Local importance contribution
        feature_contribs: Dict[str, float] = {}
        for f, val in feature_dict.items():
            global_imp = self.feature_importance.get(f, 0.20)
            contrib = round(val * global_imp, 4)
            feature_contribs[f] = contrib

        dominant_feat = max(feature_contribs.items(), key=lambda x: x[1])[0] if feature_contribs else None

        return MLPredictionResult(
            memory_id=mid,
            predicted_lifecycle_state=pred_label,
            prediction_confidence=confidence,
            class_probabilities=class_probs,
            predicted_importance_score=pred_importance,
            dominant_contributing_feature=dominant_feat,
            feature_contributions=feature_contribs,
            model_version=self.metadata.get("model_version", "v1.0"),
            algorithm=self.metadata.get("algorithm", "RandomForestClassifier"),
            is_ml_generated=True,
        )

    def predict_batch(self, memories: List[Dict[str, Any]]) -> List[MLPredictionResult]:
        """
        Executes vectorized batch inference across multiple memory records for high performance.
        """
        if not memories:
            return []

        feature_matrix = []
        feature_dicts = []
        memory_ids = []
        for m in memories:
            mid = str(m.get("memory_id") or m.get("candidate_id") or m.get("consolidated_memory_id") or "mem_unknown")
            memory_ids.append(mid)
            vals, fdict = self.extract_features(m)
            feature_matrix.append(vals)
            feature_dicts.append(fdict)

        X = np.array(feature_matrix)
        preds = self.classifier.predict(X)

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            classes = [str(c) for c in self.classifier.named_steps["classifier"].classes_]
        else:
            probs = None
            classes = []

        reg_preds = None
        if self.regressor is not None:
            try:
                reg_preds = self.regressor.predict(X)
            except Exception:
                reg_preds = None

        results: List[MLPredictionResult] = []
        for i in range(len(memories)):
            pred_label = str(preds[i])
            if probs is not None:
                prob_row = probs[i]
                class_probs = {c: round(float(p), 4) for c, p in zip(classes, prob_row)}
                conf = round(float(np.max(prob_row)), 4)
            else:
                class_probs = {pred_label: 1.0}
                conf = 1.0

            reg_score = None
            if reg_preds is not None:
                reg_score = round(float(np.clip(reg_preds[i], 0.0, 1.0)), 4)

            fdict = feature_dicts[i]
            contribs: Dict[str, float] = {}
            for f, val in fdict.items():
                contribs[f] = round(val * self.feature_importance.get(f, 0.20), 4)
            dom_f = max(contribs.items(), key=lambda x: x[1])[0] if contribs else None

            results.append(
                MLPredictionResult(
                    memory_id=memory_ids[i],
                    predicted_lifecycle_state=pred_label,
                    prediction_confidence=conf,
                    class_probabilities=class_probs,
                    predicted_importance_score=reg_score,
                    dominant_contributing_feature=dom_f,
                    feature_contributions=contribs,
                    model_version=self.metadata.get("model_version", "v1.0"),
                    algorithm=self.metadata.get("algorithm", "RandomForestClassifier"),
                    is_ml_generated=True,
                )
            )
        return results


if __name__ == "__main__":
    trainer = Phase10ModelTrainer()
    clf, reg, meta = trainer.run_pipeline()

    print("\n[PHASE 10 ML MODEL TRAINING SUMMARY]")
    print(f"Algorithm:                   {meta.algorithm}")
    print(f"Features Count:              {len(meta.feature_names)}")
    print(f"Features:                    {meta.feature_names}")
    print(f"Train Samples (70%):         {meta.train_samples:,}")
    print(f"Validation Samples (15%):    {meta.val_samples:,}")
    print(f"Test Samples (15%):          {meta.test_samples:,}")
    print(f"Validation Accuracy:         {meta.val_metrics['accuracy']:.4f}")
    print(f"Validation F1 (Macro):       {meta.val_metrics['f1_macro']:.4f}")
    print(f"Validation ROC-AUC:          {meta.val_metrics.get('roc_auc', 'N/A')}")
    print(f"Test Accuracy:               {meta.test_metrics['accuracy']:.4f}")
    print(f"Test F1 (Macro):             {meta.test_metrics['f1_macro']:.4f}")
    print(f"Test ROC-AUC:                {meta.test_metrics.get('roc_auc', 'N/A')}")
    print("\nFeature Importances:")
    for f, imp in meta.feature_importance.items():
        print(f"  - {f:<30}: {imp:.4f}")
    print(f"\nModel Artifact Saved:        {LIFECYCLE_MODEL_PATH}")
