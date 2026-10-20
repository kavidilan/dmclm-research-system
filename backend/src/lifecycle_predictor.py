"""
DMCLM Lifecycle Prediction Engine
Loads the trained Random Forest model and predicts the cognitive lifecycle state
for individual memories or batches of memory records.
Provides:
- lifecycle state: RETAIN | COMPRESS | ARCHIVE | FORGET
- real model confidence (predict_proba)
- feature-grounded natural language explanation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "dmclm_model.joblib"
MEMORY_JSON_PATH = DATA_DIR / "memory_records.json"
PREDICTIONS_JSON_PATH = DATA_DIR / "memory_predictions.json"

FEATURE_NAMES = [
    "recency",
    "frequency",
    "retrieval_history",
    "semantic_relevance",
    "relationship_strength",
]


class LifecyclePredictor:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DMCLM model not found at {self.model_path}. Please run src/train_dmclm.py first."
            )
        self.model = joblib.load(self.model_path)
        self.classes = list(self.model.classes_)

    def predict_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts lifecycle state for a single memory dictionary.
        Extracts the 5 features, runs inference, and returns prediction with confidence.
        """
        features = [
            float(memory.get("recency", 0.0)),
            float(memory.get("frequency", 0.0)),
            float(memory.get("retrieval_history", 0.0)),
            float(memory.get("semantic_relevance", 0.0)),
            float(memory.get("relationship_strength", 0.0)),
        ]

        feat_arr = np.array([features], dtype=np.float64)
        pred_label = self.model.predict(feat_arr)[0]
        probabilities = self.model.predict_proba(feat_arr)[0]

        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probabilities)
        }
        confidence = round(float(max(probabilities)), 4)

        explanation = self._generate_explanation(pred_label, memory, prob_dict)

        return {
            "memory_id": str(memory.get("memory_id", "")),
            "lifecycle": pred_label,
            "confidence": confidence,
            "class_probabilities": prob_dict,
            "explanation": explanation,
            "features_used": dict(zip(FEATURE_NAMES, features))
        }

    def predict_batch(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch predicts lifecycle states across a list of memory dictionaries."""
        if not memories:
            return []

        import pandas as pd
        X_df = pd.DataFrame(
            [
                [
                    float(m.get("recency", 0.0)),
                    float(m.get("frequency", 0.0)),
                    float(m.get("retrieval_history", 0.0)),
                    float(m.get("semantic_relevance", 0.0)),
                    float(m.get("relationship_strength", 0.0)),
                ]
                for m in memories
            ],
            columns=FEATURE_NAMES
        )

        preds = self.model.predict(X_df)
        probas = self.model.predict_proba(X_df)

        results = []
        for i, m in enumerate(memories):
            pred_label = preds[i]
            probs = probas[i]
            prob_dict = {
                cls_name: round(float(p), 4)
                for cls_name, p in zip(self.classes, probs)
            }
            conf = round(float(max(probs)), 4)
            exp = self._generate_explanation(pred_label, m, prob_dict)

            results.append({
                "memory_id": str(m.get("memory_id", "")),
                "lifecycle": pred_label,
                "confidence": conf,
                "class_probabilities": prob_dict,
                "explanation": exp,
                "features_used": {
                    "recency": float(m.get("recency", 0.0)),
                    "frequency": float(m.get("frequency", 0.0)),
                    "retrieval_history": float(m.get("retrieval_history", 0.0)),
                    "semantic_relevance": float(m.get("semantic_relevance", 0.0)),
                    "relationship_strength": float(m.get("relationship_strength", 0.0)),
                }
            })

        return results

    def _generate_explanation(
        self,
        lifecycle: str,
        memory: Dict[str, Any],
        prob_dict: Dict[str, float]
    ) -> str:
        """Constructs an interpretable explanation based on feature values."""
        r = float(memory.get("recency", 0.0))
        f = float(memory.get("frequency", 0.0))
        h = float(memory.get("retrieval_history", 0.0))
        s = float(memory.get("semantic_relevance", 0.0))
        rel = float(memory.get("relationship_strength", 0.0))

        if lifecycle == "RETAIN":
            if h > 0:
                return f"Retained due to active cognitive retrieval history (H={h:.2f}) and sustained relevance."
            return f"Retained in working tier due to high temporal recency (R={r:.2f}) and significant task relevance (S={s:.2f})."
        elif lifecycle == "COMPRESS":
            return f"Designated for consolidation abstraction due to repetitive engagement (F={f:.2f}) and dense relational continuity (Rel={rel:.2f})."
        elif lifecycle == "ARCHIVE":
            return f"Migrated to secondary archive due to age decay (R={r:.2f}) with moderate preserved context."
        elif lifecycle == "FORGET":
            return f"Pruned as transient background noise due to low semantic relevance (S={s:.2f}) and sub-threshold task engagement (F={f:.2f})."
        return "Classified based on multi-dimensional DMCLM decision boundary."


def run_batch_predictions() -> List[Dict[str, Any]]:
    print("\n=== RUNNING DMCLM BATCH PREDICTIONS ===")
    if not MEMORY_JSON_PATH.exists():
        raise FileNotFoundError(f"Memory records not found at {MEMORY_JSON_PATH}")

    with open(MEMORY_JSON_PATH, "r", encoding="utf-8") as f:
        memories = json.load(f)

    predictor = LifecyclePredictor()
    predictions = predictor.predict_batch(memories)

    # Attach predictions into memory_records.json
    pred_map = {p["memory_id"]: p for p in predictions}
    for m in memories:
        p = pred_map.get(str(m.get("memory_id")))
        if p:
            m["lifecycle_prediction"] = p["lifecycle"]
            m["lifecycle_confidence"] = p["confidence"]
            m["lifecycle_explanation"] = p["explanation"]

    with open(MEMORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

    with open(PREDICTIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # Summary
    lifecycle_counts: Dict[str, int] = {}
    for p in predictions:
        lc = p["lifecycle"]
        lifecycle_counts[lc] = lifecycle_counts.get(lc, 0) + 1

    print(f"Generated predictions for {len(predictions)} memory records.")
    print("Predicted Lifecycle Distribution:")
    for lc, cnt in sorted(lifecycle_counts.items()):
        pct = (cnt / len(predictions)) * 100
        print(f"  - {lc:10}: {cnt:5d} ({pct:.2f}%)")

    return predictions


if __name__ == "__main__":
    run_batch_predictions()
