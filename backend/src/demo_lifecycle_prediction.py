import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase10_ml_training import Phase10MLPredictor

ROOT = Path(__file__).resolve().parents[2]

predictor = Phase10MLPredictor(
    classifier_path=ROOT / "models" / "dmclm_model_20features.joblib",
    metadata_path=ROOT / "models" / "dmclm_model_20features_metadata.json",
    regressor_path=ROOT / "models" / "nonexistent_regressor.joblib",
)

sample = {
    "memory_id": "viva-demo-001",
    "feature_recency": 0.8,
    "feature_frequency": 0.6,
    "feature_retrieval_history": 0.4,
    "feature_semantic_relevance": 0.9,
    "feature_relationship_strength": 0.5,
    "duration_seconds": 120,
    "interaction_count": 8,
    "browser_event_count": 3,
    "screenshot_count": 1,
    "has_browser_activity": 1,
    "has_window_context": 1,
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_consolidated": 0,
    "cluster_member_count": 1,
    "graph_degree": 2,
    "graph_avg_rel_strength": 0.5,
    "graph_max_rel_strength": 0.7,
    "is_graph_isolated": 0,
}

result = predictor.predict_single(sample)
states = ["RETAIN", "COMPRESS", "ARCHIVE", "FORGET"]
descriptions = {
    "RETAIN": "Keep useful memory",
    "COMPRESS": "Consolidate related memories",
    "ARCHIVE": "Long-term storage candidate",
    "FORGET": "Forgetting candidate",
}

print("\n" + "=" * 72)
print("            DMCLM - MEMORY LIFECYCLE PREDICTION")
print("=" * 72)
print(f"Model version : {result.model_version}")
print(f"Features used : {len(predictor.feature_names)}")
print("-" * 72)
print(f"{'LIFECYCLE':<14} {'PURPOSE':<34} {'PROBABILITY':>12}")
print("-" * 72)

for state in states:
    probability = result.class_probabilities.get(state, 0.0)
    print(f"{state:<14} {descriptions[state]:<34} {probability:>11.2%}")

print("-" * 72)
print("SAMPLE MEMORY RESULT")
print("-" * 72)
print(f"Memory ID          : {sample['memory_id']}")
print(f"Predicted lifecycle: {result.predicted_lifecycle_state}")
print(f"Confidence         : {result.confidence:.2%}")

metrics = predictor.metadata.get("val_metrics", {})
if "accuracy" in metrics:
    print(f"Validation accuracy: {metrics['accuracy']:.2%}")
if "f1_macro" in metrics:
    print(f"Validation macro F1: {metrics['f1_macro']:.2%}")

print("=" * 72)
print("Demo only: prediction performed; no memory was modified.")
