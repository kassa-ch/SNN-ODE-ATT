"""
demo2 wrapper

Keeps the same anomaly-detection core by reusing demo1 implementation,
while binding to exp2 semantics:
- abnormal_data_type = original_6
- exp_key = exp2
"""

from config import ACTIVE_MODELS
from demo1 import detect_all_models_unified, detect_experiment_runs, detection_logger, find_available_runs


def main():
    data_dir = "./origin_samples/preprocess_data"
    exp_key = "exp2"
    abnormal_data_type = "original_6"

    available_folds = find_available_runs(exp_key, ACTIVE_MODELS, pattern="fold")
    detection_logger.log("general", f"Detected available folds for {exp_key}: {available_folds}")

    if available_folds:
        detect_experiment_runs(
            data_dir=data_dir,
            exp_key=exp_key,
            abnormal_data_type=abnormal_data_type,
            runs=available_folds,
            pattern="fold",
        )
    else:
        detect_all_models_unified(
            data_dir=data_dir,
            exp_key=exp_key,
            abnormal_data_type=abnormal_data_type,
        )

    detection_logger.log("general", "demo2 completed.")


if __name__ == "__main__":
    main()
