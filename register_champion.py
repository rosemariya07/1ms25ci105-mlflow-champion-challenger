"""Register the two already-logged MLflow models and assign Champion/Challenger aliases.

Expected MLflow run names: A_distilbert and B_roberta.
The script searches each run for an MLflow pyfunc model artifact containing an MLmodel file,
so it does not hard-code run IDs or assume a particular model artifact folder name.
"""
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "hf-sentiment-champion-challenger")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "sentiment-model")
MODEL_A_RUN_NAME = os.getenv("MODEL_A_RUN_NAME", "A_distilbert")
MODEL_B_RUN_NAME = os.getenv("MODEL_B_RUN_NAME", "B_roberta")
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "macro_f1")
MARGIN = float(os.getenv("CHAMPION_MARGIN", "0.01"))


def find_model_artifact(client, run_id, path=""):
    """Return the artifact path containing MLmodel, or None if not found."""
    for item in client.list_artifacts(run_id, path):
        if not item.is_dir and item.path.rsplit("/", 1)[-1] == "MLmodel":
            return item.path.rsplit("/", 1)[0] if "/" in item.path else ""
        if item.is_dir:
            found = find_model_artifact(client, run_id, item.path)
            if found is not None:
                return found
    return None


def latest_run(client, experiment_id, run_name):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    if not runs:
        raise RuntimeError(
            f"Could not find MLflow run named '{run_name}' in experiment '{EXPERIMENT_NAME}'. "
            "Check the run names in the MLflow UI or update MODEL_A_RUN_NAME / MODEL_B_RUN_NAME."
        )
    return runs[0]


def register_run(client, run, role):
    artifact_path = find_model_artifact(client, run.info.run_id)
    if artifact_path is None:
        raise RuntimeError(
            f"No logged MLflow model (MLmodel file) was found in run {run.info.run_id}. "
            "Open the run in MLflow and confirm that evaluate.py logged the model as an artifact."
        )
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}" if artifact_path else f"runs:/{run.info.run_id}"
    print(f"Registering {role}: {model_uri}")
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, result.version, "role", role)
    hf_model_id = run.data.params.get("hf_model_id") or run.data.tags.get("hf_model_id")
    if hf_model_id:
        client.set_model_version_tag(REGISTERED_MODEL_NAME, result.version, "hf_model_id", hf_model_id)
    return result


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' was not found.")

    run_a = latest_run(client, experiment.experiment_id, MODEL_A_RUN_NAME)
    run_b = latest_run(client, experiment.experiment_id, MODEL_B_RUN_NAME)
    score_a = run_a.data.metrics.get(PRIMARY_METRIC)
    score_b = run_b.data.metrics.get(PRIMARY_METRIC)
    if score_a is None or score_b is None:
        raise RuntimeError(
            f"Metric '{PRIMARY_METRIC}' must be logged in both runs. "
            f"Found A={score_a}, B={score_b}. Check the metric key in MLflow."
        )

    latency_a = run_a.data.metrics.get("avg_latency_ms", float("inf"))
    latency_b = run_b.data.metrics.get("avg_latency_ms", float("inf"))
    delta = score_b - score_a
    if delta >= MARGIN:
        winner_role = "B"
    elif delta <= -MARGIN:
        winner_role = "A"
    else:
        winner_role = "A" if latency_a <= latency_b else "B"

    print(f"{PRIMARY_METRIC}: A={score_a:.4f}, B={score_b:.4f}, delta(B-A)={delta:.4f}")
    print(f"Decision: Model {winner_role} wins (margin={MARGIN:.4f}; latency tie-breaker).")

    # Create the registered model if this is the first registration.
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    version_a = register_run(client, run_a, "model_a")
    version_b = register_run(client, run_b, "model_b")

    champion_version = version_a.version if winner_role == "A" else version_b.version
    challenger_version = version_b.version if winner_role == "A" else version_a.version
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", champion_version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", challenger_version)

    # Attach the decision metadata to the registered model versions.
    client.set_model_version_tag(REGISTERED_MODEL_NAME, champion_version, "selection_metric", PRIMARY_METRIC)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, champion_version, "selection_metric_value", str(score_a if winner_role == "A" else score_b))
    print(f"Champion alias -> {REGISTERED_MODEL_NAME} version {champion_version}")
    print(f"Challenger alias -> {REGISTERED_MODEL_NAME} version {challenger_version}")
    print("Registration and aliases completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
