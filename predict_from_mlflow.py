"""Load the Champion through its MLflow registry alias and predict on unseen text."""
import argparse
import os
import sys
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "sentiment-model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/new_data.csv", help="CSV containing a 'text' column")
    parser.add_argument("--output", default="outputs/new_data_predictions.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    data = pd.read_csv(args.input)
    if "text" not in data.columns:
        raise ValueError("Input CSV must contain a column named 'text'.")
    if data.empty:
        raise ValueError("Input CSV contains no examples.")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    run = client.get_run(version.run_id)
    hf_model_id = (
        version.tags.get("hf_model_id")
        or run.data.params.get("hf_model_id")
        or run.data.tags.get("hf_model_id")
        or "Not recorded in MLflow metadata"
    )

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"Loading model URI: {model_uri}")
    print(f"Registered model version: {version.version}")
    print(f"Hugging Face model ID: {hf_model_id}")
    model = mlflow.pyfunc.load_model(model_uri)

    # The registered MLflow model handles inference. This script does not call Hugging Face.
    predictions = model.predict(data[["text"]])
    if isinstance(predictions, pd.DataFrame):
        pred_df = predictions.reset_index(drop=True)
    elif isinstance(predictions, pd.Series):
        pred_df = predictions.to_frame(name="prediction").reset_index(drop=True)
    else:
        pred_df = pd.DataFrame({"prediction": list(predictions)})

    output = pd.concat([data.reset_index(drop=True), pred_df], axis=1)
    output["registered_model"] = REGISTERED_MODEL_NAME
    output["model_version"] = str(version.version)
    output["hf_model_id"] = hf_model_id
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output.to_csv(args.output, index=False)
    print("\nPredictions:")
    print(output.to_string(index=False))
    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
