
import os
import time
import json
import pandas as pd
import mlflow
import mlflow.pyfunc
import torch
import transformers
import sklearn
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------
# 1. MLflow Setup
# -------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("hf-sentiment-champion-challenger")

os.makedirs("outputs", exist_ok=True)

# -------------------------------
# 2. Load Evaluation Dataset
# -------------------------------

df = pd.read_csv("data/eval_inputs.csv")

texts = df["text"].astype(str).tolist()
true_labels = df["label"].str.upper().tolist()

# -------------------------------
# 3. Define Models
# -------------------------------

models = {
    "A_distilbert": {
        "id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion_candidate"
    },
    "B_roberta": {
        "id": "siebert/sentiment-roberta-large-english",
        "role": "challenger_candidate"
    }
}

# -------------------------------
# 4. Prediction and Logging
# -------------------------------

all_metrics = {}

class SentimentModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.classifier = pipeline(
            "sentiment-analysis",
            model=context.artifacts["model_dir"],
            device=-1
        )

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        results = self.classifier(texts)
        return pd.DataFrame({
            "text": texts,
            "prediction": [r["label"] for r in results],
            "score": [r["score"] for r in results]
        })

for name, details in models.items():

    print("\nLoading:", name)

    start = time.time()

    classifier = pipeline(
        "sentiment-analysis",
        model=details["id"],
        device=-1
    )

    predictions = classifier(texts)

    total_time = time.time() - start
    avg_latency = (total_time / len(texts)) * 1000

    predicted_labels = [p["label"].upper() for p in predictions]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels,
        pos_label="POSITIVE", zero_division=0
    )
    recall = recall_score(
        true_labels, predicted_labels,
        pos_label="POSITIVE", zero_division=0
    )
    f1 = f1_score(
        true_labels, predicted_labels,
        average="macro", zero_division=0
    )

    all_metrics[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
        "avg_latency_ms": avg_latency,
        "total_inference_s": total_time
    }

    # Save predictions
    prediction_df = pd.DataFrame({
        "text": texts,
        "reference": true_labels,
        "prediction": predicted_labels
    })

    prediction_file = f"outputs/predictions_{name}.csv"
    prediction_df.to_csv(prediction_file, index=False)

    # MLflow run
    with mlflow.start_run(run_name=name):

        mlflow.log_param("hf_model_id", details["id"])
        mlflow.log_param("task", "sentiment-classification")
        mlflow.log_param("role", details["role"])
        mlflow.log_param("eval_samples", len(texts))
        mlflow.log_param("seed", 42)
        mlflow.log_param("device", "cpu")
        mlflow.log_param("max_length", 512)

        mlflow.set_tag("student_usn", "1MS25CI105")
        mlflow.set_tag("dataset_name", "custom_sentiment_dataset")
        mlflow.set_tag(
            "library_versions",
            f"transformers={transformers.__version__}, "
            f"torch={torch.__version__}, "
            f"mlflow={mlflow.__version__}"
        )

        mlflow.log_metrics(all_metrics[name])

        mlflow.log_artifact(prediction_file)
        mlflow.log_artifact("data/eval_inputs.csv")

        # Confusion matrix
        cm = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=["NEGATIVE", "POSITIVE"]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["NEGATIVE", "POSITIVE"]
        )

        disp.plot()
        plt.title(name + " Confusion Matrix")
        plt.savefig(f"outputs/{name}_confusion_matrix.png")
        plt.close()

        mlflow.log_artifact(
            f"outputs/{name}_confusion_matrix.png"
        )

               
        # Download each model to its own local directory
        local_model_path = snapshot_download(
            repo_id=details["id"],
            local_dir=f"models/{name}"
        )

        # Log model in MLflow
        mlflow.pyfunc.log_model(
            name="sentiment_model",
            python_model=SentimentModel(),
            artifacts={"model_dir": local_model_path}
        )


    print(name, "completed")
    print(all_metrics[name])

# -------------------------------
# 5. Compare Models
# -------------------------------

comparison = pd.DataFrame(all_metrics).T
comparison.to_csv("outputs/metrics_summary.csv")

comparison[[
    "accuracy", "precision", "recall", "macro_f1"
]].plot(kind="bar", figsize=(10, 6))

plt.title("Champion vs Challenger Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# -------------------------------
# 6. Champion Decision
# -------------------------------

a = all_metrics["A_distilbert"]
b = all_metrics["B_roberta"]

if b["macro_f1"] >= a["macro_f1"] + 0.01:
    winner = "B_roberta"
else:
    winner = "A_distilbert"

print("\nFINAL CHAMPION:", winner)

with mlflow.start_run(run_name="compare_sentiment"):

    mlflow.log_param("primary_metric", "macro_f1")
    mlflow.log_param("margin", 0.01)

    mlflow.log_metric(
        "delta_primary_metric",
        b["macro_f1"] - a["macro_f1"]
    )

    mlflow.set_tag("winner", winner)

    mlflow.log_artifact("outputs/metrics_summary.csv")
    mlflow.log_artifact("outputs/model_comparison.png")

print("\nALL EXPERIMENTS COMPLETED")