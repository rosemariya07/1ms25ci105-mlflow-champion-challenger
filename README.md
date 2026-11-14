# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI105  
**Task:** Sentiment classification  
**MLflow experiment:** `hf-sentiment-champion-challenger`

## 1. Task and models

This project compares two pretrained Hugging Face sentiment-classification models on the same custom evaluation set of 10 text examples.

- **Model A (initial Champion):** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english`
- **Dataset:** A custom 10-example sentiment evaluation set. The same examples and reference labels are used for both models and are stored in `data/eval_inputs.csv`.
- **Task:** Binary sentiment classification (positive/negative).

## 2. How to run

Run these commands from the repository root in the same Python environment used for the MLflow experiment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the MLflow tracking server in a separate terminal using the same backend and artifact-store configuration used when the evaluation runs were created. For a local development setup, the assignment gives this example:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000
```

Then run the existing evaluation script (the project file created for the model comparison):

```bash
python evaluate.py
```

Register both logged models and assign the aliases:

```bash
python register_champion.py
```

Predict on the unseen examples in `data/new_data.csv` using the Champion alias:

```bash
python predict_from_mlflow.py
```

The prediction script loads `models:/sentiment-model@champion` through MLflow and saves the result to `outputs/new_data_predictions.csv`. It does not call the Hugging Face pipeline directly.

**Important configuration check:** `register_champion.py` expects the MLflow experiment name `hf-sentiment-champion-challenger` and run names `A_distilbert` and `B_roberta`. If the names in the MLflow UI differ, set `MLFLOW_EXPERIMENT_NAME`, `MODEL_A_RUN_NAME`, or `MODEL_B_RUN_NAME` before running the registration script. The script searches the runs for a logged MLflow model artifact; each run must contain an `MLmodel` file.

## 3. Champion rule

The primary metric is **macro-F1**, for which a higher value is better. The Challenger must improve macro-F1 by at least **0.01 (one percentage point)** to replace the current Champion. If the difference is smaller than the margin, the model with lower average inference latency is preferred. The rule is applied to the recorded evaluation results; the metric and margin are not changed after reviewing the results.

## 4. Results table

The following values are the recorded results from the 10-example evaluation run.

| Metric | Model A: DistilBERT | Model B: RoBERTa |
|---|---:|---:|
| Accuracy | 0.9000 | 1.0000 |
| Precision | 0.8333 | 1.0000 |
| Recall | 1.0000 | 1.0000 |
| Macro-F1 | 0.8990 | 1.0000 |
| Average latency (ms) | 373.35 | 1536.44 |
| Total inference time (s) | 3.73 | 15.36 |

**Macro-F1 difference (Model B − Model A):** 0.1010 (approximately 10.1 percentage points). Model B meets the 0.01 improvement requirement.

## 5. Decision

Under the stated decision rule, Model B (RoBERTa) is selected as the Champion because its macro-F1 score is 1.0000, compared with 0.8990 for Model A (DistilBERT). The improvement is 0.1010, which is greater than the required margin of 0.01. Model B also achieved 1.0000 accuracy, precision and recall on this evaluation set, while Model A achieved 0.9000 accuracy, 0.8333 precision and 1.0000 recall.

The main trade-off is inference speed. Model B took an average of 1536.44 ms per example, whereas Model A took 373.35 ms. Therefore, Model B was approximately 4.12 times slower in this run. The evaluation contains only 10 examples, so the result is useful for demonstrating the comparison workflow but is not enough to establish that Model B will always perform better on a larger or different dataset. A larger, representative test set would be needed to confirm the result. Based on the measured macro-F1 and the preselected margin, Model B is assigned the `champion` alias and Model A the `challenger` alias.

## 6. Limitations and declaration

- The evaluation set contains only 10 examples, so the measured metrics may change with a larger and more diverse dataset.
- The latency measurements depend on the computer, device, model cache, and runtime environment.
- The project compares pretrained models on a small custom dataset; it does not fine-tune either model.
- An AI assistant was used to help prepare and explain parts of the project code and documentation. The student should review, run, understand, and be prepared to explain the submitted work during the viva.

## Repository contents

```text
1ms25ci105-mlflow-champion-challenger/
├── README.md
├── requirements.txt
├── evaluate.py                 # existing evaluation script; include your working copy
├── register_champion.py
├── predict_from_mlflow.py
├── data/
│   ├── eval_inputs.csv         # copy your actual evaluation dataset here
│   └── new_data.csv
├── outputs/
│   ├── predictions_*.csv      # copy actual prediction outputs here
│   ├── metrics_summary.json   # include actual logged metrics
│   ├── plots/                  # include your actual evaluation plots
│   └── new_data_predictions.csv
└── screenshots/
    ├── mlflow_compare.png
    ├── registry_aliases.png
    └── new_data_prediction_terminal.png
```

Do not commit `mlruns/`, `mlflow.db`, or downloaded model-weight files. Commit the project at least three times during development. Submit the GitHub repository link on the LMS. If the repository is private, add the instructor as a collaborator.
