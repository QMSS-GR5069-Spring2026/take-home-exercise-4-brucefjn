# Databricks notebook source
# MAGIC %md
# MAGIC # Homework #5: Model Deployment — F1 Race Winner Prediction
# MAGIC
# MAGIC **Author:** jf3774
# MAGIC **Course:** GR5069 — Data Pipeline in Practice (Spring 2026)
# MAGIC
# MAGIC ## Task
# MAGIC Using the F1 (Ergast) dataset, build two predictive models, log each in MLflow, and
# MAGIC write predictions from each model into two separate tables in my own database.
# MAGIC
# MAGIC ## Approach
# MAGIC - **Target:** Binary classification — did the driver **win** the race? (`positionOrder == 1`)
# MAGIC - **Model 1:** `LogisticRegression`
# MAGIC - **Model 2:** `RandomForestClassifier`
# MAGIC - **Database:** `gr5069.jf3774.hw5_predictions_logreg` and `gr5069.jf3774.hw5_predictions_rf`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Newer mlflow needs `Sentinel` from `typing_extensions`, which is missing on the
# MAGIC stock runtime — upgrade and restart Python so the new packages take effect.

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports

# COMMAND ----------

import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import mlflow
import mlflow.sklearn

print("MLflow version:", mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load F1 data
# MAGIC Read CSVs from the course volume, infer schema, and pull into pandas — same pattern
# MAGIC as the class linear-regression example.

# COMMAND ----------

BASE_PATH = "/Volumes/gr5069/raw/f1_data"

results_df      = spark.read.csv(f"{BASE_PATH}/results.csv",      header=True, inferSchema=True).toPandas()
races_df        = spark.read.csv(f"{BASE_PATH}/races.csv",        header=True, inferSchema=True).toPandas()
drivers_df      = spark.read.csv(f"{BASE_PATH}/drivers.csv",      header=True, inferSchema=True).toPandas()
constructors_df = spark.read.csv(f"{BASE_PATH}/constructors.csv", header=True, inferSchema=True).toPandas()
qualifying_df   = spark.read.csv(f"{BASE_PATH}/qualifying.csv",   header=True, inferSchema=True).toPandas()

print("results     :", results_df.shape)
print("races       :", races_df.shape)
print("drivers     :", drivers_df.shape)
print("constructors:", constructors_df.shape)
print("qualifying  :", qualifying_df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature engineering and joins
# MAGIC One row per (race, driver). Features blend grid position, qualifying performance,
# MAGIC constructor identity, circuit, and driver age — all known *before* the race finishes.
# MAGIC Label is `1` if the driver finished 1st, else `0`.

# COMMAND ----------

# Start from results — the row grain we want
df = results_df[[
    "raceId", "driverId", "constructorId", "grid", "positionOrder", "points", "laps"
]].copy()

# Merge race metadata (year, round, circuit)
df = df.merge(
    races_df[["raceId", "year", "round", "circuitId"]],
    on="raceId",
    how="left",
)

# Merge driver info -> compute driver age in years
drivers_small = drivers_df[["driverId", "dob", "nationality"]].rename(
    columns={"nationality": "driver_nationality"}
)
df = df.merge(drivers_small, on="driverId", how="left")
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["driver_age"] = df["year"] - df["dob"].dt.year

# Merge constructor nationality
cons_small = constructors_df[["constructorId", "nationality"]].rename(
    columns={"nationality": "constructor_nationality"}
)
df = df.merge(cons_small, on="constructorId", how="left")

# Merge qualifying position (best of Q1/Q2/Q3 -> we just take min)
qual_small = (
    qualifying_df
    .groupby(["raceId", "driverId"], as_index=False)["position"]
    .min()
    .rename(columns={"position": "qual_position"})
)
df = df.merge(qual_small, on=["raceId", "driverId"], how="left")

print("Merged shape:", df.shape)

# COMMAND ----------

# Target: did this driver win the race?
df["winner"] = (df["positionOrder"] == 1).astype(int)

# Clean
df = df[df["grid"] > 0].copy()              # grid 0 = pit lane / DNS
df = df.dropna(subset=["driver_age"])
df["qual_position"] = df["qual_position"].fillna(99)  # didn't qualify -> sentinel value

# Encode the two categoricals we kept as strings
df["driver_nat_code"]      = df["driver_nationality"].astype("category").cat.codes
df["constructor_nat_code"] = df["constructor_nationality"].astype("category").cat.codes

# Final feature set
feature_cols = [
    "grid", "qual_position", "constructorId", "circuitId",
    "year", "round", "driver_age", "driver_nat_code", "constructor_nat_code",
]
X = df[feature_cols]
y = df["winner"]

# Keep IDs for the prediction tables we'll write later
df_ids = df[["raceId", "driverId", "constructorId"]].copy()

print("X shape    :", X.shape)
print("Win rate   :", round(y.mean(), 4))
print("Class counts:", y.value_counts().to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train / test split
# MAGIC Stratified by the target so the test set actually contains winners (only ~5% of rows).

# COMMAND ----------

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, df_ids, test_size=0.25, random_state=42, stratify=y,
)

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train win rate:", round(y_train.mean(), 4))
print("Test  win rate:", round(y_test.mean(), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create prediction tables
# MAGIC One Delta table per model in my personal schema. `CREATE OR REPLACE` keeps the
# MAGIC notebook idempotent — rerunning won't fail or accumulate stale rows.

# COMMAND ----------

MY_SCHEMA = "gr5069.jf3774"
TABLE_LOGREG = f"{MY_SCHEMA}.hw5_predictions_logreg"
TABLE_RF     = f"{MY_SCHEMA}.hw5_predictions_rf"

CREATE_TABLE_DDL = """
CREATE OR REPLACE TABLE {full_name} (
    raceId            INT,
    driverId          INT,
    constructorId     INT,
    actual_winner     INT,
    predicted_winner  INT,
    predicted_proba   DOUBLE,
    model_name        STRING,
    run_id            STRING
) USING DELTA
"""

spark.sql(CREATE_TABLE_DDL.format(full_name=TABLE_LOGREG))
spark.sql(CREATE_TABLE_DDL.format(full_name=TABLE_RF))

print(f"Created {TABLE_LOGREG}")
print(f"Created {TABLE_RF}")
spark.sql(f"SHOW TABLES IN {MY_SCHEMA}").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow experiment setup

# COMMAND ----------

EXPERIMENT_NAME = "/Users/jf3774@columbia.edu/hw5_f1_winner_prediction"

try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
except Exception:
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)
print("Experiment ID:", experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Logging helper
# MAGIC One reusable function that trains a model, logs everything to MLflow, and returns
# MAGIC the predictions and run id. Logs hyperparameters, the model, four metrics, and two
# MAGIC artifacts (confusion matrix PNG + ROC curve PNG).

# COMMAND ----------

def train_and_log(experiment_id, run_name, model, params,
                  X_train, X_test, y_train, y_test):
    """Fit `model`, log to MLflow, return (preds, proba, run_id, metrics)."""
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # ---- Train ----
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # ---- Log the model ----
        mlflow.sklearn.log_model(model, "model")

        # ---- Log hyperparameters ----
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # ---- Log four metrics ----
        metrics = {
            "accuracy":  accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall":    recall_score(y_test, preds, zero_division=0),
            "f1":        f1_score(y_test, preds, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, proba),
        }
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name:10s}: {metric_value:.4f}")

        # ---- Artifact 1: confusion matrix ----
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Winner", "Winner"],
            yticklabels=["Not Winner", "Winner"], ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix — {run_name}")
        plt.tight_layout()
        cm_path = os.path.join(tempfile.gettempdir(), f"confusion_matrix_{run_name}.png")
        fig_cm.savefig(cm_path)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path)

        # ---- Artifact 2: ROC curve ----
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve — {run_name}")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(tempfile.gettempdir(), f"roc_curve_{run_name}.png")
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path)

        run_id = run.info.run_id
        print(f"  -> run_id: {run_id}")
        return preds, proba, run_id, metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model 1 — Logistic Regression

# COMMAND ----------

logreg_params = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42,
}

logreg_model = LogisticRegression(**logreg_params)

logreg_preds, logreg_proba, logreg_run_id, logreg_metrics = train_and_log(
    experiment_id, "Model_1_LogisticRegression", logreg_model, logreg_params,
    X_train, X_test, y_train, y_test,
)
print(f"\nLogistic Regression run_id: {logreg_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model 2 — Random Forest

# COMMAND ----------

rf_params = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

rf_model = RandomForestClassifier(**rf_params)

rf_preds, rf_proba, rf_run_id, rf_metrics = train_and_log(
    experiment_id, "Model_2_RandomForest", rf_model, rf_params,
    X_train, X_test, y_train, y_test,
)
print(f"\nRandom Forest run_id: {rf_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Compare the two models

# COMMAND ----------

metric_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]

comparison = pd.DataFrame({
    "Metric":             metric_order,
    "LogisticRegression": [logreg_metrics[m] for m in metric_order],
    "RandomForest":       [rf_metrics[m]     for m in metric_order],
})
comparison["Winner"] = comparison.apply(
    lambda row: "RandomForest" if row["RandomForest"] >= row["LogisticRegression"]
    else "LogisticRegression",
    axis=1,
)
display(comparison.round(4))

# COMMAND ----------

# Side-by-side bar chart
x = np.arange(len(comparison))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, comparison["LogisticRegression"], width,
       label="LogisticRegression", color="steelblue")
ax.bar(x + width / 2, comparison["RandomForest"],       width,
       label="RandomForest",       color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Metric"])
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Logistic Regression vs Random Forest")
ax.legend()
ax.set_ylim(0, 1.05)
for i, row in comparison.iterrows():
    ax.text(i - width / 2, row["LogisticRegression"] + 0.02,
            f'{row["LogisticRegression"]:.3f}', ha="center", fontsize=9)
    ax.text(i + width / 2, row["RandomForest"] + 0.02,
            f'{row["RandomForest"]:.3f}', ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Write predictions to the database
# MAGIC Build a pandas DataFrame for each model with IDs + actual + predicted + metadata,
# MAGIC convert to Spark, and append to the matching table — same handoff pattern as the
# MAGIC class example.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType


def build_predictions_df(ids_df, y_actual, y_pred, y_proba, model_name, run_id):
    """Shape predictions to match the prediction-table schema."""
    out = ids_df.copy()
    out["actual_winner"]    = y_actual.values.astype(int)
    out["predicted_winner"] = y_pred.astype(int)
    out["predicted_proba"]  = y_proba.astype(float)
    out["model_name"]       = model_name
    out["run_id"]           = run_id
    return out


def write_predictions_to_table(pred_pdf, table_name):
    """Convert pandas -> Spark, cast every column to match the table DDL, then append."""
    sdf = spark.createDataFrame(pred_pdf)
    sdf_typed = sdf.select(
        F.col("raceId").cast(IntegerType()).alias("raceId"),
        F.col("driverId").cast(IntegerType()).alias("driverId"),
        F.col("constructorId").cast(IntegerType()).alias("constructorId"),
        F.col("actual_winner").cast(IntegerType()).alias("actual_winner"),
        F.col("predicted_winner").cast(IntegerType()).alias("predicted_winner"),
        F.col("predicted_proba").cast(DoubleType()).alias("predicted_proba"),
        F.col("model_name").cast(StringType()).alias("model_name"),
        F.col("run_id").cast(StringType()).alias("run_id"),
    )
    sdf_typed.write.mode("append").saveAsTable(table_name)


# ---- Logistic Regression -> its own table ----
logreg_pred_pdf = build_predictions_df(
    ids_test, y_test, logreg_preds, logreg_proba,
    "LogisticRegression", logreg_run_id,
)
write_predictions_to_table(logreg_pred_pdf, TABLE_LOGREG)
print(f"Wrote {len(logreg_pred_pdf):,} rows to {TABLE_LOGREG}")

# ---- Random Forest -> its own table ----
rf_pred_pdf = build_predictions_df(
    ids_test, y_test, rf_preds, rf_proba,
    "RandomForest", rf_run_id,
)
write_predictions_to_table(rf_pred_pdf, TABLE_RF)
print(f"Wrote {len(rf_pred_pdf):,} rows to {TABLE_RF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Verify the tables

# COMMAND ----------

print(f"=== {TABLE_LOGREG} ===")
spark.sql(f"SELECT COUNT(*) AS row_count FROM {TABLE_LOGREG}").show()
spark.sql(f"SELECT * FROM {TABLE_LOGREG} LIMIT 5").show()

print(f"=== {TABLE_RF} ===")
spark.sql(f"SELECT COUNT(*) AS row_count FROM {TABLE_RF}").show()
spark.sql(f"SELECT * FROM {TABLE_RF} LIMIT 5").show()

# COMMAND ----------

# Prediction distribution per model
print("Logistic Regression — predicted_winner distribution:")
spark.sql(
    f"SELECT predicted_winner, COUNT(*) AS cnt "
    f"FROM {TABLE_LOGREG} GROUP BY predicted_winner ORDER BY predicted_winner"
).show()

print("Random Forest — predicted_winner distribution:")
spark.sql(
    f"SELECT predicted_winner, COUNT(*) AS cnt "
    f"FROM {TABLE_RF} GROUP BY predicted_winner ORDER BY predicted_winner"
).show()
