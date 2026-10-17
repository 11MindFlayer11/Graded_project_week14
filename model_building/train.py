"""Stage 3 - Model building with experiment tracking (MLflow) and registration (HF Hub).

Run: python model_building/train.py
Set MLFLOW_TRACKING_URI to use a tracking server; otherwise a local SQLite store is used.
"""
import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import ARTIFACT_DIR, PROCESSED_DIR, ROOT, SPLIT_FILES, get_api, repo_ids

warnings.filterwarnings("ignore")
SEED = 42
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load data
api = get_api()
if api is not None:
    from huggingface_hub import hf_hub_download
    ids = repo_ids(api)
    for name in SPLIT_FILES:
        hf_hub_download(ids["dataset"], f"processed/{name}", repo_type="dataset",
                        token=api.token, local_dir=str(PROCESSED_DIR.parent))
    print("Loaded train/test splits from the Hugging Face dataset repo.")
else:
    print("HF_TOKEN not set - loading train/test splits from local files.")
Xtrain = pd.read_csv(PROCESSED_DIR / "Xtrain.csv")
Xtest = pd.read_csv(PROCESSED_DIR / "Xtest.csv")
ytrain = pd.read_csv(PROCESSED_DIR / "ytrain.csv")["ProdTaken"]
ytest = pd.read_csv(PROCESSED_DIR / "ytest.csv")["ProdTaken"]

numeric = Xtrain.select_dtypes(include="number").columns.tolist()
categorical = Xtrain.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())

# ------------------------------------------------- candidate models + search spaces
candidates = {
    "logistic_regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"clf__max_depth": [3, 5, 8, 12, None], "clf__min_samples_leaf": [1, 5, 10, 20]}),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [150, 300, 500], "clf__max_depth": [8, 12, 20, None],
         "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", 0.5]}),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [150, 300], "clf__learning_rate": [0.05, 0.1, 0.2],
         "clf__max_depth": [2, 3, 5], "clf__subsample": [0.8, 1.0]}),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight, random_state=SEED, n_jobs=2),
        {"clf__n_estimators": [150, 300, 500], "clf__learning_rate": [0.05, 0.1, 0.2],
         "clf__max_depth": [3, 5, 8], "clf__subsample": [0.8, 1.0], "clf__colsample_bytree": [0.7, 1.0]}),
}


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, prob)}


# ----------------------------------------------------------- experiment tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
mlflow.set_experiment("tourism-wellness-package")
print("MLflow tracking URI:", mlflow.get_tracking_uri())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, searches = [], {}
with mlflow.start_run(run_name="model-comparison") as parent:
    mlflow.log_params({"train_rows": len(Xtrain), "test_rows": len(Xtest), "cv_folds": 5,
                       "search_scoring": "f1", "n_iter_per_model": 12})
    for name, (estimator, grid) in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("clf", estimator)])
        search = RandomizedSearchCV(pipe, grid, n_iter=12, scoring="f1", cv=cv,
                                    random_state=SEED, n_jobs=1, refit=True)
        with mlflow.start_run(run_name=name, nested=True):
            search.fit(Xtrain, ytrain)
            prob = search.predict_proba(Xtest)[:, 1]
            test_m = evaluate(ytest, prob, 0.5)
            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_f1", search.best_score_)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        searches[name] = search
        rows.append({"model": name, "cv_f1": search.best_score_, **{f"test_{k}": v for k, v in test_m.items()},
                     "best_params": json.dumps({k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                                               default=str)})
        print(f"{name:20s} cv_f1={search.best_score_:.4f}  test_roc_auc={test_m['roc_auc']:.4f}  test_f1={test_m['f1']:.4f}")

    results = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]          # selected on cross-validation, NOT on the test set
    best = searches[best_name].best_estimator_

    # Choose the decision threshold from out-of-fold predictions on the training data only
    oof = cross_val_predict(best, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
    grid_t = np.round(np.arange(0.20, 0.81, 0.01), 2)
    threshold = float(grid_t[int(np.argmax([f1_score(ytrain, oof >= t) for t in grid_t]))])

    test_prob = best.predict_proba(Xtest)[:, 1]
    final = evaluate(ytest, test_prob, threshold)
    default = evaluate(ytest, test_prob, 0.5)
    cm = confusion_matrix(ytest, (test_prob >= threshold).astype(int)).tolist()

    mlflow.log_params({"selected_model": best_name, "decision_threshold": threshold})
    mlflow.log_metrics({f"final_test_{k}": v for k, v in final.items()})
    mlflow.sklearn.log_model(best, "model", registered_model_name="tourism-wellness-model",
                             serialization_format="cloudpickle")

    print(f"\nSelected model: {best_name} | decision threshold: {threshold}")
    print("Test metrics @ threshold 0.50:", {k: round(v, 4) for k, v in default.items()})
    print(f"Test metrics @ threshold {threshold}:", {k: round(v, 4) for k, v in final.items()})
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)

    # ------------------------------------------------------------ save artifacts
    bundle = {"pipeline": best, "threshold": threshold, "model_name": best_name,
              "features": Xtrain.columns.tolist()}
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")
    results.to_csv(ARTIFACT_DIR / "experiment_results.csv", index=False)
    summary = {"selected_model": best_name, "decision_threshold": threshold,
               "test_metrics_at_threshold": final, "test_metrics_at_0.5": default, "confusion_matrix": cm,
               "mlflow_run_id": parent.info.run_id}
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    mlflow.log_artifact(str(ARTIFACT_DIR / "metrics.json"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "experiment_results.csv"))

# ------------------------------------------------- register model on the HF Hub
if api is not None:
    api.create_repo(ids["model"], repo_type="model", private=False, exist_ok=True)
    card = (f"# Wellness Tourism Package purchase model\n\nSelected model: **{best_name}** "
            f"(decision threshold {threshold}).\n\nTest ROC-AUC {final['roc_auc']:.3f}, "
            f"recall {final['recall']:.3f}, precision {final['precision']:.3f}, F1 {final['f1']:.3f}.\n")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=ids["model"], repo_type="model")
    for fname in ["model.joblib", "metrics.json", "experiment_results.csv"]:
        api.upload_file(path_or_fileobj=str(ARTIFACT_DIR / fname), path_in_repo=fname,
                        repo_id=ids["model"], repo_type="model", commit_message=f"Register {fname}")
    print(f"Model registered: https://huggingface.co/{ids['model']}")
else:
    print("HF_TOKEN not set - model saved locally only:", ARTIFACT_DIR / "model.joblib")
