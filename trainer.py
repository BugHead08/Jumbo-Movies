# trainer.py
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Union
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
import imblearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC  # If you don't need probabilities, consider LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse

# ============================
# Constants & paths (absolute)
# ============================
BASE_DIR: Path = Path(__file__).resolve().parent

# Where per-sentiment cluster artifacts (TF-IDF + KMeans) are stored
ARTIFACT_DIR: Path = BASE_DIR / "cluster_models"

# Where classification models are stored
CLASSIF_DIR: Path = BASE_DIR / "classification_models"

# Global TF-IDF for the whole corpus
GLOBAL_TFIDF_PATH: Path = BASE_DIR / "tfidf_vectorizer.pkl"

# Classifier model paths (under classification_models/)
NB_PATH: Path = CLASSIF_DIR / "naive_bayes_model.pkl"
SVM_PATH: Path = CLASSIF_DIR / "svm_model.pkl"
LR_PATH: Path = CLASSIF_DIR / "logreg_model.pkl"

# Valid labels accepted for training; clustering will only target pos/neg
VALID_LABELS = {"positive", "neutral", "negative"}
TARGET_SENTIMENTS = ("positive", "negative")  # <-- only these get clusters

# ============================
# Utilities
# ============================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_dense(X):
    return X.toarray() if issparse(X) else X

def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)

def _can_stratify(y: pd.Series, min_count: int = 2) -> bool:
    """Return True if every class appears at least `min_count` times."""
    return all(c >= min_count for c in Counter(y).values())

def _validate_inputs(
    texts: List[str],
    labels: Optional[List[str]]
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Clean and validate inputs:
    - texts: must contain at least 2 non-empty lines
    - labels: optional; if provided, must be same length as texts and in VALID_LABELS
    """
    texts = [str(t).strip() for t in texts if str(t).strip() != ""]
    if len(texts) < 2:
        raise ValueError("At least 2 sentences are required for training/clustering.")

    if labels is None:
        return texts, None

    if len(labels) != len(texts):
        raise ValueError(f"Label count ({len(labels)}) must match text count ({len(texts)}).")

    labels = [str(l).lower().strip() for l in labels]
    for l in labels:
        if l not in VALID_LABELS:
            raise ValueError("Labels must be one of: positive, neutral, negative.")
    return texts, labels

# ============================
# Global TF-IDF
# ============================
def fit_global_tfidf(
    texts: List[str],
    max_features: int = 1000,
    *,
    verbose: bool = True
) -> TfidfVectorizer:
    """
    Fit a global TF-IDF on all input texts and persist it.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),     # sensible defaults for short texts
        sublinear_tf=True
    )
    tfidf.fit(pd.Series(texts))
    joblib.dump(tfidf, GLOBAL_TFIDF_PATH, compress=3)
    _log(f"✅ Saved global TF-IDF → {GLOBAL_TFIDF_PATH}", verbose)
    return tfidf

# ============================
# Classification (NB, SVM, LR)
# ============================
def train_nb_svm_lr(
    tfidf: TfidfVectorizer,
    texts: List[str],
    labels: List[str],
    smote_k: int = 1,
    test_size_nb: float = 0.2,
    test_size_svm: float = 0.3,
    test_size_lr: float = 0.25,
    random_state: int = 42,
    *,
    verbose: bool = True,
) -> Tuple[Optional[MultinomialNB], Optional[SVC], Optional[LogisticRegression], Dict[str, Any]]:
    """
    Train Naive Bayes, SVM, and Logistic Regression when at least 2 classes exist.
    Uses SMOTE on the TRAIN splits (requires dense arrays).
    Saves all models to disk and returns simple training stats.
    """
    # Ensure classifier directory exists
    _ensure_dir(CLASSIF_DIR)

    y = pd.Series(labels)
    X = tfidf.transform(pd.Series(texts))
    metrics: Dict[str, Any] = {
        "sklearn_version": sklearn.__version__,
        "imblearn_version": imblearn.__version__,
    }

    if y.nunique() < 2:
        note = "Only one class present; classification skipped."
        _log(f"ℹ️ {note}", verbose)
        return None, None, None, {"note": note}

    use_strat = _can_stratify(y, min_count=2)

    # ----- Naive Bayes -----
    X_nb_tr, X_nb_te, y_nb_tr, y_nb_te = train_test_split(
        X, y, test_size=test_size_nb, random_state=random_state, stratify=y if use_strat else None
    )
    X_nb_tr_dense = _to_dense(X_nb_tr)
    X_nb_tr_sm, y_nb_tr_sm = SMOTE(random_state=random_state, k_neighbors=smote_k)\
        .fit_resample(X_nb_tr_dense, y_nb_tr)
    nb = MultinomialNB().fit(X_nb_tr_sm, y_nb_tr_sm)
    joblib.dump(nb, NB_PATH, compress=3)
    _log(f"✅ Saved Naive Bayes → {NB_PATH}", verbose)
    metrics["nb_train_class_counts"] = dict(pd.Series(y_nb_tr_sm).value_counts().to_dict())

    # ----- SVM (linear) -----
    X_svm_tr, X_svm_te, y_svm_tr, y_svm_te = train_test_split(
        X, y, test_size=test_size_svm, random_state=random_state, stratify=y if use_strat else None
    )
    X_svm_tr_dense = _to_dense(X_svm_tr)
    X_svm_tr_sm, y_svm_tr_sm = SMOTE(random_state=random_state, k_neighbors=smote_k)\
        .fit_resample(X_svm_tr_dense, y_svm_tr)
    svm = SVC(kernel="linear", probability=True, random_state=random_state).fit(X_svm_tr_sm, y_svm_tr_sm)
    joblib.dump(svm, SVM_PATH, compress=3)
    _log(f"✅ Saved SVM → {SVM_PATH}", verbose)
    metrics["svm_train_class_counts"] = dict(pd.Series(y_svm_tr_sm).value_counts().to_dict())

    # ----- Logistic Regression -----
    X_lr_tr, X_lr_te, y_lr_tr, y_lr_te = train_test_split(
        X, y, test_size=test_size_lr, random_state=random_state, stratify=y if use_strat else None
    )
    X_lr_tr_dense = _to_dense(X_lr_tr)
    X_lr_tr_sm, y_lr_tr_sm = SMOTE(random_state=random_state, k_neighbors=smote_k)\
        .fit_resample(X_lr_tr_dense, y_lr_tr)

    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto"
    ).fit(X_lr_tr_sm, y_lr_tr_sm)

    joblib.dump(lr, LR_PATH, compress=3)
    _log(f"✅ Saved Logistic Regression → {LR_PATH}", verbose)
    metrics["lr_train_class_counts"] = dict(pd.Series(y_lr_tr_sm).value_counts().to_dict())

    return nb, svm, lr, metrics

# ============================
# KMeans helpers
# ============================
def _best_kmeans(X_dense: np.ndarray, random_state: int = 42) -> Tuple[KMeans, int, float]:
    """
    Choose k in [2..min(n_samples, 10)] by maximizing silhouette score.
    Returns (model, best_k, best_score). Falls back to k=2 if needed.
    Uses n_init=10 for broad scikit-learn compatibility.
    """
    n = X_dense.shape[0]
    if n < 2:
        raise ValueError("At least 2 samples are required for KMeans.")
    best_score, best_k, best_model = -1.0, 2, None
    for k in range(2, min(n, 10) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_dense)
            score = silhouette_score(X_dense, labels)
            if score > best_score:
                best_score, best_k, best_model = score, k, km
        except Exception:
            continue
    if best_model is None:
        best_model = KMeans(n_clusters=2, random_state=random_state, n_init=10).fit(X_dense)
        best_k, best_score = 2, -1.0
    return best_model, best_k, best_score

def _save_local_cluster(tfidf_local: TfidfVectorizer, km: KMeans, code: str, sentiment: str, *, verbose: bool = True):
    """
    Persist a local TF-IDF and its KMeans model for a given (source, sentiment) bucket.
    """
    _ensure_dir(ARTIFACT_DIR)
    tfidf_path = ARTIFACT_DIR / f"tfidf_{code}_{sentiment}.pkl"
    kmeans_path = ARTIFACT_DIR / f"kmeans_{code}_{sentiment}.pkl"
    joblib.dump(tfidf_local, tfidf_path, compress=3)
    joblib.dump(km,          kmeans_path, compress=3)
    _log(f"✅ Saved local TF-IDF → {tfidf_path}", verbose)
    _log(f"✅ Saved KMeans → {kmeans_path}", verbose)

def _cluster_group(
    texts: List[str],
    code: str,
    sentiment: str,
    max_features: int,
    random_state: int,
    *,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Cluster a subset of texts and save local artifacts.
    Returns a small summary (source, sentiment, k, silhouette).

    Robust behavior:
      - len(texts) == 0 → skip (no artifacts)
      - len(texts) == 1 → save a 1-cluster KMeans so downstream apps always find artifacts
      - len(texts) >= 2 → choose k via silhouette (2..min(n,10))
    """
    n = len(texts)
    if n == 0:
        note = "data=0"
        _log(f"⚠️ Skip clustering for {code}-{sentiment}: {note}", verbose)
        return {"source": code, "sentiment": sentiment, "k": 0, "silhouette": float("nan"), "note": note}

    tfidf_local = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_local = tfidf_local.fit_transform(pd.Series(texts))
    Xd = _to_dense(X_local)

    if n == 1:
        km = KMeans(n_clusters=1, random_state=random_state, n_init=1).fit(Xd)
        _save_local_cluster(tfidf_local, km, code, sentiment, verbose=verbose)
        _log(f"📊 {code.upper()} - {sentiment}: k=1 (only 1 sample), silhouette=NaN", verbose)
        return {"source": code, "sentiment": sentiment, "k": 1, "silhouette": float("nan"), "note": "k=1(data=1)"}

    km, k, sc = _best_kmeans(Xd, random_state=random_state)
    _save_local_cluster(tfidf_local, km, code, sentiment, verbose=verbose)
    _log(f"📊 {code.upper()} - {sentiment}: k={k}, silhouette={sc:.4f}", verbose)
    return {"source": code, "sentiment": sentiment, "k": k, "silhouette": round(sc, 6)}

# ============================
# Artifact helpers
# ============================
def list_artifacts(*, verbose: bool = True) -> List[str]:
    """
    Returns a list of saved artifact paths and (optionally) prints them).
    """
    paths: List[str] = []
    # Global TF-IDF
    if GLOBAL_TFIDF_PATH.exists():
        paths.append(str(GLOBAL_TFIDF_PATH))
    # Classifiers
    if CLASSIF_DIR.exists():
        for f in sorted(CLASSIF_DIR.glob("*.pkl")):
            paths.append(str(f))
    # Clustering artifacts
    if ARTIFACT_DIR.exists():
        for f in sorted(ARTIFACT_DIR.glob("*.pkl")):
            paths.append(str(f))
    if verbose:
        if paths:
            print("📂 Artifacts saved:")
            for p in paths:
                print(f"  • {p}")
        else:
            print("ℹ️ No artifacts found yet.")
    return paths

def save_training_report(summary: Dict[str, Any], path: Union[str, Path] = "training_report.json", *, verbose: bool = True) -> str:
    """
    Persist the summary dict (returned by train_and_cluster) as a JSON report.
    """
    out_path = Path(path)
    summary = {**summary, "generated_at": datetime.utcnow().isoformat() + "Z"}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _log(f"📝 Saved training report → {out_path}", verbose)
    return str(out_path)

# ============================
# Main training API
# ============================
def train_and_cluster(
    texts: List[str],
    labels: Optional[List[str]] = None,
    *,
    max_features: int = 1000,
    smote_k: int = 1,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train global TF-IDF; optionally train NB, SVM & LR if labels are provided; then run K-Means.
    Logs what is saved and where (if verbose=True).

    Clustering strategy:
      - If classifiers are trained: cluster per sentiment using NB, SVM, and LR predictions
        for ONLY: positive and negative.
      - If no labels/classifiers: single K-Means run over all texts.
    """
    texts, labels = _validate_inputs(texts, labels)
    _ensure_dir(ARTIFACT_DIR)
    _ensure_dir(CLASSIF_DIR)

    _log("📦 Start training & clustering...", verbose)

    # 1) Global TF-IDF
    tfidf = fit_global_tfidf(texts, max_features=max_features, verbose=verbose)

    # 2) (Optional) classification
    nb = svm = lr = None
    cls_metrics: Dict[str, Any] = {}
    if labels is not None:
        nb, svm, lr, cls_metrics = train_nb_svm_lr(
            tfidf=tfidf,
            texts=texts,
            labels=labels,
            smote_k=smote_k,
            random_state=random_state,
            verbose=verbose
        )

    # 3) Clustering
    clustering_summary: List[Dict[str, Any]] = []

    # If no classifiers at all, cluster everything at once
    if nb is None and svm is None and lr is None:
        _log("📌 Clustering all texts (no labels)...", verbose)
        clustering_summary.append(
            _cluster_group(texts, "all", "all", max_features=max_features, random_state=random_state, verbose=verbose)
        )
    else:
        # Helper to run clustering for one classifier's predictions
        def _cluster_by(pred_labels, code: str):
            for sent in TARGET_SENTIMENTS:  # only positive & negative
                subset = [t for t, p in zip(texts, pred_labels) if p == sent]
                clustering_summary.append(
                    _cluster_group(subset, code, sent, max_features=max_features, random_state=random_state, verbose=verbose)
                )

        # NB
        if nb is not None:
            _log("📌 Clustering by NB predictions...", verbose)
            y_nb = nb.predict(tfidf.transform(pd.Series(texts)))
            _cluster_by(y_nb, "nb")

        # SVM
        if svm is not None:
            _log("📌 Clustering by SVM predictions...", verbose)
            y_svm = svm.predict(_to_dense(tfidf.transform(pd.Series(texts))))
            _cluster_by(y_svm, "svm")

        # LR
        if lr is not None:
            _log("📌 Clustering by LR predictions...", verbose)
            y_lr = lr.predict(_to_dense(tfidf.transform(pd.Series(texts))))
            _cluster_by(y_lr, "lr")

    # 4) Artifact list
    artifacts = list_artifacts(verbose=verbose)

    summary = {
        "text_count": len(texts),
        "has_labels": labels is not None,
        "classification": cls_metrics,
        "clustering": clustering_summary,
        "artifacts": artifacts
    }
    return summary

# ============================
# Optional inference utility
# ============================
def predict_texts(texts: List[str]) -> pd.DataFrame:
    """
    Predict NB/SVM/LR labels for new texts after training.
    Returns a DataFrame with columns: text, nb_pred, svm_pred, lr_pred
    """
    if not GLOBAL_TFIDF_PATH.exists():
        raise FileNotFoundError("Global TF-IDF not found. Please train the model first.")

    tfidf = joblib.load(GLOBAL_TFIDF_PATH)
    X = tfidf.transform(pd.Series(texts))
    Xd = _to_dense(X)

    out = {"text": texts}

    if NB_PATH.exists():
        nb = joblib.load(NB_PATH)
        out["nb_pred"] = nb.predict(X)
    else:
        out["nb_pred"] = [None] * len(texts)

    if SVM_PATH.exists():
        svm = joblib.load(SVM_PATH)
        out["svm_pred"] = svm.predict(Xd)
    else:
        out["svm_pred"] = [None] * len(texts)

    if LR_PATH.exists():
        lr = joblib.load(LR_PATH)
        out["lr_pred"] = lr.predict(Xd)
    else:
        out["lr_pred"] = [None] * len(texts)

    return pd.DataFrame(out)

# ============================
# Example usage (optional)
# ============================
if __name__ == "__main__":
    # Minimal smoke test
    sample_texts = [
        "I love the visuals",
        "The plot is boring",
        "Music is great",
        "Ending disappointed me",
        "Acting was fantastic",
        "Dialog felt stiff"
    ]
    sample_labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

    summary = train_and_cluster(sample_texts, sample_labels, verbose=True)
    save_training_report(summary)
    print(predict_texts(["Great acting", "Too slow for me"]))
