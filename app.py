"""Streamlit front-end for the Online Shoppers Purchasing Intention classifiers.

Loads the five pipelines saved by the notebook (model/*.joblib), lets the user upload a
test CSV (or use the bundled one), and shows metrics, confusion matrix and reports.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, ConfusionMatrixDisplay,
                             classification_report, roc_curve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_PATH = os.path.join(BASE_DIR, "model", "feature_info.json")
SAMPLE_PATH = os.path.join(BASE_DIR, "test_data.csv")
LABELS = ["No purchase", "Purchase"]


# ----------------------------------------------------------------------------------------
# Loading and validation helpers
# ----------------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Read the metadata file and load every saved pipeline once per server session."""
    with open(INFO_PATH) as f:
        info = json.load(f)
    models = {name: joblib.load(os.path.join(BASE_DIR, path)) for name, path in info["models"].items()}
    return info, models


def to_binary(series):
    """Turn True/False, 'TRUE'/'FALSE' or 1/0 values into numeric 1/0 (invalid values become NaN)."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        series = (series.astype(str).str.strip().str.lower()
                  .map({"true": 1, "false": 0, "1": 1, "0": 0}))
    return pd.to_numeric(series, errors="coerce")


def prepare(raw, info):
    """Check the uploaded frame against the training schema.

    Returns (X, y, problems). y is None when the target column is absent or unusable.
    """
    df = raw.copy()
    df.columns = df.columns.str.strip()
    features, target = info["feature_columns"], info["target"]

    missing = [c for c in features if c not in df.columns]
    if missing:
        return None, None, [f"Missing required columns: {', '.join(missing)}"]

    X = df[features].copy()
    X["Weekend"] = to_binary(X["Weekend"])

    problems = []
    nan_counts = X[info["numeric_columns"]].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        problems.append("Missing or non-numeric values in: " + ", ".join(nan_counts.index))

    y = None
    if target in df.columns:
        y_try = to_binary(df[target])
        if y_try.isna().any() or not set(y_try.unique()) <= {0, 1}:
            problems.append(f"Column '{target}' must contain only 0/1 or True/False values.")
        else:
            y = y_try.astype(int)
    return X, y, problems


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def get_scores(model, features):
    """Positive-class score for AUC: probability if available, else SVM decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate(model, features, target):
    """The six metrics required by the assignment."""
    pred = model.predict(features)
    return {
        "Accuracy": accuracy_score(target, pred),
        "AUC": roc_auc_score(target, get_scores(model, features)),
        "Precision": precision_score(target, pred, zero_division=0),
        "Recall": recall_score(target, pred, zero_division=0),
        "F1": f1_score(target, pred, zero_division=0),
        "MCC": matthews_corrcoef(target, pred),
    }


@st.cache_data(show_spinner="Scoring all five models...")
def compare_all(_models, X, y):
    """Metrics for every model on the same data (the leading underscore skips hashing)."""
    table = pd.DataFrame({name: evaluate(m, X, y) for name, m in _models.items()}).T
    table.index.name = "Model"
    return table.round(4)


# ----------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Shopper Intent Lab", page_icon="🛒", layout="wide")
    info, models = load_assets()

    st.title("🛒 Shopper Intent Lab")
    st.caption("Will an online-shop browsing session end in a purchase? Compare five classifiers "
               "on the UCI Online Shoppers Purchasing Intention data.")

    # ---- sidebar controls
    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Choose a model", list(models.keys()))
        options = ["Upload a CSV"] + (["Use the bundled test set"] if os.path.exists(SAMPLE_PATH) else [])
        source = st.radio("Data source", options)
        upload = st.file_uploader("Test data (CSV)", type="csv") if source == "Upload a CSV" else None
        with st.expander("Columns the CSV must contain"):
            st.write(", ".join(info["feature_columns"]))
            st.write(f"Optional target column: **{info['target']}** (needed for metrics).")

    # ---- read data
    raw = None
    try:
        if source == "Use the bundled test set":
            raw = pd.read_csv(SAMPLE_PATH)
        elif upload is not None:
            raw = pd.read_csv(upload)
    except Exception as err:  # unreadable or empty file
        st.error(f"Could not read the CSV: {err}")
        st.stop()
    if raw is None:
        st.info("Upload a test CSV in the sidebar, or switch to the bundled test set, to begin.")
        st.stop()

    X, y, problems = prepare(raw, info)
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    model = models[model_name]
    pred = model.predict(X)

    # ---- data summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(X):,}")
    c2.metric("Features used", len(info["feature_columns"]))
    if y is not None:
        c3.metric("Purchase share in data", f"{y.mean():.1%}")
    with st.expander("Preview of the uploaded data"):
        st.dataframe(raw.head(20), use_container_width=True)

    tabs = st.tabs(["Metrics", "Confusion matrix", "Classification report",
                    "ROC curve", "All models", "Predictions"])

    # ---- tabs that need labels
    if y is None:
        for i in range(5):
            with tabs[i]:
                st.warning(f"This CSV has no '{info['target']}' column, so metrics cannot be computed. "
                           "Predictions are available in the last tab.")
    else:
        with tabs[0]:
            st.subheader(f"{model_name}: test-set metrics")
            scores = evaluate(model, X, y)
            cols = st.columns(6)
            for col, (name, value) in zip(cols, scores.items()):
                col.metric(name, f"{value:.4f}")
            baseline = max(y.mean(), 1 - y.mean())
            st.caption(f"Reference point: always predicting the majority class would give "
                       f"{baseline:.1%} accuracy, so judge models by AUC, Recall, F1 and MCC as well.")

        with tabs[1]:
            as_pct = st.checkbox("Show as share of each actual class", value=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y, pred, display_labels=LABELS, cmap="Blues", ax=ax, colorbar=False,
                normalize="true" if as_pct else None,
                values_format=".2f" if as_pct else "d")
            ax.set_title(f"{model_name}")
            ax.grid(False)
            st.pyplot(fig)
            plt.close(fig)

        with tabs[2]:
            report = classification_report(y, pred, target_names=LABELS, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        with tabs[3]:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            fpr, tpr, _ = roc_curve(y, get_scores(model, X))
            ax.plot(fpr, tpr, label=model_name)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)

        with tabs[4]:
            st.subheader("All five models on this data")
            table = compare_all(models, X, y)
            st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf").format("{:.4f}"),
                         use_container_width=True)

    # ---- predictions (works with or without labels)
    with tabs[5]:
        out = raw.copy()
        out["Predicted"] = pd.Series(pred, index=out.index).map({0: LABELS[0], 1: LABELS[1]})
        out["Score"] = get_scores(model, X).round(4)
        st.dataframe(out.head(200), use_container_width=True)
        st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()
