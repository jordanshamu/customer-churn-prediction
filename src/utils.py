"""
utils.py — Reusable ML Pipeline Utilities
Customer Churn Prediction | Jordan Shamukiga
========================================================
Modular, reusable functions for the full ML pipeline:
  - Data loading & validation
  - Preprocessing & feature engineering
  - Model training & evaluation
  - Business cost-benefit analysis
  - Visualization helpers

Usage:
    from src.utils import load_and_validate, evaluate_model, cost_benefit_analysis
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & STYLING
# ─────────────────────────────────────────────────────────────────────────────

# Palette consistent with portfolio style
PALETTE = {
    "primary":   "#2C3E50",   # deep navy
    "accent":    "#E74C3C",   # alert red (churn)
    "safe":      "#27AE60",   # retained green
    "mid":       "#F39C12",   # amber (at-risk)
    "light":     "#ECF0F1",
    "grid":      "#BDC3C7",
    "highlight": "#3498DB",
    "text":      "#2C3E50",
}

MODEL_COLORS = {
    "Logistic Regression": "#3498DB",
    "Random Forest":       "#E67E22",
    "XGBoost":             "#8E44AD",
}

def set_plot_style():
    """Apply consistent publication-quality plot style across all charts."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.labelsize":    12,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.titlepad":     14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        PALETTE["light"],
        "grid.linewidth":    0.8,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "font.family":       "DejaVu Sans",
        "figure.dpi":        130,
    })

set_plot_style()


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(filepath: str, target_col: str = "Churn") -> pd.DataFrame:
    """
    Load dataset and run basic validation checks.

    Parameters
    ----------
    filepath   : Path to the CSV file.
    target_col : Name of the binary churn target column.

    Returns
    -------
    df : Validated DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Target check
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

    # Class balance
    churn_rate = df[target_col].mean() if df[target_col].dtype in [float, int] else (df[target_col] == "Yes").mean()
    print(f"📊 Churn rate: {churn_rate:.1%}")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"⚠️  Missing values detected:\n{missing[missing > 0]}")
    else:
        print("✅ No missing values")

    # Duplicate rows
    dupes = df.duplicated().sum()
    print(f"{'⚠️ ' if dupes else '✅'} Duplicate rows: {dupes}")

    return df


def summarize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame: dtype, nunique, null%, sample values."""
    summary = pd.DataFrame({
        "dtype":    df.dtypes,
        "nunique":  df.nunique(),
        "null_pct": (df.isnull().mean() * 100).round(2),
        "sample":   df.iloc[0],
    })
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def encode_binary(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Label-encode Yes/No columns to 1/0."""
    df = df.copy()
    for col in cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, dict]:
    """
    One-hot encode nominal categoricals. Returns transformed df and column mapping.
    """
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    # Convert bool columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering specific to Telco churn data.
    Creates interaction and ratio features that improve model signal.
    """
    df = df.copy()

    # Tenure bands (behavioral segmentation tie-in from Project 3)
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0–6mo", "7–12mo", "13–24mo", "25–48mo", "49–72mo"]
    )

    # Revenue per month relative to contract (loyalty score proxy)
    df["monthly_charges_log"] = np.log1p(df["MonthlyCharges"])

    # Total charges per tenure month (normalised spend)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Service count (how embedded is the customer)
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing = [c for c in service_cols if c in df.columns]
    if existing:
        df["num_services"] = df[existing].apply(
            lambda row: sum(v == "Yes" for v in row), axis=1
        )

    # High-value flag (top quartile by MonthlyCharges)
    q75 = df["MonthlyCharges"].quantile(0.75)
    df["is_high_value"] = (df["MonthlyCharges"] >= q75).astype(int)

    # Month-to-month contract flag (highest churn risk contract type)
    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # Electronic check payment flag (correlated with churn in telco data)
    if "PaymentMethod" in df.columns:
        df["pays_by_echeck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    print(f"✅ Feature engineering complete. New columns: tenure_band, monthly_charges_log, "
          f"avg_monthly_spend, num_services, is_high_value, is_month_to_month, pays_by_echeck")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Full model evaluation: ROC-AUC, PR-AUC, F1, confusion matrix, classification report.

    Returns
    -------
    metrics : dict with all key evaluation metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    f1       = f1_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model":        model_name,
        "threshold":    threshold,
        "roc_auc":      round(roc_auc, 4),
        "pr_auc":       round(pr_auc, 4),
        "f1":           round(f1, 4),
        "precision":    round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall":       round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  {model_name}")
        print(f"{'─'*50}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {'':>12} Pred No   Pred Yes")
        print(f"  {'Actual No':>12}  {tn:>6}    {fp:>6}")
        print(f"  {'Actual Yes':>12}  {fn:>6}    {tp:>6}")
        print(f"{'─'*50}")

    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """Stratified k-fold CV returning mean ± std for ROC-AUC and F1."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    roc_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=skf, scoring="f1",      n_jobs=-1)

    return {
        "cv_roc_auc_mean": round(roc_scores.mean(), 4),
        "cv_roc_auc_std":  round(roc_scores.std(),  4),
        "cv_f1_mean":      round(f1_scores.mean(),  4),
        "cv_f1_std":       round(f1_scores.std(),   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUSINESS COST-BENEFIT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def cost_benefit_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_false_negative: float = 500,   # Lost CLV per missed churner
    cost_false_positive: float = 50,    # Retention offer cost per false alarm
    benefit_true_positive: float = 450, # Saved CLV per successful retention
    n_customers: int = None,
) -> pd.DataFrame:
    """
    Sweep classification thresholds and compute net business value at each point.

    Parameters
    ----------
    cost_false_negative  : Revenue lost per churner we miss (default $500 CLV).
    cost_false_positive  : Cost of a retention offer sent to a non-churner (default $50).
    benefit_true_positive: Net value of successfully retaining a churner (default $450).
    n_customers          : Scale to fleet size (optional).

    Returns
    -------
    DataFrame with threshold, net_value, precision, recall at each threshold.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        net = (tp * benefit_true_positive) - (fp * cost_false_positive) - (fn * cost_false_negative)
        records.append({
            "threshold":  round(t, 2),
            "net_value":  net,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0,
        })

    df_cba = pd.DataFrame(records)
    optimal_row = df_cba.loc[df_cba["net_value"].idxmax()]
    print(f"\n💰 Optimal threshold by business value: {optimal_row['threshold']:.2f}")
    print(f"   Net value at optimal threshold     : ${optimal_row['net_value']:,.0f}")
    print(f"   Precision / Recall                 : {optimal_row['precision']:.2%} / {optimal_row['recall']:.2%}")

    return df_cba


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig, filename: str, viz_dir: str = "../visualizations") -> str:
    """Save figure to visualizations directory. Returns save path."""
    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(viz_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  💾 Saved → {path}")
    return path


def plot_churn_distribution(df: pd.DataFrame, target_col: str = "Churn",
                             viz_dir: str = "../visualizations"):
    """Bar chart of churn vs retained with rate annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[target_col].value_counts()
    colors = [PALETTE["safe"], PALETTE["accent"]]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}\n({val/len(df):.1%})", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Churn vs. Retained Customer Distribution", fontsize=15, fontweight="bold")
    ax.set_xlabel("Customer Status", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained", "Churned"])
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    save_fig(fig, "01_churn_distribution.png", viz_dir)
    plt.show()


def plot_roc_curves(models_probs: dict, y_test: pd.Series, viz_dir: str = "../visualizations"):
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random (AUC = 0.50)")

    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2.5, color=MODEL_COLORS.get(name, "#333"),
                label=f"{name}  (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "06_roc_curves.png", viz_dir)
    plt.show()


def plot_precision_recall_curves(models_probs: dict, y_test: pd.Series,
                                  viz_dir: str = "../visualizations"):
    """Overlay Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1.2, alpha=0.5,
               label=f"Baseline (AP = {baseline:.2f})")

    for name, y_prob in models_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2.5, color=MODEL_COLORS.get(name, "#333"),
                label=f"{name}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "07_precision_recall_curves.png", viz_dir)
    plt.show()


def plot_confusion_matrix(cm_data: np.ndarray, model_name: str,
                           viz_dir: str = "../visualizations", filename: str = None):
    """Annotated confusion matrix heatmap."""
    labels = np.array([
        [f"True Negative\n{cm_data[0,0]:,}", f"False Positive\n{cm_data[0,1]:,}"],
        [f"False Negative\n{cm_data[1,0]:,}", f"True Positive\n{cm_data[1,1]:,}"],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_data, annot=labels, fmt="", cmap="Blues", ax=ax,
                linewidths=2, linecolor="white",
                xticklabels=["Predicted: Retained", "Predicted: Churned"],
                yticklabels=["Actual: Retained", "Actual: Churned"],
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = filename or f"cm_{model_name.lower().replace(' ', '_')}.png"
    save_fig(fig, fname, viz_dir)
    plt.show()


def plot_cost_benefit(df_cba: pd.DataFrame, optimal_threshold: float,
                       viz_dir: str = "../visualizations"):
    """Net business value curve across classification thresholds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_cba["threshold"], df_cba["net_value"] / 1000,
            color=PALETTE["primary"], lw=2.5)
    ax.fill_between(df_cba["threshold"], df_cba["net_value"] / 1000,
                    alpha=0.12, color=PALETTE["primary"])
    ax.axvline(optimal_threshold, color=PALETTE["accent"], linestyle="--",
               lw=2, label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.axhline(0, color=PALETTE["grid"], lw=1)

    opt_row = df_cba.loc[df_cba["threshold"] == optimal_threshold]
    if not opt_row.empty:
        opt_val = opt_row["net_value"].values[0] / 1000
        ax.annotate(f"${opt_val:,.0f}K",
                    xy=(optimal_threshold, opt_val),
                    xytext=(optimal_threshold + 0.06, opt_val),
                    fontsize=11, fontweight="bold", color=PALETTE["accent"],
                    arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Net Business Value ($K)", fontsize=12)
    ax.set_title("Threshold Optimization — Net Business Value", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "09_cost_benefit_threshold.png", viz_dir)
    plt.show()


def plot_feature_importance(importances: pd.Series, model_name: str, top_n: int = 20,
                             viz_dir: str = "../visualizations"):
    """Horizontal bar chart of top N feature importances."""
    top = importances.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    colors = [PALETTE["accent"] if i < 5 else PALETTE["highlight"] for i in range(len(top))]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"08_feature_importance_{model_name.lower().replace(' ', '_')}.png", viz_dir)
    plt.show()


def plot_model_comparison(metrics_list: list, viz_dir: str = "../visualizations"):
    """Grouped bar chart comparing models across ROC-AUC, PR-AUC, and F1."""
    models = [m["model"] for m in metrics_list]
    roc    = [m["roc_auc"] for m in metrics_list]
    pr     = [m["pr_auc"]  for m in metrics_list]
    f1     = [m["f1"]      for m in metrics_list]

    x = np.arange(len(models))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w, roc, w, label="ROC-AUC",  color=PALETTE["highlight"], alpha=0.9)
    b2 = ax.bar(x,     pr,  w, label="PR-AUC",   color=PALETTE["mid"],       alpha=0.9)
    b3 = ax.bar(x + w, f1,  w, label="F1 Score", color=PALETTE["accent"],    alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — ROC-AUC · PR-AUC · F1", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "10_model_comparison.png", viz_dir)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRICS EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_metrics(metrics_dict: dict, output_path: str = "../reports/churn_metrics.json"):
    """Serialize metrics to JSON for downstream reporting."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"✅ Metrics exported → {output_path}")
