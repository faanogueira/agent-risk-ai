"""
Gera artefatos visuais de avaliação do modelo campeão para o relatório
(reports/figures/): curva ROC, curva Precision-Recall, matriz de
confusão e importância de features via SHAP.

Executar com:  python -m src.evaluate
"""
from __future__ import annotations

import json
import logging

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from src import config, data_processing, feature_engineering, train as train_mod

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(config.MODEL_PATH)
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    X, y, winsor_limits = train_mod.prepare_dataset()
    _, X_holdout, _, y_holdout = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=config.RANDOM_STATE
    )

    y_proba = model.predict_proba(X_holdout)[:, 1]
    threshold = metadata["holdout_metrics"]["best_threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    # --- ROC ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_holdout, y_proba, ax=ax, name="XGBoost (tuned)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    ax.set_title("Curva ROC — Holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curve.png")
    plt.close(fig)

    # --- Precision-Recall --------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_holdout, y_proba, ax=ax, name="XGBoost (tuned)")
    ax.axhline(y=y_holdout.mean(), linestyle="--", color="gray", label="Baseline (prevalência)")
    ax.set_title("Curva Precisão-Recall — Holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "precision_recall_curve.png")
    plt.close(fig)

    # --- Matriz de confusão --------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_holdout, y_pred, display_labels=["Adimplente", "Inadimplente"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Matriz de Confusão @ threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png")
    plt.close(fig)

    # --- SHAP summary ---------------------------------------------------
    explainer = joblib.load(config.MODELS_DIR / "shap_explainer.joblib")
    X_transformed = model.named_steps["prep"].transform(X_holdout)
    from src import pipeline as pp
    feature_names = pp.get_feature_names(model.named_steps["prep"])
    shap_values = explainer.shap_values(X_transformed)

    fig = plt.figure(figsize=(7, 6))
    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names, show=False, max_display=15
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close(fig)

    logger.info("Figuras salvas em %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
