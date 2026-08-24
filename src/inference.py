"""
Camada de inferência: carrega o modelo treinado uma única vez e expõe
funções puras de predição/explicação reutilizadas pelo servidor MCP.

Mantida separada do `mcp_server/` de propósito: a lógica de ML não deve
depender do protocolo de transporte (MCP, REST, CLI, etc.).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config, data_processing, feature_engineering, pipeline as pp

REQUIRED_INPUT_FIELDS = [
    "age", "gender", "owns_car", "owns_house", "no_of_children",
    "net_yearly_income", "no_of_days_employed", "occupation_type",
    "total_family_members", "migrant_worker", "yearly_debt_payments",
    "credit_limit", "credit_limit_used(%)", "credit_score",
    "prev_defaults", "default_in_last_6months",
]


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    if not config.MODEL_PATH.exists() or not config.METADATA_PATH.exists():
        raise ModelNotTrainedError(
            f"Modelo não encontrado em {config.MODEL_PATH}. Execute "
            "`python -m src.train` antes de iniciar o servidor MCP."
        )
    model = joblib.load(config.MODEL_PATH)
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    explainer_path = config.MODELS_DIR / "shap_explainer.joblib"
    explainer = joblib.load(explainer_path) if explainer_path.exists() else None
    return {"model": model, "metadata": metadata, "explainer": explainer}


def _prepare_single_record(record: dict[str, Any]) -> pd.DataFrame:
    """Aplica exatamente o mesmo pipeline de limpeza/features do treino
    a um único registro (dict) vindo de uma chamada de ferramenta MCP."""
    missing = [f for f in REQUIRED_INPUT_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes: {missing}")

    df = pd.DataFrame([record])
    df[config.ID_COL] = "AD_HOC"

    artifacts = load_artifacts()
    winsor_limits = artifacts["metadata"]["winsor_limits"]

    df = data_processing.clean(df, is_train=False)
    df = data_processing.apply_winsor_limits(df, winsor_limits)
    df = feature_engineering.add_domain_features(df)
    X, _ = data_processing.split_features_target(df)
    X = X.drop(columns=[config.ID_COL])
    return X


def risk_tier(probability: float) -> str:
    for low, high, label in config.RISK_TIERS:
        if low <= probability < high:
            return label
    return "INDEFINIDO"


def predict_one(record: dict[str, Any]) -> dict[str, Any]:
    artifacts = load_artifacts()
    model = artifacts["model"]
    threshold = artifacts["metadata"]["holdout_metrics"]["best_threshold"]

    X = _prepare_single_record(record)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "probability_of_default": round(probability, 4),
        "predicted_class": prediction,
        "predicted_label": "INADIMPLENTE" if prediction == 1 else "ADIMPLENTE",
        "decision_threshold_used": threshold,
        "risk_tier": risk_tier(probability),
    }


def explain_one(record: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
    artifacts = load_artifacts()
    model = artifacts["model"]
    explainer = artifacts["explainer"]
    if explainer is None:
        raise ModelNotTrainedError("Explainer SHAP não encontrado. Re-treine o modelo.")

    X = _prepare_single_record(record)
    X_transformed = model.named_steps["prep"].transform(X)
    feature_names = pp.get_feature_names(model.named_steps["prep"])

    shap_values = explainer.shap_values(X_transformed)[0]
    base_value = float(np.array(explainer.expected_value).reshape(-1)[0])

    contributions = sorted(
        zip(feature_names, shap_values.tolist(), X_transformed[0].tolist()),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:top_k]

    factors = [
        {
            "feature": name.replace("num__", "").replace("cat__", ""),
            "value_after_preprocessing": round(value, 4),
            "shap_contribution": round(shap_val, 4),
            "effect": "AUMENTA risco de default" if shap_val > 0 else "REDUZ risco de default",
        }
        for name, shap_val, value in contributions
    ]

    prediction = predict_one(record)
    return {
        **prediction,
        "base_rate_log_odds": round(base_value, 4),
        "top_factors": factors,
    }


def score_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [predict_one(r) for r in records]


def get_model_info() -> dict[str, Any]:
    artifacts = load_artifacts()
    metadata = artifacts["metadata"]
    return {
        "model_version": metadata["model_version"],
        "algorithm": metadata["algorithm"],
        "trained_at_utc": metadata["trained_at_utc"],
        "n_train_rows": metadata["n_train_rows"],
        "positive_rate_train": metadata["positive_rate_train"],
        "holdout_metrics": metadata["holdout_metrics"],
        "cv_pr_auc_by_model": metadata["cv_pr_auc_by_model"],
        "top_features_shap": metadata["top_features_shap"][:10],
        "risk_tiers": config.RISK_TIERS,
    }
