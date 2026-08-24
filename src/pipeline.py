"""
Monta o ColumnTransformer de pré-processamento.

Mantemos o pré-processamento DENTRO do Pipeline do sklearn (e não como
passos manuais soltos) por um motivo central de rigor metodológico:
isso garante que imputação/encoding sejam "fit" apenas nos folds de
treino durante a validação cruzada, eliminando vazamento de dados
(data leakage) entre treino e validação.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_IMPUTE_MEDIAN = [
    "age",
    "no_of_children",
    "net_yearly_income",
    "no_of_days_employed",
    "total_family_members",
    "migrant_worker",
    "yearly_debt_payments",
    "credit_limit",
    "credit_limit_used(%)",
    "credit_score",
    "prev_defaults",
    "default_in_last_6months",
    "is_retired_or_unemployed",
    "debt_to_income_ratio",
    "credit_limit_to_income_ratio",
    "credit_utilization_frac",
    "utilization_x_prev_defaults",
    "income_per_family_member",
    "has_children",
    "employment_years",
    "employment_tenure_ratio",
    "has_prev_default",
    "risk_flags_sum",
    "log_net_yearly_income",
    "log_credit_limit",
    "log_yearly_debt_payments",
]

CATEGORICAL_COLS = ["gender", "owns_car", "owns_house", "occupation_type"]


def build_preprocessor(numeric_cols: list[str] | None = None) -> ColumnTransformer:
    numeric_cols = numeric_cols or NUMERIC_IMPUTE_MEDIAN

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extrai nomes de features pós-transformação (para SHAP/importância)."""
    return list(preprocessor.get_feature_names_out())
