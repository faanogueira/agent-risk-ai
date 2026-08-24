"""
Engenharia de features orientada a domínio (risco de crédito).

Cada feature aqui tem uma justificativa de negócio explícita — o
objetivo é ir além de "jogar tudo no modelo" e construir variáveis que
um analista de risco reconheceria como relevantes para inadimplência.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    # --- Alavancagem e comprometimento de renda ---------------------------
    # DTI (Debt-to-Income): quanto da renda anual é comprometida com dívida.
    df["debt_to_income_ratio"] = df["yearly_debt_payments"] / (df["net_yearly_income"] + eps)

    # Quanto do limite total de crédito concedido representa em relação à renda.
    df["credit_limit_to_income_ratio"] = df["credit_limit"] / (df["net_yearly_income"] + eps)

    # Uso do limite (%) já vem pronto, mas criamos a versão em fração
    # e uma interação com o histórico de inadimplência.
    df["credit_utilization_frac"] = df["credit_limit_used(%)"] / 100.0
    df["utilization_x_prev_defaults"] = df["credit_utilization_frac"] * df["prev_defaults"]

    # --- Estabilidade financeira / demográfica ------------------------------
    df["income_per_family_member"] = df["net_yearly_income"] / (df["total_family_members"] + eps)
    df["has_children"] = (df["no_of_children"].fillna(0) > 0).astype(int)

    # Tempo de emprego em anos (mais interpretável que dias); NaN indica
    # aposentado/não empregado, tratado por `is_retired_or_unemployed`.
    df["employment_years"] = df["no_of_days_employed"] / 365.25

    # Idade em que a pessoa começou a trabalhar no emprego atual — tenure
    # muito curto relativo à idade pode indicar instabilidade.
    df["employment_tenure_ratio"] = df["employment_years"] / (df["age"] + eps)

    # --- Sinais de risco já observado -------------------------------------
    df["has_prev_default"] = (df["prev_defaults"] > 0).astype(int)
    df["risk_flags_sum"] = (
        df["has_prev_default"]
        + df["default_in_last_6months"]
        + (df["credit_utilization_frac"] > 0.8).astype(int)
    )

    # --- Transformações de escala --------------------------------------
    # Rendas e limites de crédito são fortemente assimétricos à direita;
    # log1p estabiliza a variância e ajuda modelos lineares/kNN (as
    # árvores são invariantes a isso, mas mantemos por completude e para
    # permitir comparação justa com baselines lineares).
    for col in ["net_yearly_income", "credit_limit", "yearly_debt_payments"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


ENGINEERED_FEATURE_NAMES = [
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
