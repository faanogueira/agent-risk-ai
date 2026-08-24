"""
Carregamento e limpeza de dados.

Responsabilidade única: transformar o CSV bruto em um DataFrame confiável,
tratando os problemas de qualidade de dados identificados na análise
exploratória (EDA) — sem ainda criar features derivadas (isso fica a
cargo de `feature_engineering.py`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Carrega um CSV bruto (train ou test)."""
    df = pd.read_csv(path)
    logger.info("Carregado %s com shape=%s", path.name, df.shape)
    return df


def clean(df: pd.DataFrame, *, is_train: bool = True) -> pd.DataFrame:
    """
    Aplica a limpeza de qualidade de dados descoberta na EDA:

    1. Remove colunas sem sinal preditivo (PII).
    2. Corrige o valor-sentinela de `no_of_days_employed` (~365243 dias),
       que na prática marca aposentados/não empregados — não é literal.
    3. Normaliza a categoria residual de `gender` ("XNA" -> NaN, tratada
       depois pelo imputador).
    4. Winsoriza (capa) variáveis monetárias de cauda longa extrema
       (`net_yearly_income`, `credit_limit`, `yearly_debt_payments`)
       no percentil configurado, evitando que outliers de poucas
       observações dominem o treino de modelos baseados em árvore.
    5. Garante tipos numéricos coerentes.

    Os limites de winsorização são aprendidos apenas no conjunto de
    treino (evita vazamento de dados) e devem ser reaplicados no teste
    via `apply_winsor_limits`.
    """
    df = df.copy()

    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    # 1) Sentinela de dias empregados -> vira NaN + flag booleana explícita
    is_anomalous = df["no_of_days_employed"] > config.DAYS_EMPLOYED_ANOMALY_THRESHOLD
    df["is_retired_or_unemployed"] = is_anomalous.astype(int)
    df.loc[is_anomalous, "no_of_days_employed"] = np.nan

    # 2) Categoria residual em gender
    df["gender"] = df["gender"].replace(list(config.RARE_GENDER_VALUES), np.nan)

    # 3) Tipos
    for col in ["no_of_children", "total_family_members", "migrant_worker"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_winsor_limits(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Aprende os limites superiores de winsorização SOMENTE no treino."""
    return {c: float(df[c].quantile(config.INCOME_CAP_QUANTILE)) for c in cols if c in df.columns}


def apply_winsor_limits(df: pd.DataFrame, limits: dict[str, float]) -> pd.DataFrame:
    """Aplica limites já aprendidos (treino ou teste) — sem vazamento."""
    df = df.copy()
    for col, cap in limits.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separa X e y (y é None se a coluna alvo não existir, ex.: test.csv)."""
    y = df[config.TARGET_COL] if config.TARGET_COL in df.columns else None
    X = df.drop(columns=[c for c in [config.TARGET_COL] if c in df.columns])
    return X, y
