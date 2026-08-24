"""
Testes unitários — validam a integridade do pipeline de dados e a
sanidade da camada de inferência (usados como rede de segurança para
re-treinos futuros).

Executar com:  pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from src import config, data_processing, feature_engineering, inference


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    return data_processing.load_raw(config.TRAIN_PATH).head(500)


def test_clean_removes_days_employed_anomaly(raw_sample):
    cleaned = data_processing.clean(raw_sample)
    anomalous_mask = raw_sample["no_of_days_employed"] > config.DAYS_EMPLOYED_ANOMALY_THRESHOLD
    assert cleaned.loc[anomalous_mask, "no_of_days_employed"].isna().all()
    assert cleaned.loc[anomalous_mask, "is_retired_or_unemployed"].eq(1).all()


def test_clean_drops_pii_columns(raw_sample):
    cleaned = data_processing.clean(raw_sample)
    assert "name" not in cleaned.columns


def test_winsor_limits_only_from_train(raw_sample):
    cleaned = data_processing.clean(raw_sample)
    limits = data_processing.compute_winsor_limits(cleaned, ["net_yearly_income"])
    capped = data_processing.apply_winsor_limits(cleaned, limits)
    assert capped["net_yearly_income"].max() <= limits["net_yearly_income"] + 1e-6


def test_domain_features_are_finite(raw_sample):
    cleaned = data_processing.clean(raw_sample)
    enriched = feature_engineering.add_domain_features(cleaned)
    for col in feature_engineering.ENGINEERED_FEATURE_NAMES:
        assert col in enriched.columns
        finite_ratio = np.isfinite(enriched[col].fillna(0)).mean()
        assert finite_ratio == 1.0, f"Coluna {col} contém valores não finitos"


def test_debt_to_income_ratio_is_nonnegative(raw_sample):
    # NaN é esperado aqui (linhas com renda ou dívida ausentes, tratadas
    # depois pelo imputador do pipeline) — o teste valida apenas que
    # nenhum valor CALCULADO é negativo.
    cleaned = data_processing.clean(raw_sample)
    enriched = feature_engineering.add_domain_features(cleaned)
    ratio = enriched["debt_to_income_ratio"].dropna()
    assert (ratio >= 0).all()


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Modelo ainda não treinado")
def test_predict_one_returns_valid_probability():
    record = {
        "age": 35, "gender": "F", "owns_car": "Y", "owns_house": "Y",
        "no_of_children": 1, "net_yearly_income": 150000.0,
        "no_of_days_employed": 1500, "occupation_type": "Core staff",
        "total_family_members": 3, "migrant_worker": 0,
        "yearly_debt_payments": 20000.0, "credit_limit": 40000.0,
        "credit_limit_used(%)": 30, "credit_score": 780,
        "prev_defaults": 0, "default_in_last_6months": 0,
    }
    result = inference.predict_one(record)
    assert 0.0 <= result["probability_of_default"] <= 1.0
    assert result["predicted_class"] in (0, 1)
    assert result["risk_tier"] in {t[2] for t in config.RISK_TIERS}


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Modelo ainda não treinado")
def test_high_risk_profile_scores_higher_than_low_risk_profile():
    low_risk = {
        "age": 40, "gender": "F", "owns_car": "Y", "owns_house": "Y",
        "no_of_children": 0, "net_yearly_income": 300000.0,
        "no_of_days_employed": 3000, "occupation_type": "Managers",
        "total_family_members": 2, "migrant_worker": 0,
        "yearly_debt_payments": 10000.0, "credit_limit": 50000.0,
        "credit_limit_used(%)": 10, "credit_score": 900,
        "prev_defaults": 0, "default_in_last_6months": 0,
    }
    high_risk = {**low_risk, "credit_score": 450, "prev_defaults": 3,
                 "default_in_last_6months": 1, "credit_limit_used(%)": 95}

    p_low = inference.predict_one(low_risk)["probability_of_default"]
    p_high = inference.predict_one(high_risk)["probability_of_default"]
    assert p_high > p_low
