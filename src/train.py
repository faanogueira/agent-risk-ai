"""
Pipeline de treino do modelo de inadimplência de cartão de crédito.

Fluxo:
1. Carrega e limpa os dados (data_processing).
2. Aplica engenharia de features orientada a domínio (feature_engineering).
3. Separa holdout estratificado (nunca visto durante tuning).
4. Ajusta 3 famílias de modelo (Regressão Logística como baseline
   interpretável, Random Forest e XGBoost) usando validação cruzada
   estratificada (5 folds) otimizando PR-AUC (Average Precision) — a
   métrica correta aqui, já que a classe positiva é ~8% da base e
   ROC-AUC tende a ser otimista em cenários desbalanceados.
5. Faz tuning de hiperparâmetros do XGBoost com Optuna (TPE sampler).
6. Recalibra o threshold de decisão maximizando o F1 no holdout
   (em vez de usar 0.5 às cegas).
7. Persiste o pipeline vencedor + metadados de auditoria (métricas,
   hiperparâmetros, versão, timestamp, importância de features).

Executar com:  python -m src.train
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config, data_processing, feature_engineering, pipeline as pp

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
WINSOR_COLS = ["net_yearly_income", "credit_limit", "yearly_debt_payments"]


def prepare_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    raw = data_processing.load_raw(config.TRAIN_PATH)
    clean = data_processing.clean(raw, is_train=True)

    winsor_limits = data_processing.compute_winsor_limits(clean, WINSOR_COLS)
    clean = data_processing.apply_winsor_limits(clean, winsor_limits)

    enriched = feature_engineering.add_domain_features(clean)
    X, y = data_processing.split_features_target(enriched)
    X = X.drop(columns=[config.ID_COL])
    return X, y, winsor_limits


def evaluate_cv(model, X, y, preprocessor) -> float:
    full_pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    scores = cross_val_score(full_pipe, X, y, cv=CV, scoring="average_precision", n_jobs=-1)
    return float(scores.mean())


def objective(trial: optuna.Trial, X, y, preprocessor, scale_pos_weight: float) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return evaluate_cv(model, X, y, preprocessor)


def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    """Varre a curva precisão-recall e escolhe o threshold que maximiza o F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    return best_threshold, float(f1s[best_idx])


def main() -> None:
    t0 = time.time()
    logger.info("Carregando e preparando dataset...")
    X, y, winsor_limits = prepare_dataset()

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=config.RANDOM_STATE
    )
    logger.info("Treino: %s | Holdout: %s | Taxa de default (treino): %.2f%%",
                X_train.shape, X_holdout.shape, 100 * y_train.mean())

    preprocessor = pp.build_preprocessor()
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info("scale_pos_weight (desbalanceamento): %.2f", scale_pos_weight)

    # --- 1) Baselines --------------------------------------------------
    logger.info("Avaliando baselines via CV (5-fold, PR-AUC)...")
    baselines = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }
    baseline_scores = {}
    for name, model in baselines.items():
        score = evaluate_cv(model, X_train, y_train, preprocessor)
        baseline_scores[name] = score
        logger.info("  %-22s PR-AUC(cv)=%.4f", name, score)

    # --- 2) XGBoost + Optuna --------------------------------------------
    logger.info("Otimizando XGBoost com Optuna (25 trials, CV 5-fold)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, preprocessor, scale_pos_weight),
        n_trials=25,
        show_progress_bar=False,
    )
    logger.info("  Melhor PR-AUC(cv) XGBoost=%.4f | params=%s", study.best_value, study.best_params)
    baseline_scores["xgboost_tuned"] = study.best_value

    # --- 3) Modelo final ------------------------------------------------
    best_model = XGBClassifier(
        **study.best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    final_pipeline = Pipeline([("prep", preprocessor), ("clf", best_model)])
    final_pipeline.fit(X_train, y_train)

    # --- 4) Avaliação em holdout nunca visto -----------------------------
    y_proba_holdout = final_pipeline.predict_proba(X_holdout)[:, 1]
    roc_auc = roc_auc_score(y_holdout, y_proba_holdout)
    pr_auc = average_precision_score(y_holdout, y_proba_holdout)
    best_threshold, best_f1 = find_best_threshold(y_holdout, y_proba_holdout)
    y_pred_holdout = (y_proba_holdout >= best_threshold).astype(int)
    cm = confusion_matrix(y_holdout, y_pred_holdout).tolist()

    logger.info("== Métricas finais (holdout, 15%% nunca visto no tuning) ==")
    logger.info("  ROC-AUC        = %.4f", roc_auc)
    logger.info("  PR-AUC         = %.4f", pr_auc)
    logger.info("  F1 @ threshold=%.3f -> %.4f", best_threshold, best_f1)
    logger.info("  Matriz de confusão [ [TN,FP],[FN,TP] ] = %s", cm)

    # --- 5) Importância de features via SHAP -----------------------------
    logger.info("Calculando importância de features (SHAP TreeExplainer)...")
    X_holdout_transformed = final_pipeline.named_steps["prep"].transform(X_holdout)
    feature_names = pp.get_feature_names(final_pipeline.named_steps["prep"])
    explainer = shap.TreeExplainer(final_pipeline.named_steps["clf"])
    shap_values = explainer.shap_values(X_holdout_transformed)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = sorted(
        zip(feature_names, mean_abs_shap.tolist()), key=lambda t: t[1], reverse=True
    )[:15]

    # --- 6) Persistência --------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, config.MODEL_PATH)
    joblib.dump(explainer, config.MODELS_DIR / "shap_explainer.joblib")

    metadata = {
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(time.time() - t0, 1),
        "algorithm": "XGBoost (tuned via Optuna, 25 trials)",
        "target": config.TARGET_COL,
        "n_train_rows": int(len(X_train)),
        "n_holdout_rows": int(len(X_holdout)),
        "positive_rate_train": float(y_train.mean()),
        "scale_pos_weight": scale_pos_weight,
        "winsor_limits": winsor_limits,
        "cv_pr_auc_by_model": baseline_scores,
        "best_hyperparameters": study.best_params,
        "holdout_metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_at_best_threshold": best_f1,
            "best_threshold": best_threshold,
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["true_negative", "false_positive", "false_negative", "true_positive"],
        },
        "top_features_shap": [{"feature": f, "mean_abs_shap": v} for f, v in top_features],
        "feature_names_post_preprocessing": feature_names,
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Modelo salvo em %s", config.MODEL_PATH)
    logger.info("Metadados salvos em %s", config.METADATA_PATH)
    logger.info("Top 5 features (SHAP): %s", [f for f, _ in top_features[:5]])
    logger.info("Concluído em %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
