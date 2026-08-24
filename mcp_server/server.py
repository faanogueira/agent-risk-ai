"""
Servidor MCP (Model Context Protocol) — Agente de Risco IA (Agent Risk AI).

Expõe o modelo de inadimplência de cartão de crédito treinado em
`src/train.py` como um conjunto de ferramentas que qualquer cliente MCP
(Claude Desktop, Claude Code, ou outro host compatível) pode chamar
diretamente durante uma conversa — transformando um modelo de ML em
um analista autônomo de inteligência e risco de crédito no time.

Ferramentas expostas:
    - predict_default            -> probabilidade e classe para 1 cliente
    - explain_prediction         -> top fatores (SHAP) por trás do score
    - score_portfolio_csv        -> score em lote de um CSV no disco
    - portfolio_risk_summary     -> agregados de risco de uma carteira
    - get_model_performance      -> métricas de auditoria do modelo
    - what_if_analysis           -> simula o impacto de mudar 1 variável

Executar (stdio, para uso com Claude Desktop/Code):
    python -m mcp_server.server
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.mcpserver import MCPServer

from src import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

mcp = MCPServer(
    name="agent-risk-ai",
    title="Agente de Risco IA",
    version="1.0.0",
    instructions=(
        "Agente de Risco IA — Servidor de inteligência e avaliação de risco de crédito. "
        "Use `predict_default` para avaliar um cliente individual, `explain_prediction` "
        "quando o usuário perguntar 'por quê' um score foi alto, `what_if_analysis` para "
        "simular o impacto de mudanças em variáveis, `score_portfolio_csv` e `portfolio_risk_summary` "
        "para analisar carteiras inteiras a partir de um arquivo CSV, e `get_model_performance` "
        "quando o usuário questionar a confiabilidade e métricas do modelo."
    ),
)


@mcp.tool(
    name="predict_default",
    description=(
        "Prevê a probabilidade de inadimplência (default) de cartão de "
        "crédito de UM cliente a partir dos seus dados cadastrais e de "
        "comportamento de crédito. Retorna probabilidade, classe prevista "
        "e faixa de risco (MUITO_BAIXO a MUITO_ALTO)."
    ),
)
def predict_default(
    age: int,
    gender: str,
    owns_car: str,
    owns_house: str,
    no_of_children: float,
    net_yearly_income: float,
    no_of_days_employed: float,
    occupation_type: str,
    total_family_members: float,
    migrant_worker: float,
    yearly_debt_payments: float,
    credit_limit: float,
    credit_score: float,
    prev_defaults: int,
    default_in_last_6months: int,
    credit_limit_used_pct: float,
) -> dict[str, Any]:
    record = _build_record(
        age, gender, owns_car, owns_house, no_of_children, net_yearly_income,
        no_of_days_employed, occupation_type, total_family_members, migrant_worker,
        yearly_debt_payments, credit_limit, credit_score, prev_defaults,
        default_in_last_6months, credit_limit_used_pct,
    )
    return inference.predict_one(record)


@mcp.tool(
    name="explain_prediction",
    description=(
        "Explica POR QUE o modelo deu determinado score de risco a um "
        "cliente, listando os fatores (via SHAP) que mais aumentaram ou "
        "reduziram a probabilidade de inadimplência prevista. Use quando "
        "o usuário pedir justificativa, transparência ou auditoria do score "
        "(ex.: exigência regulatória de explicabilidade de crédito)."
    ),
)
def explain_prediction(
    age: int,
    gender: str,
    owns_car: str,
    owns_house: str,
    no_of_children: float,
    net_yearly_income: float,
    no_of_days_employed: float,
    occupation_type: str,
    total_family_members: float,
    migrant_worker: float,
    yearly_debt_payments: float,
    credit_limit: float,
    credit_score: float,
    prev_defaults: int,
    default_in_last_6months: int,
    credit_limit_used_pct: float,
    top_k: int = 8,
) -> dict[str, Any]:
    record = _build_record(
        age, gender, owns_car, owns_house, no_of_children, net_yearly_income,
        no_of_days_employed, occupation_type, total_family_members, migrant_worker,
        yearly_debt_payments, credit_limit, credit_score, prev_defaults,
        default_in_last_6months, credit_limit_used_pct,
    )
    return inference.explain_one(record, top_k=top_k)


@mcp.tool(
    name="score_portfolio_csv",
    description=(
        "Aplica o modelo a TODOS os clientes de um arquivo CSV no disco "
        "(mesmo schema do dataset de treino, sem a coluna alvo) e retorna "
        "uma amostra dos resultados mais uma contagem por faixa de risco. "
        "Use para analisar uma carteira inteira de uma vez, em vez de "
        "cliente por cliente."
    ),
)
def score_portfolio_csv(csv_path: str, sample_size: int = 10) -> dict[str, Any]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        return {"error": f"Arquivo não encontrado: {path}"}

    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    results = inference.score_batch(records)

    df_results = pd.DataFrame(results)
    tier_counts = df_results["risk_tier"].value_counts().to_dict()

    ids = df["customer_id"].tolist() if "customer_id" in df.columns else list(range(len(df)))
    for r, cid in zip(results, ids):
        r["customer_id"] = cid

    return {
        "total_scored": len(results),
        "risk_tier_distribution": tier_counts,
        "mean_probability_of_default": round(df_results["probability_of_default"].mean(), 4),
        "sample_results": results[:sample_size],
    }


@mcp.tool(
    name="portfolio_risk_summary",
    description=(
        "Gera um resumo executivo de risco de uma carteira de clientes a "
        "partir de um CSV: perda esperada aproximada (exposição x PD), "
        "distribuição por faixa de risco e top clientes de maior risco. "
        "Use para relatórios gerenciais ou dashboards de risco de carteira."
    ),
)
def portfolio_risk_summary(csv_path: str, top_n_riskiest: int = 5) -> dict[str, Any]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        return {"error": f"Arquivo não encontrado: {path}"}

    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    results = inference.score_batch(records)
    df_results = pd.DataFrame(results)

    exposure = df["credit_limit"] if "credit_limit" in df.columns else pd.Series([0] * len(df))
    df_results["exposure"] = exposure.values
    # Perda esperada simplificada = Probabilidade de Default x Exposição
    # (assume LGD = 100% por simplicidade; refine com dados de recuperação reais)
    df_results["expected_loss"] = df_results["probability_of_default"] * df_results["exposure"]

    if "customer_id" in df.columns:
        df_results["customer_id"] = df["customer_id"].values

    riskiest = df_results.sort_values("probability_of_default", ascending=False).head(top_n_riskiest)
    riskiest_cols = [c for c in ["customer_id", "probability_of_default", "risk_tier", "exposure"] if c in riskiest.columns]

    return {
        "n_customers": len(df_results),
        "total_exposure": round(float(df_results["exposure"].sum()), 2),
        "total_expected_loss": round(float(df_results["expected_loss"].sum()), 2),
        "expected_loss_rate_pct": round(
            100 * df_results["expected_loss"].sum() / max(df_results["exposure"].sum(), 1e-9), 3
        ),
        "risk_tier_distribution": df_results["risk_tier"].value_counts().to_dict(),
        "top_riskiest_customers": riskiest[riskiest_cols].to_dict(orient="records"),
    }


@mcp.tool(
    name="get_model_performance",
    description=(
        "Retorna as métricas de auditoria e performance do modelo em "
        "produção: ROC-AUC, PR-AUC, F1, matriz de confusão, threshold de "
        "decisão, hiperparâmetros e principais features (via SHAP). Use "
        "quando o usuário perguntar 'o modelo é confiável?', 'qual a "
        "acurácia?' ou pedir a ficha técnica do modelo."
    ),
)
def get_model_performance() -> dict[str, Any]:
    return inference.get_model_info()


@mcp.tool(
    name="what_if_analysis",
    description=(
        "Simula como a probabilidade de default de UM cliente mudaria se "
        "uma única variável fosse diferente (ex.: 'e se o limite de "
        "crédito usado caísse para 30%?'). Útil para simular políticas de "
        "concessão ou orientar um cliente sobre como reduzir seu risco."
    ),
)
def what_if_analysis(
    age: int,
    gender: str,
    owns_car: str,
    owns_house: str,
    no_of_children: float,
    net_yearly_income: float,
    no_of_days_employed: float,
    occupation_type: str,
    total_family_members: float,
    migrant_worker: float,
    yearly_debt_payments: float,
    credit_limit: float,
    credit_score: float,
    prev_defaults: int,
    default_in_last_6months: int,
    credit_limit_used_pct: float,
    field_to_change: str,
    new_value: float,
) -> dict[str, Any]:
    baseline_record = _build_record(
        age, gender, owns_car, owns_house, no_of_children, net_yearly_income,
        no_of_days_employed, occupation_type, total_family_members, migrant_worker,
        yearly_debt_payments, credit_limit, credit_score, prev_defaults,
        default_in_last_6months, credit_limit_used_pct,
    )
    if field_to_change not in baseline_record:
        return {"error": f"Campo desconhecido: {field_to_change}. Campos válidos: {list(baseline_record.keys())}"}

    scenario_record = dict(baseline_record)
    scenario_record[field_to_change] = new_value

    baseline_result = inference.predict_one(baseline_record)
    scenario_result = inference.predict_one(scenario_record)

    delta = scenario_result["probability_of_default"] - baseline_result["probability_of_default"]
    return {
        "field_changed": field_to_change,
        "original_value": baseline_record[field_to_change],
        "new_value": new_value,
        "baseline_probability_of_default": baseline_result["probability_of_default"],
        "scenario_probability_of_default": scenario_result["probability_of_default"],
        "delta": round(delta, 4),
        "interpretation": "Risco AUMENTOU" if delta > 0 else ("Risco DIMINUIU" if delta < 0 else "Sem alteração"),
    }


def _build_record(
    age, gender, owns_car, owns_house, no_of_children, net_yearly_income,
    no_of_days_employed, occupation_type, total_family_members, migrant_worker,
    yearly_debt_payments, credit_limit, credit_score, prev_defaults,
    default_in_last_6months, credit_limit_used_pct,
) -> dict[str, Any]:
    return {
        "age": age,
        "gender": gender,
        "owns_car": owns_car,
        "owns_house": owns_house,
        "no_of_children": no_of_children,
        "net_yearly_income": net_yearly_income,
        "no_of_days_employed": no_of_days_employed,
        "occupation_type": occupation_type,
        "total_family_members": total_family_members,
        "migrant_worker": migrant_worker,
        "yearly_debt_payments": yearly_debt_payments,
        "credit_limit": credit_limit,
        "credit_limit_used(%)": credit_limit_used_pct,
        "credit_score": credit_score,
        "prev_defaults": prev_defaults,
        "default_in_last_6months": default_in_last_6months,
    }


def main() -> None:
    logger.info("Iniciando servidor MCP 'agent-risk-ai' (Agente de Risco IA) [stdio]...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
