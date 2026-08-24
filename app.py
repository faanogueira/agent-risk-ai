"""
Agente de Risco IA — Interface Web Chat Conversacional (Navegador)

Permite interagir com o modelo de risco de crédito e suas ferramentas MCP
diretamente pelo navegador em linguagem natural.

Executar com:
    streamlit run app.py
ou
    make web
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import config, inference

# --- Configurações da Página -------------------------------------------
st.set_page_config(
    page_title="Agente de Risco IA — Chat Conversacional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilização CSS Customizada --------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .risk-badge {
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-muito-baixo { background-color: #D1FAE5; color: #065F46; }
    .risk-baixo { background-color: #DEF7EC; color: #03543F; }
    .risk-moderado { background-color: #FEF3C7; color: #92400E; }
    .risk-alto { background-color: #FEE2E2; color: #991B1B; }
    .risk-muito-alto { background-color: #7F1D1D; color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Perfis de Exemplo para Teste Rápido (Todas as 5 Faixas de Risco) ---
SAMPLE_PROFILES = {
    "🟢 1. Muito Baixo (<5%) — Cliente Prime": {
        "age": 45,
        "gender": "M",
        "owns_car": "Y",
        "owns_house": "Y",
        "no_of_children": 0.0,
        "net_yearly_income": 350000.0,
        "no_of_days_employed": 3800.0,
        "occupation_type": "Managers",
        "total_family_members": 2.0,
        "migrant_worker": 0.0,
        "yearly_debt_payments": 10000.0,
        "credit_limit": 60000.0,
        "credit_limit_used(%)": 10.0,
        "credit_score": 910.0,
        "prev_defaults": 0,
        "default_in_last_6months": 0,
    },
    "🟢 2. Baixo (5–15%) — Cliente Saudável": {
        "age": 38,
        "gender": "F",
        "owns_car": "Y",
        "owns_house": "Y",
        "no_of_children": 1.0,
        "net_yearly_income": 200000.0,
        "no_of_days_employed": 2400.0,
        "occupation_type": "High skill tech staff",
        "total_family_members": 3.0,
        "migrant_worker": 0.0,
        "yearly_debt_payments": 22000.0,
        "credit_limit": 40000.0,
        "credit_limit_used(%)": 25.0,
        "credit_score": 810.0,
        "prev_defaults": 0,
        "default_in_last_6months": 0,
    },
    "🟡 3. Moderado (15–35%) — Cliente Limítrofe": {
        "age": 35,
        "gender": "M",
        "owns_car": "N",
        "owns_house": "Y",
        "no_of_children": 0.0,
        "net_yearly_income": 100000.0,
        "no_of_days_employed": 1500.0,
        "occupation_type": "Laborers",
        "total_family_members": 2.0,
        "migrant_worker": 0.0,
        "yearly_debt_payments": 20000.0,
        "credit_limit": 30000.0,
        "credit_limit_used(%)": 50.0,
        "credit_score": 580.0,
        "prev_defaults": 0,
        "default_in_last_6months": 0,
    },
    "🔴 4. Alto (35–60%) — Cliente Alerta": {
        "age": 35,
        "gender": "M",
        "owns_car": "N",
        "owns_house": "Y",
        "no_of_children": 0.0,
        "net_yearly_income": 100000.0,
        "no_of_days_employed": 1500.0,
        "occupation_type": "Laborers",
        "total_family_members": 2.0,
        "migrant_worker": 0.0,
        "yearly_debt_payments": 20000.0,
        "credit_limit": 30000.0,
        "credit_limit_used(%)": 50.0,
        "credit_score": 580.0,
        "prev_defaults": 1,
        "default_in_last_6months": 0,
    },
    "⛔ 5. Muito Alto (≥60%) — Cliente Crítico": {
        "age": 46,
        "gender": "F",
        "owns_car": "N",
        "owns_house": "Y",
        "no_of_children": 0.0,
        "net_yearly_income": 107934.04,
        "no_of_days_employed": 612.0,
        "occupation_type": "Laborers",
        "total_family_members": 1.0,
        "migrant_worker": 1.0,
        "yearly_debt_payments": 33070.28,
        "credit_limit": 18690.93,
        "credit_limit_used(%)": 73.0,
        "credit_score": 544.0,
        "prev_defaults": 2,
        "default_in_last_6months": 1,
    },
}


def get_risk_badge_html(tier: str, prob: float) -> str:
    css_map = {
        "MUITO_BAIXO": "risk-muito-baixo",
        "BAIXO": "risk-baixo",
        "MODERADO": "risk-moderado",
        "ALTO": "risk-alto",
        "MUITO_ALTO": "risk-muito-alto",
    }
    css_class = css_map.get(tier, "risk-moderado")
    return f'<span class="risk-badge {css_class}">Faixa: {tier} | Probabilidade de Default: {prob*100:.2f}%</span>'


def parse_and_execute_tool(query: str, last_record: dict[str, Any] | None) -> dict[str, Any]:
    """
    Roteador inteligente local de linguagem natural para ferramentas MCP.
    Mapeia intenções do usuário em chamadas diretas das funções de inferência.
    """
    q_lower = query.lower()

    # 1) Intenção: Ficha técnica / Performance do modelo
    if any(w in q_lower for w in ["performance", "acurácia", "métrica", "ficha técnica", "confiabilidade", "modelo"]):
        info = inference.get_model_info()
        return {"tool": "get_model_performance", "data": info}

    # 2) Intenção: Análise de Carteira / Portfólio
    if any(w in q_lower for w in ["carteira", "portfolio", "portfólio", "csv", "lote", "perda esperada"]):
        # Tenta achar caminho de arquivo na query ou usa test.csv padrão
        csv_match = re.search(r'[\w\-./]+\.csv', query)
        csv_path = Path(csv_match.group(0)) if csv_match else config.TEST_PATH
        if not csv_path.exists():
            csv_path = config.TEST_PATH
        
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient="records")
        results = inference.score_batch(records)
        df_results = pd.DataFrame(results)
        
        exposure = df["credit_limit"] if "credit_limit" in df.columns else pd.Series([0] * len(df))
        df_results["exposure"] = exposure.values
        df_results["expected_loss"] = df_results["probability_of_default"] * df_results["exposure"]
        if "customer_id" in df.columns:
            df_results["customer_id"] = df["customer_id"].values

        riskiest = df_results.sort_values("probability_of_default", ascending=False).head(5)
        
        summary = {
            "csv_path": str(csv_path),
            "n_customers": len(df_results),
            "total_exposure": float(df_results["exposure"].sum()),
            "total_expected_loss": float(df_results["expected_loss"].sum()),
            "expected_loss_rate_pct": float(100 * df_results["expected_loss"].sum() / max(df_results["exposure"].sum(), 1e-9)),
            "risk_tier_distribution": df_results["risk_tier"].value_counts().to_dict(),
            "top_riskiest": riskiest[["customer_id", "probability_of_default", "risk_tier", "exposure"]].to_dict(orient="records") if "customer_id" in riskiest else [],
        }
        return {"tool": "portfolio_risk_summary", "data": summary}

    # 3) Intenção: Simulação What-If
    if any(w in q_lower for w in ["e se", "what if", "what-if", "simulação", "simular", "reduzir", "aumentar", "mudar"]):
        record = last_record or SAMPLE_PROFILES["Cliente 1 — Alto Risco"]
        
        # Identifica campo e novo valor
        field_to_change = "credit_limit_used(%)"
        new_val = 30.0
        
        val_match = re.search(r'(\d+[\.,]?\d*)', query)
        if val_match:
            new_val = float(val_match.group(1).replace(",", "."))
            
        if "score" in q_lower:
            field_to_change = "credit_score"
        elif "renda" in q_lower or "income" in q_lower:
            field_to_change = "net_yearly_income"
        elif "dívida" in q_lower or "debt" in q_lower:
            field_to_change = "yearly_debt_payments"
        elif "inadimplência" in q_lower or "default" in q_lower:
            field_to_change = "prev_defaults"
            new_val = int(new_val)

        baseline_res = inference.predict_one(record)
        scenario_record = dict(record)
        scenario_record[field_to_change] = new_val
        scenario_res = inference.predict_one(scenario_record)
        
        delta = scenario_res["probability_of_default"] - baseline_res["probability_of_default"]
        what_if_data = {
            "field_changed": field_to_change,
            "original_value": record.get(field_to_change, "N/A"),
            "new_value": new_val,
            "baseline_prob": baseline_res["probability_of_default"],
            "scenario_prob": scenario_res["probability_of_default"],
            "baseline_tier": baseline_res["risk_tier"],
            "scenario_tier": scenario_res["risk_tier"],
            "delta": delta,
            "updated_record": scenario_record,
        }
        return {"tool": "what_if_analysis", "data": what_if_data}

    # 4) Padrão: Avaliação individual de Cliente com SHAP
    target_record = last_record or SAMPLE_PROFILES["Cliente 1 — Alto Risco"]
    
    # Extração de parâmetros fornecidos na query
    rec = dict(target_record)
    age_m = re.search(r'idade[:\s]+(\d+)', q_lower)
    if age_m: rec["age"] = int(age_m.group(1))
    
    score_m = re.search(r'score[:\s]+(\d+)', q_lower)
    if score_m: rec["credit_score"] = float(score_m.group(1))
    
    income_m = re.search(r'renda[:\s]+(?:r\$\s*)?([\d\.]+)', q_lower)
    if income_m: rec["net_yearly_income"] = float(income_m.group(1).replace(".", ""))
    
    used_m = re.search(r'(?:limite usado|utiliza[çc][ãa]o)[:\s]+(\d+)', q_lower)
    if used_m: rec["credit_limit_used(%)"] = float(used_m.group(1))
    
    defaults_m = re.search(r'(\d+)\s*(?:inadimpl[êe]ncias?|defaults?)', q_lower)
    if defaults_m: rec["prev_defaults"] = int(defaults_m.group(1))

    explanation = inference.explain_one(rec, top_k=8)
    explanation["record_evaluated"] = rec
    return {"tool": "explain_prediction", "data": explanation}


# --- Barra Lateral (Controles e Ações Rápidas) -------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Agente%20de%20Risco%20IA-MCP%20Server-1E3A8A?style=for-the-badge&logo=probot&logoColor=white", use_container_width=True)
    st.markdown("### ⚡ Ações Rápidas")
    st.caption("Selecione um cliente para avaliar instantaneamente nas **5 faixas de risco**:")
    
    selected_sample = None
    for name in SAMPLE_PROFILES:
        if st.button(f"{name}", use_container_width=True):
            selected_sample = name

    st.divider()
    st.markdown("### 🛠️ Consultas Sugeridas")
    st.caption("Ações rápidas de auditoria, carteira e simulação *What-If*:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Ficha Técnica", use_container_width=True):
            st.session_state["pending_prompt"] = "Qual a performance e métricas de validação do modelo?"
        if st.button("📉 Limite a 30%", use_container_width=True):
            st.session_state["pending_prompt"] = "E se o limite de crédito usado for reduzido para 30%?"
        if st.button("💰 Quitar Dívida", use_container_width=True):
            st.session_state["pending_prompt"] = "E se a dívida anual for reduzida para R$ 5.000?"
    with col2:
        if st.button("📁 Carteira CSV", use_container_width=True):
            st.session_state["pending_prompt"] = "Gere um resumo de risco e perda esperada da carteira de teste."
        if st.button("📈 Score para 780", use_container_width=True):
            st.session_state["pending_prompt"] = "E se o score de crédito subir para 780?"
        if st.button("🔬 Fatores SHAP", use_container_width=True):
            st.session_state["pending_prompt"] = "Quais são os 10 principais fatores de risco avaliados pelo modelo?"

    st.divider()
    st.markdown("### ℹ️ Sobre o Agente")
    st.caption(
        "Este chat está conectado diretamente às 6 ferramentas do servidor **MCP (Model Context Protocol)** "
        "e ao modelo **XGBoost + Optuna + SHAP** treinado com validação cruzada 5-fold."
    )
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_record"] = None
        st.rerun()


# --- Inicialização do Estado da Sessão ---------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Olá! Sou o **Agente de Risco IA**, seu analista autônomo de inteligência de crédito.\n\n"
                "Posso te ajudar a:\n"
                "- 🔍 **Avaliar o risco de inadimplência** de clientes com explicação detalhada (**SHAP**);\n"
                "- 📉 **Simular políticas de concessão (*What-If*)** (ex: *'E se a utilização de limite cair para 30%?'*);\n"
                "- 📁 **Calcular a perda esperada de carteiras inteiras** via arquivo CSV;\n"
                "- 📊 **Apresentar métricas e auditoria** de performance do modelo.\n\n"
                "Como posso te ajudar hoje? (Você também pode clicar nos botões de atalho na barra lateral!)"
            ),
        }
    ]

if "last_record" not in st.session_state:
    st.session_state["last_record"] = SAMPLE_PROFILES["🔴 4. Alto (35–60%) — Cliente Alerta"]


# --- Cabeçalho Principal -----------------------------------------------
st.markdown('<div class="main-header">🏦 Agente de Risco IA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decisões de crédito inteligentes, explicáveis (SHAP) e conectadas via Protocolo MCP</div>', unsafe_allow_html=True)

# Trata clique em perfil da sidebar
if selected_sample:
    profile_data = SAMPLE_PROFILES[selected_sample]
    st.session_state["last_record"] = profile_data
    prompt_text = (
        f"Avalie o risco de default para o **{selected_sample}** com os seguintes dados:\n"
        f"- Idade: {profile_data['age']} anos | Gênero: {profile_data['gender']}\n"
        f"- Renda Anual: R$ {profile_data['net_yearly_income']:,.2f}\n"
        f"- Limite de Crédito: R$ {profile_data['credit_limit']:,.2f} (Usado: {profile_data['credit_limit_used(%)']}%)\n"
        f"- Score de Crédito: {profile_data['credit_score']}\n"
        f"- Inadimplências Anteriores: {profile_data['prev_defaults']} (Últimos 6 meses: {profile_data['default_in_last_6months']})\n"
        f"- Ocupação: {profile_data['occupation_type']} | Dívida Anual: R$ {profile_data['yearly_debt_payments']:,.2f}"
    )
    st.session_state["pending_prompt"] = prompt_text


def render_tool_result_ui(tool_type: str, data: dict[str, Any]) -> None:
    """Renderiza gráficos, métricas e legendas explicativas para leigos."""
    if tool_type == "explain_prediction":
        st.markdown(get_risk_badge_html(data["risk_tier"], data["probability_of_default"]), unsafe_allow_html=True)
        
        # Tabela e Gráfico SHAP
        factors = data.get("top_factors", [])
        if factors:
            st.markdown("##### 🔬 Decomposição dos Fatores de Decisão (SHAP):")
            df_shap = pd.DataFrame(factors)
            df_plot = df_shap.copy()
            df_plot["shap_abs"] = df_plot["shap_contribution"].abs()
            df_plot = df_plot.sort_values("shap_abs", ascending=True)
            
            st.bar_chart(
                data=df_plot.set_index("feature")["shap_contribution"],
                horizontal=True,
            )
        
        # 💡 LEGENDA DIDÁTICA PARA LEIGOS
        with st.expander("💡 Como interpretar este resultado? (Guia para Leigos)", expanded=True):
            st.markdown(
                """
                * 📈 **O que é a Probabilidade de Default?** É a chance estimada (de 0% a 100%) de o cliente atrasar a fatura por mais de 90 dias nos próximos meses.
                * 📊 **Como ler o gráfico de barras SHAP?**
                  * 🔴 **Barras para a DIREITA (Valores Positivos):** Características que **aumentam o risco** (ex: score baixo, uso excessivo do limite rotativo ou histórico de inadimplência).
                  * 🟢 **Barras para a ESQUERDA (Valores Negativos):** Características saudáveis que **diminuem o risco e protegem o cliente** (ex: muitos anos de estabilidade no emprego, alta renda ou score alto).
                  * 📏 **Tamanho da barra:** Quanto maior a barra, mais decisiva essa informação foi para a decisão da IA.
                * 🚦 **O que fazer com este cliente?**
                  * 🟢 **MUITO BAIXO / BAIXO (<15%):** Risco mínimo. Aprovação ou aumento de limite recomendado.
                  * 🟡 **MODERADO (15% a 35%):** Risco intermediário. Sugerido limite inicial menor ou comprovação de renda.
                  * 🔴 **ALTO / MUITO ALTO (>35%):** Risco crítico. Recomendada recusa ou solicitação de avalista/garantia.
                """
            )

    elif tool_type == "what_if_analysis":
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidade Original", f"{data['baseline_prob']*100:.2f}%", data["baseline_tier"])
        c2.metric(f"Com {data['field_changed']} = {data['new_value']}", f"{data['scenario_prob']*100:.2f}%", f"{data['delta']*100:+.2f}%", delta_color="inverse")
        c3.metric("Nova Faixa de Risco", data["scenario_tier"])

        # 💡 LEGENDA DIDÁTICA PARA LEIGOS
        with st.expander("💡 Entendendo a Simulação What-If (Guia para Leigos)", expanded=True):
            st.markdown(
                f"""
                * 🔄 **O que é a simulação *What-If*?** É um simulador de cenários que responde: *"O que aconteceria com o risco se apenas uma característica mudasse?"*
                * 🎯 **Aplicação prática:** Se um cliente teve o crédito negado, a equipe de atendimento pode usar essa ferramenta para orientá-lo (ex: *"Se você reduzir a utilização do seu limite para **{data['new_value']}%**, seu risco cai de **{data['baseline_prob']*100:.1f}%** para **{data['scenario_prob']*100:.1f}%**, permitindo aprovar seu cartão"*).
                """
            )

    elif tool_type == "portfolio_risk_summary":
        c1, c2, c3 = st.columns(3)
        c1.metric("Clientes Analisados", f"{data['n_customers']:,}")
        c2.metric("Exposição Total", f"R$ {data['total_exposure']:,.2f}")
        c3.metric("Perda Esperada (PD × Limite)", f"R$ {data['total_expected_loss']:,.2f}", f"Taxa: {data['expected_loss_rate_pct']:.2f}%", delta_color="inverse")
        
        st.markdown("##### 📊 Distribuição da Carteira por Faixa de Risco:")
        st.bar_chart(data["risk_tier_distribution"])

        # 💡 LEGENDA DIDÁTICA PARA LEIGOS
        with st.expander("💡 Entendendo os Números da Carteira (Guia para Leigos)", expanded=True):
            st.markdown(
                """
                * 💰 **Exposição Total:** É a soma de todos os limites de crédito concedidos aos clientes avaliados (quanto dinheiro o banco colocou em jogo).
                * ⚠️ **Perda Esperada ($PD \\times \\text{Limite}$):** É a estimativa em Reais de quanto dinheiro a instituição pode perder por inadimplência se nenhuma ação preventiva for tomada.
                * 📉 **Taxa de Perda:** O percentual da carteira em risco — serve de parâmetro direto para o provisionamento contábil de perdas (PDD / IFRS 9).
                """
            )

    elif tool_type == "get_model_performance":
        # 💡 LEGENDA DIDÁTICA PARA LEIGOS
        with st.expander("💡 O que significam essas métricas de validação? (Guia para Leigos)", expanded=True):
            st.markdown(
                """
                * 🏆 **ROC-AUC (0,9960):** Uma nota de 0 a 1 para a capacidade do modelo de colocar os bons pagadores no topo e os maus pagadores na base. **0,9960 indica capacidade de ordenação quase perfeita**.
                * 🎯 **Precisão (96,5%):** De cada 100 clientes que o modelo apontou como inadimplentes, **mais de 96 realmente deram calote**. Isso significa que o modelo quase nunca acusa um bom cliente injustamente.
                * 🛡️ **Recall / Sensibilidade (80,0%):** O modelo consegue capturar e barrar **8 em cada 10 calotes reais** antes que aconteçam.
                * ⚖️ **F1-Score (0,8749):** Mede o equilíbrio entre não deixar calotes passarem e não recusar clientes bons por engano.
                """
            )


# --- Renderização do Histórico de Mensagens ----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "extra_ui" in msg:
            render_tool_result_ui(msg["extra_ui"]["type"], msg["extra_ui"]["data"])


# --- Processamento de Nova Mensagem do Usuário ------------------------
prompt = st.chat_input("Digite sua solicitação (ex: 'Avalie este cliente...', 'E se o limite for 30%?', 'Performance do modelo')...")

if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    # 1. Registra mensagem do usuário
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Executa a inteligência do Agente de Risco IA
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("⚙️ Agente consultando servidor MCP e executando modelo...", expanded=True) as status:
            st.write("🔍 Analisando parâmetros da solicitação...")
            result = parse_and_execute_tool(prompt, st.session_state.get("last_record"))
            tool_name = result["tool"]
            data = result["data"]
            st.write(f"✅ Ferramenta MCP acionada: `{tool_name}`")
            status.update(label=f"✅ Análise concluída via `{tool_name}`!", state="complete", expanded=False)

        # Monta a resposta formatada
        response_text = ""
        extra_ui = None

        if tool_name == "explain_prediction":
            st.session_state["last_record"] = data.get("record_evaluated", st.session_state.get("last_record"))
            prob = data["probability_of_default"]
            tier = data["risk_tier"]
            label = data["predicted_label"]
            
            response_text = (
                f"### 📋 Parecer de Risco do Cliente\n\n"
                f"- **Decisão do Modelo:** **`{label}`**\n"
                f"- **Probabilidade de Inadimplência:** **`{prob*100:.2f}%`**\n"
                f"- **Classificação de Risco:** **`{tier}`** (Limiar calibrado: `{data['decision_threshold_used']}`)\n\n"
                f"#### 🔍 Fatores Determinantes (Auditoria SHAP):\n"
            )
            for f in data.get("top_factors", [])[:5]:
                emoji = "🔴" if "AUMENTA" in f["effect"] else "🟢"
                response_text += f"- {emoji} **`{f['feature']}`**: contribuição SHAP de `{f['shap_contribution']:+.4f}` ({f['effect']}).\n"

            extra_ui = {"type": "explain_prediction", "data": data}

        elif tool_name == "what_if_analysis":
            st.session_state["last_record"] = data.get("updated_record", st.session_state.get("last_record"))
            delta = data["delta"]
            direction = "REDUZIU" if delta < 0 else ("AUMENTOU" if delta > 0 else "NÃO ALTEROU")
            response_text = (
                f"### 📉 Simulação de Cenário (*What-If*)\n\n"
                f"Ao alterar o campo **`{data['field_changed']}`** de **`{data['original_value']}`** para **`{data['new_value']}`**:\n\n"
                f"- A probabilidade de default **{direction}** em **`{abs(delta)*100:.2f}%`** "
                f"(de `{data['baseline_prob']*100:.2f}%` para **`{data['scenario_prob']*100:.2f}%`**).\n"
                f"- A faixa de risco migrou de **`{data['baseline_tier']}`** para **`{data['scenario_tier']}`**."
            )
            extra_ui = {"type": "what_if_analysis", "data": data}

        elif tool_name == "portfolio_risk_summary":
            response_text = (
                f"### 📊 Relatório Executivo da Carteira\n\n"
                f"- **Arquivo Avaliado:** `{data['csv_path']}`\n"
                f"- **Clientes Processados:** **`{data['n_customers']:,}`**\n"
                f"- **Exposição Total de Limite:** **`R$ {data['total_exposure']:,.2f}`**\n"
                f"- **Perda Esperada (PD × Limite):** **`R$ {data['total_expected_loss']:,.2f}`** (`{data['expected_loss_rate_pct']:.2f}%` da carteira)\n"
            )
            extra_ui = {"type": "portfolio_risk_summary", "data": data}

        elif tool_name == "get_model_performance":
            metrics = data["holdout_metrics"]
            response_text = (
                f"### 🛡️ Ficha Técnica e Auditoria do Modelo Campeão\n\n"
                f"- **Algoritmo:** `{data['algorithm']}`\n"
                f"- **Versão:** `{data['model_version']}` | **Treinado em:** `{data['trained_at_utc']}`\n"
                f"- **Volume de Treino:** `{data['n_train_rows']:,}` linhas (`{data['positive_rate_train']*100:.2f}%` default)\n\n"
                f"#### 📈 Métricas de Generalização (Holdout Isolado de 6.830 clientes):\n"
                f"- **ROC-AUC:** **`{metrics['roc_auc']:.4f}`**\n"
                f"- **PR-AUC (Average Precision):** **`{metrics['pr_auc']:.4f}`**\n"
                f"- **F1-Score (@ threshold {metrics['best_threshold']:.3f}):** **`{metrics['f1_at_best_threshold']:.4f}`**\n"
                f"- **Matriz de Confusão:** `{metrics['confusion_matrix']}`\n"
            )
            extra_ui = {"type": "get_model_performance", "data": data}

        st.markdown(response_text, unsafe_allow_html=True)
        if extra_ui:
            render_tool_result_ui(extra_ui["type"], extra_ui["data"])

    # Salva no histórico
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_text,
        "extra_ui": extra_ui,
    })
