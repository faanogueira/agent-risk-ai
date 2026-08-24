"""
Configurações centrais do projeto Agente de Risco IA (Agent Risk AI).

Mantém caminhos, sementes e constantes de negócio em um único lugar,
evitando "magic numbers" espalhados pelo código.
"""
from pathlib import Path

# --- Caminhos ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "credit_default_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --- Reprodutibilidade --------------------------------------------------
RANDOM_STATE = 42

# --- Definição do problema ----------------------------------------------
TARGET_COL = "credit_card_default"
ID_COL = "customer_id"

# Colunas descartadas por não carregarem sinal preditivo (PII / cardinalidade alta)
DROP_COLS = ["name"]

# Sentinela do Home Credit-like dataset: "não empregado" costuma vir
# codificado como um valor absurdamente alto (~365243 dias ~ 1000 anos).
DAYS_EMPLOYED_ANOMALY_THRESHOLD = 300_000

# Percentil usado para winsorizar (capar) variáveis monetárias com cauda
# longa extrema (ex.: renda anual chegando a 140 milhões vs. mediana de 170 mil).
INCOME_CAP_QUANTILE = 0.995

# Categoria residual observada em `gender` (1 registro "XNA" em 45k linhas)
RARE_GENDER_VALUES = {"XNA"}

# --- Faixas de risco usadas pelo servidor MCP para tornar o score acionável ---
RISK_TIERS = [
    (0.00, 0.05, "MUITO_BAIXO"),
    (0.05, 0.15, "BAIXO"),
    (0.15, 0.35, "MODERADO"),
    (0.35, 0.60, "ALTO"),
    (0.60, 1.01, "MUITO_ALTO"),
]

# Threshold de decisão default (será sobrescrito pelo valor otimizado
# salvo em model_metadata.json após o treino, via F1/custo de negócio).
DEFAULT_DECISION_THRESHOLD = 0.5
