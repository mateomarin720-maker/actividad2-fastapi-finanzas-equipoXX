"""
financial_api.predict
----------------------
Carga el modelo serializado y genera predicciones para un símbolo dado,
a partir de las features más recientes disponibles (datos cacheados o
recién descargados con yfinance).
"""

import json
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from financial_api.data import fetch_symbol_data, RAW_DIR
from financial_api.features import FEATURE_COLUMNS, latest_feature_row

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. Corre primero: "
            "poetry run python -m financial_api.train"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No se encontraron metadatos en {METADATA_PATH}.")
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def get_symbol_dataframe(symbol: str, use_cached_data: bool = True) -> pd.DataFrame:
    """
    Obtiene el histórico de precios de un símbolo. Si use_cached_data es
    True (default) y existe copia local, se usa directamente sin llamar a
    yfinance, garantizando que la API funcione sin internet.
    """
    cache_path = RAW_DIR / f"{symbol}.csv"
    if use_cached_data and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"])
    return fetch_symbol_data(symbol)


def predict_symbol(symbol: str, prediction_horizon: int = 1, use_cached_data: bool = True) -> dict:
    model = load_model()
    df = get_symbol_dataframe(symbol, use_cached_data=use_cached_data)

    row = latest_feature_row(df)
    X = row[FEATURE_COLUMNS]

    proba_up = float(model.predict_proba(X)[0, 1])
    prediction = "up" if proba_up >= 0.5 else "down"

    metadata = load_metadata()

    return {
        "symbol": symbol.upper(),
        "prediction": prediction,
        "probability_up": round(proba_up, 4),
        "model_version": metadata.get("model_version", "unknown"),
        "prediction_horizon": "next_day" if prediction_horizon == 1 else f"next_{prediction_horizon}_days",
    }
