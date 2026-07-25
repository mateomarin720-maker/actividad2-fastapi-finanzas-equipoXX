"""
financial_api.features
-----------------------
Construye variables (features) a partir de precios históricos: retornos,
medias móviles, volatilidad y rezagos. Usadas tanto para entrenamiento
(train.py) como para inferencia en tiempo real (predict.py).
"""

import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_5d",
    "volatility_10d",
    "lag_return_1",
    "lag_return_2",
    "lag_return_3",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con columnas Date, Open, High, Low, Close, Volume
    (ordenado cronológicamente) y devuelve uno con las features derivadas
    más la columna target (retorno positivo del día siguiente).
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()

    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()
    df["lag_return_1"] = df["return_1d"].shift(1)
    df["lag_return_2"] = df["return_1d"].shift(2)
    df["lag_return_3"] = df["return_1d"].shift(3)

    # Target: ¿el retorno del día siguiente fue positivo?
    df["next_return"] = df["Close"].pct_change().shift(-1)
    df["target_up"] = (df["next_return"] > 0).astype(int)

    return df


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dado un DataFrame de precios ya ordenado, calcula las features y
    devuelve únicamente la última fila válida (sin NaN), lista para
    usarse como entrada del modelo en inferencia.
    """
    features_df = build_features(df)
    valid = features_df.dropna(subset=FEATURE_COLUMNS)
    if valid.empty:
        raise ValueError("No hay suficientes datos históricos para calcular las features.")
    return valid.iloc[[-1]]
