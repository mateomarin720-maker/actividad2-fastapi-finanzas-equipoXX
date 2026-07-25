"""
financial_api.train
--------------------
Entrena un modelo de clasificación (¿el retorno del día siguiente será
positivo?) usando las features derivadas de los precios históricos de
los símbolos descargados/cacheados, y serializa el modelo + sus metadatos.

Uso:
    poetry run python -m financial_api.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from financial_api.features import FEATURE_COLUMNS, build_features

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

MODEL_VERSION = "random_forest_v1"
PREDICTION_HORIZON_DAYS = 1


def load_all_symbols_features() -> pd.DataFrame:
    """Carga todos los CSV crudos disponibles y construye features por símbolo."""
    frames = []
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        symbol = csv_path.stem
        raw_df = pd.read_csv(csv_path, parse_dates=["Date"])
        feat_df = build_features(raw_df)
        feat_df["symbol"] = symbol
        frames.append(feat_df)

    if not frames:
        raise FileNotFoundError(
            "No se encontraron datos en data/raw/. Corre primero: "
            "poetry run python -m financial_api.data"
        )
    return pd.concat(frames, ignore_index=True)


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split cronológico (no aleatorio): las últimas observaciones de cada
    símbolo se reservan para test, evitando fuga de información temporal
    típica en series financieras.
    """
    train_frames, test_frames = [], []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("Date")
        cutoff = int(len(group) * (1 - test_size))
        train_frames.append(group.iloc[:cutoff])
        test_frames.append(group.iloc[cutoff:])
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    full_df = load_all_symbols_features()
    full_df = full_df.dropna(subset=FEATURE_COLUMNS + ["target_up"])

    # Guarda el dataset procesado combinado como evidencia/reproducibilidad
    full_df.to_csv(PROCESSED_DIR / "features_dataset.csv", index=False)

    train_df, test_df = time_based_split(full_df, test_size=0.2)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_up"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_up"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}")

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(full_df["symbol"].unique().tolist()),
        "feature_columns": FEATURE_COLUMNS,
        "target": "target_up (1 = retorno del siguiente día positivo, 0 = negativo o cero)",
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "metric_name": "roc_auc",
        "metric_value": round(float(roc_auc), 4),
        "accuracy": round(float(accuracy), 4),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "model_type": "RandomForestClassifier",
        "disclaimer": (
            "Herramienta academica de analisis de senales financieras. "
            "No constituye asesoria financiera ni recomendacion de inversion."
        ),
    }

    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en {ARTIFACTS_DIR / 'model.joblib'}")
    print(f"Metadatos guardados en {ARTIFACTS_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
