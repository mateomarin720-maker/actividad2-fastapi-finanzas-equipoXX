"""
financial_api.data
-------------------
Descarga datos históricos de activos financieros con yfinance y los guarda
localmente en data/raw/{symbol}.csv, de forma que la API pueda funcionar
sin conexión a internet durante la evaluación (usando la última copia
cacheada disponible).

Uso como script:
    poetry run python -m financial_api.data
    poetry run python -m financial_api.data --symbols AAPL MSFT GOOGL --period 2y
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
RAW_DIR = Path("data/raw")


def _generate_synthetic_prices(symbol: str, n_days: int = 500, seed: int | None = None) -> pd.DataFrame:
    """
    Genera una serie de precios sintética (caminata aleatoria con deriva) para
    un símbolo, usada ÚNICAMENTE como respaldo cuando yfinance no está
    disponible (sin internet) y no existe todavía una copia cacheada en disco.

    Esto garantiza que el proyecto se pueda ejecutar, probar y evaluar de
    forma reproducible incluso sin conexión, tal como exige la actividad.
    No debe interpretarse como datos de mercado reales.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    daily_returns = rng.normal(loc=0.0004, scale=0.018, size=n_days)
    close = 100 * np.exp(np.cumsum(daily_returns))

    high = close * (1 + np.abs(rng.normal(0, 0.006, size=n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, size=n_days)))
    open_ = close * (1 + rng.normal(0, 0.004, size=n_days))
    volume = rng.integers(1_000_000, 20_000_000, size=n_days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )
    return df


def fetch_symbol_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Intenta descargar datos históricos reales con yfinance. Si falla
    (sin internet, símbolo inválido, límite de tasa, etc.), recurre a la
    copia cacheada en disco si existe, y solo como último recurso genera
    datos sintéticos de respaldo para no bloquear el flujo.
    """
    cache_path = RAW_DIR / f"{symbol}.csv"

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            raise ValueError(f"yfinance devolvió datos vacíos para {symbol}")

        hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        print(f"[data] {symbol}: {len(hist)} filas descargadas con yfinance.")
        return hist

    except Exception as e:  # noqa: BLE001
        print(f"[data] Aviso: no se pudo descargar {symbol} con yfinance ({e}).")
        if cache_path.exists():
            print(f"[data] Usando copia local cacheada: {cache_path}")
            return pd.read_csv(cache_path, parse_dates=["Date"])

        print(f"[data] No existe cache local para {symbol}. Generando datos sintéticos de respaldo.")
        return _generate_synthetic_prices(symbol)


def download_all(symbols: list[str], period: str = "2y") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        df = fetch_symbol_data(symbol, period=period)
        out_path = RAW_DIR / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        print(f"[data] Guardado: {out_path} ({len(df)} filas)")


def main():
    parser = argparse.ArgumentParser(description="Descarga/cachea datos históricos de activos financieros")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Símbolos a descargar")
    parser.add_argument("--period", type=str, default="2y", help="Periodo histórico (ej. 1y, 2y, 5y)")
    args = parser.parse_args()

    download_all(args.symbols, period=args.period)


if __name__ == "__main__":
    main()
