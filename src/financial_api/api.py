"""
financial_api.api
------------------
Aplicación FastAPI que expone:
  - GET  /health
  - GET  /market-data/{symbol}
  - POST /predict
  - GET  /model/metadata

Documentación interactiva disponible automáticamente en /docs.
"""

from fastapi import FastAPI, HTTPException

from financial_api import predict as predict_module
from financial_api.data import RAW_DIR
from financial_api.schemas import (
    HealthResponse,
    MarketDataPoint,
    MarketDataResponse,
    ModelMetadataResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="Financial Signals API (educativo)",
    description=(
        "API académica de análisis de señales financieras. "
        "NO constituye asesoría financiera ni recomendación de inversión."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        predict_module.load_model()
        model_loaded = True
    except Exception:  # noqa: BLE001
        model_loaded = False
    return HealthResponse(status="ok" if model_loaded else "degraded", model_loaded=model_loaded)


@app.get("/market-data/{symbol}", response_model=MarketDataResponse)
def market_data(symbol: str, limit: int = 30) -> MarketDataResponse:
    cache_path = RAW_DIR / f"{symbol.upper()}.csv"
    if not cache_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No hay datos cacheados para '{symbol}'. Corre 'financial_api.data' primero.",
        )

    df = predict_module.get_symbol_dataframe(symbol.upper(), use_cached_data=True)
    df = df.sort_values("Date").tail(limit)

    points = [
        MarketDataPoint(
            date=str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
        )
        for _, row in df.iterrows()
    ]

    return MarketDataResponse(symbol=symbol.upper(), n_points=len(points), data=points)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result = predict_module.predict_symbol(
            symbol=request.symbol.upper(),
            prediction_horizon=request.prediction_horizon,
            use_cached_data=request.use_cached_data,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PredictResponse(**result)


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    """
    Igual que /predict, pero acepta una lista de símbolos y devuelve una
    predicción por cada uno. Un símbolo con error individual no detiene
    el resto del lote (queda simplemente fuera de la respuesta y se
    reporta en logs); si se prefiere fallar rápido, cambiar el manejo
    de excepciones aquí.
    """
    results = []
    for symbol in request.symbols:
        try:
            result = predict_module.predict_symbol(
                symbol=symbol.upper(),
                prediction_horizon=request.prediction_horizon,
                use_cached_data=request.use_cached_data,
            )
            results.append(PredictResponse(**result))
        except (FileNotFoundError, ValueError):
            continue

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No se pudo generar ninguna predicción para los símbolos solicitados.",
        )

    return PredictBatchResponse(results=results)


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    try:
        metadata = predict_module.load_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return ModelMetadataResponse(
        model_version=metadata["model_version"],
        trained_at=metadata["trained_at"],
        symbols=metadata["symbols"],
        feature_columns=metadata["feature_columns"],
        metric_name=metadata["metric_name"],
        metric_value=metadata["metric_value"],
        prediction_horizon_days=metadata["prediction_horizon_days"],
        disclaimer=metadata["disclaimer"],
    )
