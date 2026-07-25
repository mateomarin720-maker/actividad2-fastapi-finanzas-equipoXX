"""
financial_api.schemas
----------------------
Contratos de entrada y salida de la API, validados con Pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    prediction_horizon: int = Field(default=1, ge=1, le=5, description="Horizonte en días")
    use_cached_data: bool = Field(
        default=True,
        description="Si es True, usa la copia local cacheada en vez de llamar a yfinance",
    )


class PredictResponse(BaseModel):
    symbol: str
    prediction: Literal["up", "down"]
    probability_up: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prediction_horizon: str


class PredictBatchRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=20, examples=[["AAPL", "MSFT", "GOOGL"]])
    prediction_horizon: int = Field(default=1, ge=1, le=5)
    use_cached_data: bool = Field(default=True)


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


class MarketDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataResponse(BaseModel):
    symbol: str
    n_points: int
    data: list[MarketDataPoint]


class ModelMetadataResponse(BaseModel):
    model_version: str
    trained_at: str
    symbols: list[str]
    feature_columns: list[str]
    metric_name: str
    metric_value: float
    prediction_horizon_days: int
    disclaimer: str
