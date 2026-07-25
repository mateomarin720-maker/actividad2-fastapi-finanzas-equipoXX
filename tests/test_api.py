"""
Pruebas automatizadas de la API financiera, usando el TestClient de FastAPI.

Uso:
    poetry run pytest
"""

import pytest
from fastapi.testclient import TestClient

from financial_api.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_loaded"], bool)


def test_market_data_endpoint_known_symbol():
    response = client.get("/market-data/AAPL")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["n_points"] > 0
    assert len(body["data"]) == body["n_points"]


def test_market_data_endpoint_unknown_symbol():
    response = client.get("/market-data/NOEXISTE123")
    assert response.status_code == 404


def test_predict_endpoint_valid_request():
    response = client.post(
        "/predict",
        json={"symbol": "AAPL", "prediction_horizon": 1, "use_cached_data": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["prediction"] in ("up", "down")
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["model_version"]
    assert body["prediction_horizon"] == "next_day"


def test_predict_endpoint_invalid_request_missing_symbol():
    response = client.post("/predict", json={"prediction_horizon": 1})
    assert response.status_code == 422  # error de validación de Pydantic


def test_predict_batch_endpoint_valid_request():
    response = client.post(
        "/predict/batch",
        json={"symbols": ["AAPL", "MSFT"], "prediction_horizon": 1, "use_cached_data": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 2
    for item in body["results"]:
        assert item["prediction"] in ("up", "down")
        assert 0.0 <= item["probability_up"] <= 1.0


def test_predict_batch_endpoint_rejects_empty_list():
    response = client.post("/predict/batch", json={"symbols": []})
    assert response.status_code == 422


def test_model_metadata_endpoint():
    response = client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "random_forest_v1"
    assert isinstance(body["symbols"], list)
    assert len(body["symbols"]) >= 3
    assert "disclaimer" in body


def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
