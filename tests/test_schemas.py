"""
Pruebas de validación de los contratos Pydantic (schemas).

Uso:
    poetry run pytest
"""

import pytest
from pydantic import ValidationError

from financial_api.schemas import PredictRequest, PredictResponse


def test_predict_request_valid():
    req = PredictRequest(symbol="AAPL")
    assert req.symbol == "AAPL"
    assert req.prediction_horizon == 1
    assert req.use_cached_data is True


def test_predict_request_rejects_empty_symbol():
    with pytest.raises(ValidationError):
        PredictRequest(symbol="")


def test_predict_request_rejects_horizon_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequest(symbol="AAPL", prediction_horizon=99)


def test_predict_response_valid():
    resp = PredictResponse(
        symbol="AAPL",
        prediction="up",
        probability_up=0.63,
        model_version="random_forest_v1",
        prediction_horizon="next_day",
    )
    assert resp.prediction == "up"


def test_predict_response_rejects_invalid_prediction_label():
    with pytest.raises(ValidationError):
        PredictResponse(
            symbol="AAPL",
            prediction="sideways",  # no es "up" ni "down"
            probability_up=0.5,
            model_version="random_forest_v1",
            prediction_horizon="next_day",
        )


def test_predict_response_rejects_probability_out_of_bounds():
    with pytest.raises(ValidationError):
        PredictResponse(
            symbol="AAPL",
            prediction="up",
            probability_up=1.5,  # fuera de [0, 1]
            model_version="random_forest_v1",
            prediction_horizon="next_day",
        )
