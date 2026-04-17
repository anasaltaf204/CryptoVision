"""
app/models/schemas.py
=====================
Pydantic models for request / response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    coin: str = Field(..., description="Coin ticker, e.g. BTC")
    run_backtest: bool = Field(False, description="Include walk-forward backtest metrics")


# ── Live market sub-model ─────────────────────────────────────────────────────

class LiveMarket(BaseModel):
    price: Optional[float] = None
    price_change_pct_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    timestamp: Optional[str] = None


# ── Forecast sub-model ────────────────────────────────────────────────────────

class ForecastData(BaseModel):
    horizons: List[str] = Field(default_factory=list)
    current_price: Optional[float] = None

    # Per-model predictions (list aligned with horizons)
    ensemble: List[Optional[float]] = Field(default_factory=list)
    ensemble_lower: List[Optional[float]] = Field(default_factory=list)
    ensemble_upper: List[Optional[float]] = Field(default_factory=list)
    ensemble_uncertainty: List[Optional[float]] = Field(default_factory=list)

    prophet: List[Optional[float]] = Field(default_factory=list)
    sarima: List[Optional[float]] = Field(default_factory=list)
    lstm: List[Optional[float]] = Field(default_factory=list)
    transformer: List[Optional[float]] = Field(default_factory=list)


# ── Validation metrics sub-model ─────────────────────────────────────────────

class HorizonMetric(BaseModel):
    mae: float
    rmse: float
    mape: float
    dir_acc: float


# ── Main response ─────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    coin: str
    generated_at: str
    live_market: LiveMarket
    indicators: Dict[str, float] = Field(default_factory=dict)
    forecast: ForecastData
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    backtest_metrics: Dict[str, Any] = Field(default_factory=dict)


# ── Health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    trained_coins: List[str]
    supported_coins: List[str]
