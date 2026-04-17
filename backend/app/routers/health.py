"""
app/routers/health.py
=====================
Health and status endpoints.
"""

from fastapi import APIRouter

from app.core.trainer_state import trainer_state
from app.models.schemas import HealthResponse

router = APIRouter()

SUPPORTED_COINS = ["BTC", "ETH", "BNB", "SOL", "XRP"]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns backend readiness and which coins have trained models.
    The frontend polls this on load to know when to enable the UI.
    """
    return HealthResponse(
        status="ready" if trainer_state.is_ready else "training",
        trained_coins=trainer_state.trained_coins,
        supported_coins=SUPPORTED_COINS,
    )
