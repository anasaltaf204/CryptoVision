"""
app/core/trainer_state.py
=========================
Singleton that owns trained model objects for all coins.
Training happens once on startup; forecast calls are cheap after that.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from app.core.config import settings

log = logging.getLogger("trainer_state")

# Lazy imports — these are only pulled in when needed.
def _import_pipeline():
    """
    Dynamically import data_pipeline and model_training from DATA_DIR.

    Python 3.13 changed how @dataclass resolves cls.__module__: it now does
    sys.modules[cls.__module__].__dict__ at decoration time. If the module
    isn't registered in sys.modules before exec_module() runs, that lookup
    returns None and raises AttributeError.

    Fix: register the module object in sys.modules *before* calling
    exec_module(), so @dataclass can find it. We also insert the directory
    on sys.path so that any relative imports inside those files work normally.
    """
    import sys
    import importlib.util

    data_dir = Path(settings.DATA_DIR)

    # Support flat layout (data/) and legacy "improved/" subfolder
    for candidate in [data_dir, data_dir.parent / "improved", data_dir.parent]:
        dp = candidate / "data_pipeline.py"
        mt = candidate / "model_training.py"
        if not (dp.exists() and mt.exists()):
            continue

        # Add candidate dir to sys.path so internal imports resolve
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

        # ── data_pipeline ────────────────────────────────────────────────
        spec_dp = importlib.util.spec_from_file_location(
            "data_pipeline", str(dp),
            submodule_search_locations=[],
        )
        data_pipeline = importlib.util.module_from_spec(spec_dp)
        # Register BEFORE exec so @dataclass can resolve __module__
        sys.modules["data_pipeline"] = data_pipeline
        spec_dp.loader.exec_module(data_pipeline)

        # ── model_training ───────────────────────────────────────────────
        spec_mt = importlib.util.spec_from_file_location(
            "model_training", str(mt),
            submodule_search_locations=[],
        )
        model_training = importlib.util.module_from_spec(spec_mt)
        sys.modules["model_training"] = model_training
        spec_mt.loader.exec_module(model_training)

        return data_pipeline, model_training, candidate_str

    raise ImportError(
        f"Could not find data_pipeline.py / model_training.py under {data_dir}. "
        "Set the DATA_DIR environment variable to the folder containing those files."
    )


class TrainerState:
    """Thread-safe singleton holding all trained coin models."""

    def __init__(self):
        self._trained_data: Dict[str, dict] = {}
        self._trained_coins: list = []
        self._lock = asyncio.Lock()
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def trained_coins(self) -> list:
        return list(self._trained_coins)

    async def train_all(self, force_retrain: bool = False) -> None:
        """Train models for all coins. Called once on startup."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._train_all_sync, force_retrain
            )
            self._ready = True

    def _train_all_sync(self, force_retrain: bool = False) -> None:
        """Synchronous training logic (runs in thread pool)."""
        try:
            data_pipeline, model_training, module_dir = _import_pipeline()
        except ImportError as exc:
            log.error("Cannot import ML modules: %s", exc)
            return

        SUPPORTED_COINS = data_pipeline.SUPPORTED_COINS
        build_feature_matrix = data_pipeline.build_feature_matrix
        DataManager = data_pipeline.DataManager
        ModelTrainer = model_training.ModelTrainer
        HORIZONS = model_training.HORIZONS

        data_dir = Path(module_dir)
        trained: list = []
        skipped: list = []

        for coin in SUPPORTED_COINS:
            log.info("Training %s…", coin)
            try:
                # Locate CSV
                csv_path: Optional[str] = None
                for name in [f"{coin}.csv", f"{coin.lower()}.csv"]:
                    p = data_dir / name
                    if p.exists():
                        csv_path = str(p)
                        break

                dm = DataManager(
                    coin=coin,
                    csv_path=csv_path,
                    cache_dir=str(data_dir / settings.CACHE_DIR),
                    coingecko_api_key=settings.COINGECKO_API_KEY or None,
                    cryptopanic_token=settings.CRYPTOPANIC_TOKEN or None,
                    newsapi_key=settings.NEWSAPI_KEY or None,
                    use_websocket=False,
                )
                df = dm.load(fetch_coingecko=True, fetch_binance=True)
                sentiment = dm.sentiment()
                X, y, feature_names = build_feature_matrix(
                    df, window=settings.WINDOW, horizons=HORIZONS, sentiment=sentiment
                )

                trainer = ModelTrainer(
                    coin=coin,
                    cache_dir=str(data_dir / settings.MODEL_CACHE_DIR),
                    epochs=settings.EPOCHS,
                    window=settings.WINDOW,
                    enable_prophet=settings.ENABLE_PROPHET,
                    enable_sarima=settings.ENABLE_SARIMA,
                    enable_lstm=settings.ENABLE_LSTM,
                    enable_transformer=settings.ENABLE_TRANSFORMER,
                    run_optuna=settings.RUN_OPTUNA,
                    optuna_trials=settings.OPTUNA_TRIALS,
                )
                trainer.train(df, X, y, feature_names, force_retrain=force_retrain)

                self._trained_data[coin] = {
                    "trainer": trainer,
                    "df": df,
                    "X": X,
                    "y": y,
                    "feature_names": feature_names,
                    "dm": dm,
                    "module_dir": module_dir,
                    "model_training": model_training,
                    "data_pipeline": data_pipeline,
                }
                trained.append(coin)
                log.info("✅ %s training complete", coin)

            except Exception as exc:
                log.error("❌ Failed to train %s: %s", coin, exc, exc_info=True)
                skipped.append(coin)

        self._trained_coins = trained
        log.info("Training summary — trained: %s | skipped: %s", trained, skipped)

    def get_coin_data(self, coin: str) -> Optional[dict]:
        return self._trained_data.get(coin.upper())


# Module-level singleton
trainer_state = TrainerState()