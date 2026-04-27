from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Reproducibility — fix all seeds at module load
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available — deep-learning models disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("prophet not installed — Prophet model disabled.")

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not installed — SARIMA model disabled.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not installed — hyperparameter tuning disabled.")

log = logging.getLogger("model_training")

HORIZONS            = [7, 30, 90, 180, 365]
HORIZON_LABELS      = ["1w", "1m", "3m", "6m", "1y"]
SARIMA_MAX_HORIZON_IDX = 1   # SARIMA reliable only for ≤30-day horizons
MC_SAMPLES          = 50     # MC Dropout forward passes for uncertainty


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "",
    current_prices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    FIXES applied here:
    ─────────────────────────────────────────────────────────────────────────
    FIX 1 (MAPE explosion):
      The previous code computed MAPE directly on log-returns.
      Log-returns near zero (e.g. 0.001) made the denominator tiny, causing
      MAPE values in the millions (494928%, 871424% seen in logs).
      FIX: MAPE is now ALWAYS computed in price space. If current_prices is
      not supplied, the function falls back to treating the values as prices
      (safe for any non-log-return context such as backtest comparisons).

    FIX 2 (directional accuracy on log-returns):
      Directional accuracy is correctly computed on log-returns directly
      (sign of log-return = direction of price movement), unchanged.
    ─────────────────────────────────────────────────────────────────────────
    """
    # Directional accuracy — computed on the raw (log-return) values
    direction = (
        float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        if len(y_true) > 1 else 0.0
    )

    # Convert log-returns to price space for interpretable MAE / RMSE / MAPE.
    # When current_prices is provided: price = current_price * exp(log_return).
    # When not provided (e.g. backtest comparing price values directly):
    # use the values as-is so the function still works correctly.
    if current_prices is not None:
        y_true_p = current_prices * np.exp(y_true)
        y_pred_p = current_prices * np.exp(y_pred)
    else:
        y_true_p = y_true
        y_pred_p = y_pred

    mae  = mean_absolute_error(y_true_p, y_pred_p)
    mse  = mean_squared_error(y_true_p, y_pred_p)
    rmse = float(np.sqrt(mse))

    # Safe MAPE: denominator protected by a meaningful floor (1.0 in price
    # space, not 1e-8) so near-zero prices don't blow up the metric.
    denom = np.abs(y_true_p) + 1.0
    mape  = float(np.mean(np.abs((y_true_p - y_pred_p) / denom)) * 100)

    log.info(
        "Metrics [%s]: MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  DirAcc=%.3f",
        label, mae, rmse, mape, direction,
    )
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "dir_acc": direction}


# ---------------------------------------------------------------------------
# PyTorch models with MC Dropout
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        """2-layer LSTM with MC Dropout. Supports bidirectional mode."""

        def __init__(
            self,
            input_size:    int,
            hidden_size:   int   = 128,
            num_layers:    int   = 2,
            output_size:   int   = 5,
            dropout:       float = 0.2,
            bidirectional: bool  = False,
        ):
            super().__init__()
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            lstm_out     = hidden_size * 2 if bidirectional else hidden_size
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Sequential(
                nn.Linear(lstm_out, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.fc(self.dropout(out[:, -1, :]))

        def mc_forward(
            self, x: "torch.Tensor", n_samples: int = MC_SAMPLES
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            self.train()
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
            self.eval()
            return preds.mean(dim=0), preds.std(dim=0)

    class TransformerModel(nn.Module):
        """Encoder-only Transformer with learned positional encoding + MC Dropout."""

        def __init__(
            self,
            input_size:         int,
            d_model:            int   = 64,
            nhead:              int   = 4,
            num_encoder_layers: int   = 2,
            output_size:        int   = 5,
            dropout:            float = 0.1,
            max_seq_len:        int   = 500,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc    = nn.Embedding(max_seq_len, d_model)
            encoder_layer   = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(d_model, output_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            seq_len   = x.size(1)
            positions = torch.arange(seq_len, device=x.device)
            x = self.input_proj(x) + self.pos_enc(positions)
            x = self.encoder(x)
            return self.fc(self.dropout(x.mean(dim=1)))

        def mc_forward(
            self, x: "torch.Tensor", n_samples: int = MC_SAMPLES
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            self.train()
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
            self.eval()
            return preds.mean(dim=0), preds.std(dim=0)

    def _train_torch_model(
        model:      "nn.Module",
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        epochs:     int   = 150,
        batch_size: int   = 32,
        lr:         float = 1e-3,
        device:     str   = "cpu",
        patience:   int   = 20,
    ) -> "nn.Module":
        """Train with early stopping + cosine LR + Huber loss."""
        model     = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.HuberLoss(delta=1.0)

        Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
        yt = torch.tensor(y_train, dtype=torch.float32).to(device)
        Xv = torch.tensor(X_val,   dtype=torch.float32).to(device)
        yv = torch.tensor(y_val,   dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED)
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        )

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            avg_train = epoch_loss / len(Xt)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xv), yv).item()
            model.train()

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                log.info("  Epoch %3d/%d  train=%.6f  val=%.6f", epoch, epochs, avg_train, val_loss)

            if no_improve >= patience:
                log.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def tune_lstm_hyperparams(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    n_trials: int = 20,
    device:   str = "cpu",
) -> Dict:
    if not OPTUNA_AVAILABLE or not TORCH_AVAILABLE:
        log.info("Optuna/Torch not available — using default hyperparams")
        return {"hidden_size": 128, "num_layers": 2, "dropout": 0.2,
                "lr": 1e-3, "batch_size": 32, "bidirectional": False}

    n_feat = X_train.shape[2]

    def objective(trial):
        hidden_size   = trial.suggest_categorical("hidden_size",  [64, 128, 256])
        num_layers    = trial.suggest_int("num_layers",            1, 3)
        dropout       = trial.suggest_float("dropout",            0.1, 0.4)
        lr            = trial.suggest_float("lr",                 1e-4, 1e-2, log=True)
        batch_size    = trial.suggest_categorical("batch_size",   [16, 32, 64])
        bidirectional = trial.suggest_categorical("bidirectional", [False, True])

        model = LSTMModel(
            input_size=n_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=len(HORIZONS),
            dropout=dropout,
            bidirectional=bidirectional,
        )
        model = _train_torch_model(
            model, X_train, y_train, X_val, y_val,
            epochs=30, batch_size=batch_size, lr=lr, device=device, patience=8,
        )
        model.eval()
        Xv = torch.tensor(X_val, dtype=torch.float32)
        with torch.no_grad():
            pred = model(Xv).numpy()
        return float(mean_squared_error(y_val, pred))

    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log.info("Optuna best LSTM params: %s", best)
    return best


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def _train_prophet(close_series: pd.Series, horizon_days: int) -> Tuple:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("prophet not installed")
    log_close = np.log(close_series.values.clip(1e-8))
    df = pd.DataFrame({
        "ds": (close_series.index.tz_localize(None)
               if close_series.index.tz is not None else close_series.index),
        "y":  log_close,
    })
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.4,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    m.add_seasonality(name="halving_cycle", period=1461, fourier_order=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)
    future   = m.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = m.predict(future)
    return m, forecast


def _prophet_predict_horizons(close_series: pd.Series) -> np.ndarray:
    if not PROPHET_AVAILABLE:
        return np.full(len(HORIZONS), np.nan)
    try:
        m, forecast = _train_prophet(close_series, max(HORIZONS))
        tail = forecast.tail(max(HORIZONS))
        return np.array([float(np.exp(tail.iloc[h - 1]["yhat"])) for h in HORIZONS])
    except Exception as exc:
        log.warning("Prophet predict failed: %s", exc)
        return np.full(len(HORIZONS), np.nan)


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

def _train_sarima(close_series: pd.Series):
    if not PMDARIMA_AVAILABLE:
        raise RuntimeError("pmdarima not installed")
    log_close = np.log(close_series.values.clip(1e-8))
    model = pm.auto_arima(
        log_close, start_p=1, start_q=1, max_p=3, max_q=3,
        d=1, seasonal=False, information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore",
        random_state=GLOBAL_SEED,
    )
    log.info("SARIMA order: %s", model.order)
    return model


def _sarima_predict_horizons(model) -> np.ndarray:
    if not PMDARIMA_AVAILABLE:
        return np.full(len(HORIZONS), np.nan)
    try:
        preds = np.full(len(HORIZONS), np.nan)
        for i, h in enumerate(HORIZONS):
            if i <= SARIMA_MAX_HORIZON_IDX:
                preds[i] = float(np.exp(model.predict(n_periods=h)[-1]))
        return preds
    except Exception as exc:
        log.warning("SARIMA predict failed: %s", exc)
        return np.full(len(HORIZONS), np.nan)


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def scale_features(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    n, w, f  = X.shape
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X.reshape(-1, f)).reshape(n, w, f)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)
    return X_scaled, y_scaled, x_scaler, y_scaler


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------

@dataclass
class ModelCache:
    cache_dir: str = ".model_cache"

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key_path(self, coin: str, model_name: str, fingerprint: str) -> Path:
        return Path(self.cache_dir) / f"{coin}_{model_name}_{fingerprint[:16]}"

    def exists(self, coin: str, model_name: str, fingerprint: str) -> bool:
        return self._key_path(coin, model_name, fingerprint).with_suffix(".pkl").exists()

    def save(self, obj, coin: str, model_name: str, fingerprint: str) -> None:
        with open(self._key_path(coin, model_name, fingerprint).with_suffix(".pkl"), "wb") as fh:
            pickle.dump(obj, fh, protocol=5)

    def load(self, coin: str, model_name: str, fingerprint: str):
        with open(self._key_path(coin, model_name, fingerprint).with_suffix(".pkl"), "rb") as fh:
            return pickle.load(fh)

    def save_torch(self, model, coin: str, model_name: str, fingerprint: str) -> None:
        if TORCH_AVAILABLE:
            torch.save(
                model.state_dict(),
                self._key_path(coin, model_name, fingerprint).with_suffix(".pt"),
            )

    def load_torch(self, model, coin: str, model_name: str, fingerprint: str):
        model.load_state_dict(
            torch.load(
                self._key_path(coin, model_name, fingerprint).with_suffix(".pt"),
                map_location="cpu",
            )
        )
        model.eval()
        return model

    def torch_exists(self, coin: str, model_name: str, fingerprint: str) -> bool:
        return self._key_path(coin, model_name, fingerprint).with_suffix(".pt").exists()


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    coin:                    str
    horizons:                List[str]            = field(default_factory=lambda: HORIZON_LABELS)
    prophet_preds:           Optional[np.ndarray] = None
    sarima_preds:            Optional[np.ndarray] = None
    lstm_preds:              Optional[np.ndarray] = None
    lstm_uncertainty:        Optional[np.ndarray] = None
    transformer_preds:       Optional[np.ndarray] = None
    transformer_uncertainty: Optional[np.ndarray] = None
    ensemble_preds:          Optional[np.ndarray] = None
    ensemble_lower:          Optional[np.ndarray] = None
    ensemble_upper:          Optional[np.ndarray] = None
    metrics:                 Dict                 = field(default_factory=dict)
    current_price:           Optional[float]      = None

    def to_dict(self) -> Dict:
        out: Dict = {"coin": self.coin, "current_price": self.current_price, "predictions": {}}
        for i, label in enumerate(self.horizons):
            out["predictions"][label] = {}
            for name, arr, unc in [
                ("prophet",     self.prophet_preds,     None),
                ("sarima",      self.sarima_preds,      None),
                ("lstm",        self.lstm_preds,        self.lstm_uncertainty),
                ("transformer", self.transformer_preds, self.transformer_uncertainty),
                ("ensemble",    self.ensemble_preds,    None),
            ]:
                if arr is not None and i < len(arr) and not np.isnan(arr[i]):
                    p      = float(arr[i])
                    change = ((p - self.current_price) / self.current_price * 100
                              if self.current_price and self.current_price > 0 else None)
                    entry  = {
                        "price":      round(p, 4),
                        "change_pct": round(float(change), 2) if change is not None else None,
                    }
                    if unc is not None and i < len(unc):
                        entry["uncertainty_std"] = round(float(unc[i]), 4)
                        entry["ci_lower"]        = round(p - 1.96 * float(unc[i]), 4)
                        entry["ci_upper"]        = round(p + 1.96 * float(unc[i]), 4)
                    if name == "ensemble":
                        if self.ensemble_lower is not None and i < len(self.ensemble_lower):
                            entry["ci_lower"] = round(float(self.ensemble_lower[i]), 4)
                            entry["ci_upper"] = round(float(self.ensemble_upper[i]), 4)
                    out["predictions"][label][name] = entry
        out["metrics"] = self.metrics
        return out


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

@dataclass
class ModelTrainer:
    coin:               str
    cache_dir:          str  = ".model_cache"
    epochs:             int  = 150
    window:             int  = 60
    device:             str  = "cpu"
    enable_prophet:     bool = True
    enable_sarima:      bool = True
    enable_lstm:        bool = True
    enable_transformer: bool = True
    run_optuna:         bool = True
    optuna_trials:      int  = 20

    def __post_init__(self):
        self.cache = ModelCache(self.cache_dir)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            log.info("Using GPU: %s", torch.cuda.get_device_name(0))

    def data_fingerprint(self, df: pd.DataFrame) -> str:
        key = f"{self.coin}_{len(df)}_{df.index[-1].isoformat()}"
        return hashlib.sha256(key.encode()).hexdigest()


    # ------------------------------------------------------------------
    def _find_previous_fingerprint(self) -> Optional[str]:
        """
        Search the model cache directory for any previously saved LSTM weights
        for this coin.  Returns the most recently written fingerprint string
        (first 16 hex chars embedded in the filename), or None if nothing
        cached yet.

        This enables incremental fine-tuning: when today's data produces a
        new fingerprint, we locate yesterday's trained weights and warm-start
        from them rather than training from scratch.
        """
        pattern     = f"{self.coin}_lstm_*.pt"
        candidates  = sorted(
            Path(self.cache_dir).glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            return None
        # The filename is: {coin}_lstm_{fp16}.pt  — extract the fp part
        stem        = candidates[-1].stem          # e.g. "BTC_lstm_a3f9c2d18b4e7f01"
        parts       = stem.split("_", 2)           # ["BTC", "lstm", "a3f9c2d18b4e7f01"]
        if len(parts) == 3:
            fp16 = parts[2]
            log.info("Previous %s LSTM fingerprint found: %s…", self.coin, fp16[:8])
            return fp16
        return None

    # ------------------------------------------------------------------
    def _incremental_finetune(
        self,
        model:         "nn.Module",
        X_new_s:       np.ndarray,
        y_new_s:       np.ndarray,
        X_val_s:       np.ndarray,
        y_val_s:       np.ndarray,
        hp:            Dict,
        finetune_lr:   float = 2e-4,
        finetune_epochs: int = 15,
    ) -> "nn.Module":
        """
        Fine-tune a pre-trained model on ONLY the new rows it has never seen.

        Why this works:
        ───────────────────────────────────────────────────────────────────
        The model was already trained on all historical data.  When 1-3 new
        daily candles arrive, we do NOT retrain from scratch.  Instead we:
          1. Load the existing weights (captured all past patterns).
          2. Run a small number of gradient steps (15 epochs max, early
             stopping at patience 5) on ONLY the new samples.
          3. Use a much lower learning rate (2e-4 vs 1e-3) so the new data
             nudges the weights without overwriting what was learned before.
             This is called "catastrophic forgetting prevention".

        If X_new_s is empty (same-day restart, no new data) the function
        returns the model unchanged — no wasted computation.
        ───────────────────────────────────────────────────────────────────
        """
        if not TORCH_AVAILABLE:
            return model
        if X_new_s is None or len(X_new_s) == 0:
            log.info("  Incremental: no new samples — model unchanged")
            return model

        log.info(
            "  Incremental fine-tune: %d new samples  lr=%.1e  epochs=%d",
            len(X_new_s), finetune_lr, finetune_epochs,
        )
        return _train_torch_model(
            model, X_new_s, y_new_s, X_val_s, y_val_s,
            epochs=finetune_epochs,
            batch_size=max(8, hp.get("batch_size", 32)),
            lr=finetune_lr,
            device=self.device,
            patience=5,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        df:            pd.DataFrame,
        X:             np.ndarray,
        y:             np.ndarray,
        feature_names: List[str],
        val_split:     float = 0.15,
        force_retrain: bool  = False,
    ) -> "ModelTrainer":
        self._df            = df
        self._fp            = self.data_fingerprint(df)
        self._feature_names = feature_names

        n      = len(X)
        split  = int(n * (1 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        self._y_val_raw   = y_val
        self._y_train_raw = y_train

        # Detect how many NEW rows the model has never seen.
        # We do this by finding the previous fingerprint and reading the
        # number of rows that were used in that training run.
        # new_rows_count = current_row_count − previous_row_count
        # The new samples in X are the LAST new_rows_count entries.
        self._prev_fp         = self._find_previous_fingerprint()
        self._new_sample_count = 0
        if self._prev_fp is not None and self._prev_fp != self._fp[:16]:
            # Read the old row count from the fingerprint metadata file if
            # it exists, otherwise fall back to 0 (safe: full train).
            meta_path = Path(self.cache_dir) / f"{self.coin}_rowcount_{self._prev_fp}.txt"
            if meta_path.exists():
                try:
                    old_n = int(meta_path.read_text().strip())
                    self._new_sample_count = max(0, n - old_n)
                    log.info(
                        "%s incremental: %d new samples detected (was %d, now %d)",
                        self.coin, self._new_sample_count, old_n, n,
                    )
                except Exception:
                    self._new_sample_count = 0

        # Save current row count for next run
        rc_path = Path(self.cache_dir) / f"{self.coin}_rowcount_{self._fp[:16]}.txt"
        rc_path.write_text(str(n))
        # Clean up old rowcount files for this coin
        for old_rc in Path(self.cache_dir).glob(f"{self.coin}_rowcount_*.txt"):
            if old_rc != rc_path:
                try: old_rc.unlink()
                except Exception: pass


        # Store the last close price for each val sample so validate() can
        # convert log-returns back to price space for interpretable metrics.
        # Each sample i uses window ending at index (split + i + window - 1)
        # in the original df close array.
        close_vals = df["close"].dropna().values
        # val samples start at split index in X, which corresponds to close
        # prices starting at (split + window - 1) in the close array.
        # We need the "current" close (the last bar of each window).
        val_start_close_idx = split + self.window - 1
        self._val_current_prices = close_vals[
            val_start_close_idx: val_start_close_idx + len(y_val)
        ].astype(np.float32)
        # Pad if needed (edge case when df is shorter than expected)
        if len(self._val_current_prices) < len(y_val):
            pad = np.full(len(y_val) - len(self._val_current_prices),
                          close_vals[-1], dtype=np.float32)
            self._val_current_prices = np.concatenate([self._val_current_prices, pad])

        X_train_s, y_train_s, self._x_scaler, self._y_scaler = scale_features(X_train, y_train)
        self._X_train_s = X_train_s

        n_v, w_v, f_v = X_val.shape
        X_val_s = self._x_scaler.transform(X_val.reshape(-1, f_v)).reshape(n_v, w_v, f_v)
        y_val_s = self._y_scaler.transform(y_val)
        self._X_val_s = X_val_s
        self._y_val_s = y_val_s

        n_feat       = X.shape[2]
        close_series = df["close"].dropna()

        # ---- Prophet ----
        self._prophet_model = None
        if self.enable_prophet and PROPHET_AVAILABLE:
            if not force_retrain and self.cache.exists(self.coin, "prophet", self._fp):
                self._prophet_model = self.cache.load(self.coin, "prophet", self._fp)
                log.info("Prophet loaded from cache for %s", self.coin)
            else:
                log.info("Training Prophet for %s…", self.coin)
                self._prophet_model, _ = _train_prophet(close_series, max(HORIZONS))
                self.cache.save(self._prophet_model, self.coin, "prophet", self._fp)

        # ---- SARIMA ----
        self._sarima_model = None
        if self.enable_sarima and PMDARIMA_AVAILABLE:
            if not force_retrain and self.cache.exists(self.coin, "sarima", self._fp):
                self._sarima_model = self.cache.load(self.coin, "sarima", self._fp)
                log.info("SARIMA loaded from cache for %s", self.coin)
            else:
                log.info("Training SARIMA for %s…", self.coin)
                self._sarima_model = _train_sarima(close_series)
                self.cache.save(self._sarima_model, self.coin, "sarima", self._fp)

        # ---- Optuna hyperparameter search ----
        self._lstm_hparams = {
            "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
            "lr": 1e-3, "batch_size": 32, "bidirectional": False,
        }
        if TORCH_AVAILABLE and self.enable_lstm:
            hp_cache_key  = f"lstm_hp_{self.coin}_{self._fp[:8]}"
            hp_cache_path = Path(self.cache_dir) / f"{hp_cache_key}.pkl"
            # Also check for any previous HP file for this coin (different fingerprint)
            # so Optuna is not re-run just because the data date changed.
            prev_hp_files = sorted(
                Path(self.cache_dir).glob(f"lstm_hp_{self.coin}_*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )
            if not force_retrain and hp_cache_path.exists():
                with open(hp_cache_path, "rb") as fh:
                    self._lstm_hparams = pickle.load(fh)
                log.info("LSTM hyperparams loaded from exact cache: %s", self._lstm_hparams)
            elif not force_retrain and prev_hp_files:
                # Reuse the most recent HP search from a previous data version.
                # Architecture does not change when 1-3 new rows are added.
                with open(prev_hp_files[-1], "rb") as fh:
                    self._lstm_hparams = pickle.load(fh)
                # Save under current fingerprint so exact match next time
                with open(hp_cache_path, "wb") as fh:
                    pickle.dump(self._lstm_hparams, fh, protocol=5)
                log.info(
                    "LSTM hyperparams reused from previous run (no Optuna needed): %s",
                    self._lstm_hparams,
                )
            elif self.run_optuna:
                log.info("Running Optuna for LSTM hyperparams (%d trials)…", self.optuna_trials)
                self._lstm_hparams = tune_lstm_hyperparams(
                    X_train_s, y_train_s, X_val_s, y_val_s,
                    n_trials=self.optuna_trials, device=self.device,
                )
                with open(hp_cache_path, "wb") as fh:
                    pickle.dump(self._lstm_hparams, fh, protocol=5)

        # ---- LSTM ----
        self._lstm = None
        if TORCH_AVAILABLE and self.enable_lstm:
            hp = self._lstm_hparams
            self._lstm = LSTMModel(
                input_size=n_feat,
                hidden_size=hp.get("hidden_size", 128),
                num_layers=hp.get("num_layers", 2),
                output_size=len(HORIZONS),
                dropout=hp.get("dropout", 0.2),
                bidirectional=hp.get("bidirectional", False),
            )
            if not force_retrain and self.cache.torch_exists(self.coin, "lstm", self._fp):
                # Same fingerprint → same data → just load cached weights
                self._lstm = self.cache.load_torch(self._lstm, self.coin, "lstm", self._fp)
                log.info("LSTM loaded from cache for %s", self.coin)

            elif (not force_retrain
                  and self._prev_fp is not None
                  and self._prev_fp != self._fp[:16]
                  and self._new_sample_count > 0
                  and self.cache.torch_exists(self.coin, "lstm", self._prev_fp)):
                # New data arrived since last run → warm-start from previous
                # weights and fine-tune on ONLY the new rows.
                log.info(
                    "LSTM incremental fine-tune for %s  (%d new samples)",
                    self.coin, self._new_sample_count,
                )
                self._lstm = self.cache.load_torch(self._lstm, self.coin, "lstm", self._prev_fp)
                # Slice the LAST new_sample_count rows of the SCALED training set
                n_new  = self._new_sample_count
                X_new_s = X_train_s[-n_new:] if n_new <= len(X_train_s) else X_train_s
                y_new_s = y_train_s[-n_new:] if n_new <= len(y_train_s) else y_train_s
                self._lstm = self._incremental_finetune(
                    self._lstm, X_new_s, y_new_s, X_val_s, y_val_s, hp,
                )
                self.cache.save_torch(self._lstm, self.coin, "lstm", self._fp)

            else:
                log.info("Training LSTM from scratch for %s…", self.coin)
                self._lstm = _train_torch_model(
                    self._lstm, X_train_s, y_train_s, X_val_s, y_val_s,
                    epochs=self.epochs,
                    batch_size=hp.get("batch_size", 32),
                    lr=hp.get("lr", 1e-3),
                    device=self.device,
                    patience=20,
                )
                self.cache.save_torch(self._lstm, self.coin, "lstm", self._fp)

        # ---- Transformer ----
        self._transformer = None
        if TORCH_AVAILABLE and self.enable_transformer:
            self._transformer = TransformerModel(
                input_size=n_feat,
                output_size=len(HORIZONS),
                max_seq_len=max(self.window + 10, 500),
            )
            if not force_retrain and self.cache.torch_exists(self.coin, "transformer", self._fp):
                self._transformer = self.cache.load_torch(
                    self._transformer, self.coin, "transformer", self._fp
                )
                log.info("Transformer loaded from cache for %s", self.coin)

            elif (not force_retrain
                  and self._prev_fp is not None
                  and self._prev_fp != self._fp[:16]
                  and self._new_sample_count > 0
                  and self.cache.torch_exists(self.coin, "transformer", self._prev_fp)):
                log.info(
                    "Transformer incremental fine-tune for %s  (%d new samples)",
                    self.coin, self._new_sample_count,
                )
                self._transformer = self.cache.load_torch(
                    self._transformer, self.coin, "transformer", self._prev_fp
                )
                n_new   = self._new_sample_count
                X_new_s = X_train_s[-n_new:] if n_new <= len(X_train_s) else X_train_s
                y_new_s = y_train_s[-n_new:] if n_new <= len(y_train_s) else y_train_s
                # Reuse LSTM hparams for batch size / lr; use transformer defaults otherwise
                trans_hp = {"batch_size": 32, "lr": 1e-3}
                self._transformer = self._incremental_finetune(
                    self._transformer, X_new_s, y_new_s, X_val_s, y_val_s, trans_hp,
                )
                self.cache.save_torch(self._transformer, self.coin, "transformer", self._fp)

            else:
                log.info("Training Transformer from scratch for %s…", self.coin)
                self._transformer = _train_torch_model(
                    self._transformer, X_train_s, y_train_s, X_val_s, y_val_s,
                    epochs=self.epochs, batch_size=32, lr=1e-3,
                    device=self.device, patience=20,
                )
                self.cache.save_torch(self._transformer, self.coin, "transformer", self._fp)

        # ---- Stacking ensemble ----
        self._stacking_ridge  = None
        self._model_da_weights = {}   # directional-accuracy weights per model per horizon

        # Compute val-set DA weights for every model (used in DA-weighted ensemble)
        self._model_da_weights = self._compute_da_weights(X_val_s, y_val)

        # Train Ridge stacking on consistent log-return space
        self._stacking_ridge = self._train_stacking_oof(
            X_train, y_train, X_val, y_val, force_retrain=force_retrain,
        )

        self.cache.save((self._x_scaler, self._y_scaler), self.coin, "scalers", self._fp)
        return self

    # ------------------------------------------------------------------
    def _compute_da_weights(
        self,
        X_val_s: np.ndarray,
        y_val:   np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-horizon directional accuracy for each available model on
        the validation set.  Returns a dict {model_name: array of shape
        (n_horizons,)} where each value is the DA (0–1) for that horizon.
        Used to build a DA-weighted ensemble as a fallback / cross-check.
        """
        da: Dict[str, np.ndarray] = {}

        if not (TORCH_AVAILABLE and hasattr(self, "_y_scaler")):
            return da

        Xv = torch.tensor(X_val_s, dtype=torch.float32)

        for name, model in [("lstm", self._lstm), ("transformer", self._transformer)]:
            if model is None:
                continue
            model.eval()
            with torch.no_grad():
                pred_lr = self._y_scaler.inverse_transform(model(Xv).numpy())
            da_arr = np.array([
                float(np.mean(np.sign(y_val[:, i]) == np.sign(pred_lr[:, i])))
                for i in range(len(HORIZONS))
            ])
            da[name] = da_arr

        # Prophet: scalar forecast → log-return relative to last training close
        if self._prophet_model is not None and hasattr(self, "_df"):
            last_close = float(self._df["close"].dropna().iloc[-1])
            ph_price   = _prophet_predict_horizons(self._df["close"].dropna())
            da_arr     = np.zeros(len(HORIZONS))
            for i, p in enumerate(ph_price):
                if not np.isnan(p) and last_close > 0:
                    ph_dir   = np.sign(np.log(max(p, 1e-8) / max(last_close, 1e-8)))
                    true_dir = np.sign(y_val[:, i])
                    da_arr[i] = float(np.mean(true_dir == ph_dir))
            da["prophet"] = da_arr

        # SARIMA: same approach, only short horizons
        if self._sarima_model is not None:
            last_close = float(self._df["close"].dropna().iloc[-1])
            sa_price   = _sarima_predict_horizons(self._sarima_model)
            da_arr     = np.zeros(len(HORIZONS))
            for i, p in enumerate(sa_price):
                if i <= SARIMA_MAX_HORIZON_IDX and not np.isnan(p) and last_close > 0:
                    sa_dir   = np.sign(np.log(max(p, 1e-8) / max(last_close, 1e-8)))
                    true_dir = np.sign(y_val[:, i])
                    da_arr[i] = float(np.mean(true_dir == sa_dir))
            da["sarima"] = da_arr

        return da

    # ------------------------------------------------------------------
    def _train_stacking_oof(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray,
        X_val:         np.ndarray,
        y_val:         np.ndarray,
        force_retrain: bool = False,
        n_splits:      int  = 5,
    ) -> Optional[RidgeCV]:
        """
        Out-of-fold stacking — FIX for the space-mismatch bug.

        ─────────────────────────────────────────────────────────────────────
        ROOT CAUSE OF PREVIOUS NEGATIVE ENSEMBLE VALUES
        ─────────────────────────────────────────────────────────────────────
        The old code mixed two incompatible spaces inside the same Ridge fit:
          • LSTM / Transformer OOF predictions were in log-return space
            (inverse_transform of scaled log-returns → still log-returns, e.g.
            0.05, -0.10) because inverse_transform just un-scales, it does NOT
            convert log-returns to prices.
          • Prophet / SARIMA predictions were in PRICE space ($76,000).
          • meta_y = y_train[mask] was log-return space.

        Ridge tried to learn a linear combination of a 0.05-scale column and
        a $76,000-scale column to predict a 0.05-scale target.  The resulting
        coefficients were meaningless, producing large negative predictions
        that, when later used in forecast() as price-space values, gave
        -$53,000 ensemble outputs.

        FIX: Everything in this function is now consistently in LOG-RETURN
        space:
          • LSTM / Transformer OOF preds: kept as unscaled log-returns
            (inverse_transform of MinMaxScaler output).
          • Prophet / SARIMA: NOT included in meta_X here; they are in price
            space and cannot be mixed. Their contribution is handled in
            forecast() through the DA-weighted ensemble instead.
          • meta_y = y_train[mask]: log-returns (unchanged).

        At forecast() time the Ridge output is a log-return prediction, which
        is then converted to price via base_price * exp(ridge_output). This
        guarantees the ensemble is always a valid positive price.
        ─────────────────────────────────────────────────────────────────────
        """
        stack_cache_path = Path(self.cache_dir) / f"{self.coin}_stacking_oof_{self._fp[:16]}.pkl"
        if not force_retrain and stack_cache_path.exists():
            with open(stack_cache_path, "rb") as fh:
                ridge = pickle.load(fh)
            log.info("OOF stacking meta-learner loaded from cache for %s", self.coin)
            return ridge

        if not TORCH_AVAILABLE:
            log.info("Torch not available — skipping OOF stacking")
            return None

        n = len(X_train)
        if n < 200:
            log.info("Too few samples for OOF stacking (%d) — falling back to val stacking", n)
            return self._train_stacking_val(X_val, y_val, force_retrain)

        log.info("Building OOF stacking meta-features for %s…", self.coin)
        tscv      = TimeSeriesSplit(n_splits=n_splits)
        n_feat    = X_train.shape[2]
        lstm_oof  = np.zeros((n, len(HORIZONS)))
        trans_oof = np.zeros((n, len(HORIZONS)))
        mask      = np.zeros(n, dtype=bool)
        hp        = self._lstm_hparams

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            log.info("  OOF fold %d/%d  train=%d  val=%d",
                     fold_idx + 1, n_splits, len(tr_idx), len(val_idx))
            X_tr_f, y_tr_f   = X_train[tr_idx], y_train[tr_idx]
            X_val_f, y_val_f = X_train[val_idx], y_train[val_idx]

            if len(X_tr_f) < 50:
                continue

            X_tr_s, y_tr_s, xsc_f, ysc_f = scale_features(X_tr_f, y_tr_f)
            nv, wv, fv = X_val_f.shape
            X_val_s    = xsc_f.transform(X_val_f.reshape(-1, fv)).reshape(nv, wv, fv)
            y_val_s    = ysc_f.transform(y_val_f)

            # LSTM fold
            if self._lstm is not None:
                tmp_lstm = LSTMModel(
                    input_size=n_feat,
                    hidden_size=hp.get("hidden_size", 128),
                    num_layers=hp.get("num_layers", 2),
                    output_size=len(HORIZONS),
                    dropout=hp.get("dropout", 0.2),
                    bidirectional=hp.get("bidirectional", False),
                )
                tmp_lstm = _train_torch_model(
                    tmp_lstm, X_tr_s, y_tr_s, X_val_s, y_val_s,
                    epochs=40, batch_size=hp.get("batch_size", 32),
                    lr=hp.get("lr", 1e-3), device=self.device, patience=8,
                )
                tmp_lstm.eval()
                Xv_t = torch.tensor(X_val_s, dtype=torch.float32)
                with torch.no_grad():
                    # FIX: inverse_transform un-scales to log-return space.
                    # Do NOT convert to price space here — keep log-returns
                    # so that meta_X and meta_y are in the same space.
                    pred_lr = ysc_f.inverse_transform(tmp_lstm(Xv_t).numpy())
                lstm_oof[val_idx] = pred_lr

            # Transformer fold
            if self._transformer is not None:
                tmp_trans = TransformerModel(
                    input_size=n_feat,
                    output_size=len(HORIZONS),
                    max_seq_len=max(self.window + 10, 500),
                )
                tmp_trans = _train_torch_model(
                    tmp_trans, X_tr_s, y_tr_s, X_val_s, y_val_s,
                    epochs=40, batch_size=32, lr=1e-3,
                    device=self.device, patience=8,
                )
                tmp_trans.eval()
                Xv_t = torch.tensor(X_val_s, dtype=torch.float32)
                with torch.no_grad():
                    # FIX: same as above — keep in log-return space
                    pred_lr = ysc_f.inverse_transform(tmp_trans(Xv_t).numpy())
                trans_oof[val_idx] = pred_lr

            mask[val_idx] = True

        if not mask.any():
            log.warning("OOF stacking produced no valid folds — falling back to val stacking")
            return self._train_stacking_val(X_val, y_val, force_retrain)

        # Build meta matrix — ONLY LSTM + Transformer (both in log-return space).
        # FIX: Prophet/SARIMA are in price space and are NOT included here.
        # Their contribution is handled separately in forecast() via DA-weighted blending.
        meta_cols = []
        if self._lstm is not None:
            meta_cols.append(lstm_oof[mask])
        if self._transformer is not None:
            meta_cols.append(trans_oof[mask])

        if len(meta_cols) < 2:
            log.info("Not enough deep-learning OOF models — falling back to val stacking")
            return self._train_stacking_val(X_val, y_val, force_retrain)

        # meta_X: (n_masked, n_models * n_horizons) — all log-return space
        # meta_y: (n_masked, n_horizons)            — log-return space ✓
        meta_X = np.hstack(meta_cols)
        meta_y = y_train[mask]   # log-returns — same space as meta_X ✓

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(meta_X, meta_y)
        log.info("OOF stacking RidgeCV fitted for %s  alpha=%.4f  meta_X=%s",
                 self.coin, ridge.alpha_, meta_X.shape)

        with open(stack_cache_path, "wb") as fh:
            pickle.dump(ridge, fh, protocol=5)
        return ridge

    # ------------------------------------------------------------------
    def _train_stacking_val(
        self,
        X_val_s:       np.ndarray,
        y_val_raw:     np.ndarray,
        force_retrain: bool = False,
    ) -> Optional[RidgeCV]:
        """
        Fallback val-set stacking — FIX applied here too.
        Only LSTM + Transformer are included in meta_X (both in log-return
        space). Prophet/SARIMA are excluded for the same space-mismatch
        reason described in _train_stacking_oof.
        """
        stack_cache_path = Path(self.cache_dir) / f"{self.coin}_stacking_{self._fp[:16]}.pkl"
        if not force_retrain and stack_cache_path.exists():
            with open(stack_cache_path, "rb") as fh:
                return pickle.load(fh)

        val_preds_list = []

        if TORCH_AVAILABLE:
            Xv = torch.tensor(X_val_s, dtype=torch.float32)
            # FIX: inverse_transform → log-return space, consistent with meta_y
            if self._lstm is not None:
                self._lstm.eval()
                with torch.no_grad():
                    p_lr = self._y_scaler.inverse_transform(self._lstm(Xv).numpy())
                val_preds_list.append(p_lr)

            if self._transformer is not None:
                self._transformer.eval()
                with torch.no_grad():
                    p_lr = self._y_scaler.inverse_transform(self._transformer(Xv).numpy())
                val_preds_list.append(p_lr)

        # FIX: Do NOT add Prophet or SARIMA here — they are price-space values
        # and would break the Ridge fit. They are handled in forecast() instead.

        if len(val_preds_list) < 2:
            log.info("Not enough models for val stacking — will use DA-weighted ensemble")
            return None

        meta_X = np.hstack(val_preds_list)
        ridge  = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(meta_X, y_val_raw)   # both log-return space ✓
        log.info("Val stacking RidgeCV fitted for %s  alpha=%.4f", self.coin, ridge.alpha_)

        with open(stack_cache_path, "wb") as fh:
            pickle.dump(ridge, fh, protocol=5)
        return ridge

    # ------------------------------------------------------------------
    def validate(self) -> Dict:
        """
        Compute validation metrics for ALL 5 models: LSTM, Transformer,
        Prophet, SARIMA, and Ensemble.

        FIXES applied:
        ─────────────────────────────────────────────────────────────────────
        FIX 1 (MAPE explosion — millions %):
          Pass current_prices to compute_metrics so log-returns are
          converted to price space before computing MAPE. Without this,
          the denominator approaches 0 for small returns, producing
          MAPE values of 494928%, 871424%, etc.

        FIX 2 (only 2 models shown in UI):
          Add Prophet, SARIMA, and Ensemble keys to the returned metrics
          dict. The frontend ModelAccuracyCards iterates MODELS and looks
          up metrics[model.key]; previously "prophet", "sarima", and
          "ensemble" were missing so 3 out of 5 cards were blank.
        ─────────────────────────────────────────────────────────────────────
        """
        metrics: Dict = {}
        if not hasattr(self, "_y_val_raw") or len(self._y_val_raw) < 2:
            return metrics

        y_true = self._y_val_raw   # shape: (n_val, n_horizons) — log-returns

        # Use stored current prices (close at the start of each val window)
        # so compute_metrics can work in price space for MAE / RMSE / MAPE.
        cp = getattr(self, "_val_current_prices", None)  # shape: (n_val,)

        Xv = None
        if TORCH_AVAILABLE and hasattr(self, "_X_val_s"):
            Xv = torch.tensor(self._X_val_s, dtype=torch.float32)

        # ── LSTM ──────────────────────────────────────────────────────────
        if self._lstm is not None and Xv is not None:
            self._lstm.eval()
            with torch.no_grad():
                lstm_pred_lr = self._y_scaler.inverse_transform(self._lstm(Xv).numpy())
            for i, label in enumerate(HORIZON_LABELS):
                metrics.setdefault("lstm", {})[label] = compute_metrics(
                    y_true[:, i],
                    lstm_pred_lr[:, i],
                    f"lstm/{label}",
                    current_prices=cp,  # FIX: enables price-space MAPE
                )

        # ── Transformer ───────────────────────────────────────────────────
        if self._transformer is not None and Xv is not None:
            self._transformer.eval()
            with torch.no_grad():
                trans_pred_lr = self._y_scaler.inverse_transform(self._transformer(Xv).numpy())
            for i, label in enumerate(HORIZON_LABELS):
                metrics.setdefault("transformer", {})[label] = compute_metrics(
                    y_true[:, i],
                    trans_pred_lr[:, i],
                    f"transformer/{label}",
                    current_prices=cp,  # FIX: enables price-space MAPE
                )

        # ── Prophet ───────────────────────────────────────────────────────
        # Prophet gives one price prediction per horizon (not per val sample).
        # Convert to a log-return relative to the last known training close,
        # then broadcast to all val samples for direction comparison.
        if self._prophet_model is not None and hasattr(self, "_df"):
            last_close = float(self._df["close"].dropna().iloc[-1])
            ph_prices  = _prophet_predict_horizons(self._df["close"].dropna())
            for i, label in enumerate(HORIZON_LABELS):
                if np.isnan(ph_prices[i]) or last_close <= 0:
                    continue
                # Convert Prophet's price prediction to log-return
                ph_lr     = float(np.log(max(ph_prices[i], 1e-8) / max(last_close, 1e-8)))
                ph_pred   = np.full(len(y_true), ph_lr)
                metrics.setdefault("prophet", {})[label] = compute_metrics(
                    y_true[:, i],
                    ph_pred,
                    f"prophet/{label}",
                    current_prices=cp,  # FIX: price-space MAPE
                )

        # ── SARIMA ────────────────────────────────────────────────────────
        # Same approach as Prophet; only reliable for short horizons.
        if self._sarima_model is not None and hasattr(self, "_df"):
            last_close = float(self._df["close"].dropna().iloc[-1])
            sa_prices  = _sarima_predict_horizons(self._sarima_model)
            for i, label in enumerate(HORIZON_LABELS):
                if i > SARIMA_MAX_HORIZON_IDX:
                    continue
                if np.isnan(sa_prices[i]) or last_close <= 0:
                    continue
                sa_lr   = float(np.log(max(sa_prices[i], 1e-8) / max(last_close, 1e-8)))
                sa_pred = np.full(len(y_true), sa_lr)
                metrics.setdefault("sarima", {})[label] = compute_metrics(
                    y_true[:, i],
                    sa_pred,
                    f"sarima/{label}",
                    current_prices=cp,  # FIX: price-space MAPE
                )

        # ── Ensemble ──────────────────────────────────────────────────────
        # Build the same ensemble prediction that forecast() produces but
        # on the val set, so we can report its accuracy in the UI.
        if Xv is not None and (self._lstm is not None or self._transformer is not None):
            # Collect log-return preds from deep-learning models
            dl_preds_lr: List[np.ndarray] = []
            if self._lstm is not None:
                dl_preds_lr.append(lstm_pred_lr)
            if self._transformer is not None:
                dl_preds_lr.append(trans_pred_lr)

            if self._stacking_ridge is not None and len(dl_preds_lr) >= 2:
                # Ridge was trained on [lstm_lr | transformer_lr] → log-return output
                n_models_train = self._stacking_ridge.n_features_in_ // len(HORIZONS)
                if len(dl_preds_lr) == n_models_train:
                    ens_meta  = np.hstack(dl_preds_lr)          # (n_val, 2*n_horizons)
                    ens_lr    = self._stacking_ridge.predict(ens_meta)  # (n_val, n_horizons)
                else:
                    ens_lr    = np.mean(dl_preds_lr, axis=0)
            else:
                ens_lr = np.mean(dl_preds_lr, axis=0)

            # Blend in Prophet/SARIMA via DA weights (they are in log-return
            # form after the conversion above) to make the ensemble truly use
            # all 4 models the same way forecast() does.
            ph_blend = None
            sa_blend = None
            if "prophet" in metrics and hasattr(self, "_df") and last_close > 0:
                ph_prices_v = _prophet_predict_horizons(self._df["close"].dropna())
                ph_lr_arr   = np.array([
                    np.log(max(ph_prices_v[i], 1e-8) / max(last_close, 1e-8))
                    if not np.isnan(ph_prices_v[i]) else 0.0
                    for i in range(len(HORIZONS))
                ])
                ph_blend = np.tile(ph_lr_arr, (len(y_true), 1))

            if "sarima" in metrics:
                sa_prices_v = _sarima_predict_horizons(self._sarima_model)
                sa_lr_arr   = np.array([
                    np.log(max(sa_prices_v[i], 1e-8) / max(last_close, 1e-8))
                    if (i <= SARIMA_MAX_HORIZON_IDX and not np.isnan(sa_prices_v[i])) else 0.0
                    for i in range(len(HORIZONS))
                ])
                sa_blend = np.tile(sa_lr_arr, (len(y_true), 1))

            # Final ensemble: average deep-model Ridge output with stat models
            blend_preds = [ens_lr]
            if ph_blend is not None:
                blend_preds.append(ph_blend)
            if sa_blend is not None:
                blend_preds.append(sa_blend)
            final_ens_lr = np.mean(blend_preds, axis=0)   # (n_val, n_horizons)

            for i, label in enumerate(HORIZON_LABELS):
                metrics.setdefault("ensemble", {})[label] = compute_metrics(
                    y_true[:, i],
                    final_ens_lr[:, i],
                    f"ensemble/{label}",
                    current_prices=cp,  # FIX: price-space MAPE
                )

        return metrics

    # ------------------------------------------------------------------
    def forecast(
        self,
        latest_window: np.ndarray,
        current_price: float = 0.0,
        live_price:    Optional[float] = None,
    ) -> ForecastResult:
        """
        Generate forecasts.

        FIX: Ensemble is now assembled entirely in log-return space, then
        converted to price space at the very end. This guarantees:
          1. The Ridge meta-learner output is a log-return (same space it
             was trained on), not a meaningless mix of scales.
          2. Converting log-returns via base_price * exp(log_ret) always
             produces a positive price — negative ensemble values are
             impossible.
          3. Prophet / SARIMA contributions are also converted to
             log-returns before blending so everything is on the same scale.

        FIX: Confidence intervals are clamped so the lower bound is never
        below 1 % of current price (prevents negative CI display in the UI).
        """
        result     = ForecastResult(coin=self.coin, current_price=live_price or current_price)
        base_price = live_price or current_price

        # ── Prophet (price space output — stored for display) ─────────────
        if self._prophet_model is not None and hasattr(self, "_df"):
            result.prophet_preds = _prophet_predict_horizons(self._df["close"].dropna())

        # ── SARIMA (price space output — stored for display) ──────────────
        if self._sarima_model is not None:
            result.sarima_preds = _sarima_predict_horizons(self._sarima_model)

        # ── Deep learning models (log-return → price) ─────────────────────
        dl_log_rets: List[np.ndarray] = []   # log-return preds for Ridge stacking

        if TORCH_AVAILABLE and hasattr(self, "_x_scaler"):
            n, w, f = latest_window.shape
            win_s   = self._x_scaler.transform(latest_window.reshape(-1, f)).reshape(n, w, f)
            Xw      = torch.tensor(win_s, dtype=torch.float32)

            if self._lstm is not None:
                mean_t, std_t = self._lstm.mc_forward(Xw, n_samples=MC_SAMPLES)
                log_ret_pred  = self._y_scaler.inverse_transform(mean_t.detach().numpy())[0]
                log_ret_std   = std_t.detach().numpy()[0] / (self._y_scaler.scale_ + 1e-8)
                # FIX: Cap std to max 30 % of current price to prevent
                # extreme CI widths that make the lower bound negative.
                log_ret_std = np.clip(log_ret_std, 0.0, 0.30)
                result.lstm_preds       = base_price * np.exp(log_ret_pred)
                result.lstm_uncertainty = result.lstm_preds * log_ret_std
                dl_log_rets.append(log_ret_pred)

            if self._transformer is not None:
                mean_t, std_t = self._transformer.mc_forward(Xw, n_samples=MC_SAMPLES)
                log_ret_pred  = self._y_scaler.inverse_transform(mean_t.detach().numpy())[0]
                log_ret_std   = std_t.detach().numpy()[0] / (self._y_scaler.scale_ + 1e-8)
                # FIX: Cap std
                log_ret_std = np.clip(log_ret_std, 0.0, 0.30)
                result.transformer_preds       = base_price * np.exp(log_ret_pred)
                result.transformer_uncertainty = result.transformer_preds * log_ret_std
                dl_log_rets.append(log_ret_pred)

        # ── Build ensemble in log-return space ────────────────────────────
        if len(dl_log_rets) == 0:
            # No deep models available — fall back to mean of stat models
            stat_preds = []
            if result.prophet_preds is not None:
                stat_preds.append(result.prophet_preds)
            if result.sarima_preds is not None:
                sa = np.where(np.isnan(result.sarima_preds),
                              result.prophet_preds if result.prophet_preds is not None
                              else np.full(len(HORIZONS), base_price),
                              result.sarima_preds)
                stat_preds.append(sa)
            if stat_preds:
                result.ensemble_preds = np.nanmean(stat_preds, axis=0)
            return result

        # Step 1 — Ridge log-return ensemble from deep-learning models
        if self._stacking_ridge is not None and len(dl_log_rets) >= 2:
            n_models_train = self._stacking_ridge.n_features_in_ // len(HORIZONS)
            if len(dl_log_rets) == n_models_train:
                meta_X    = np.hstack([lr[np.newaxis, :] for lr in dl_log_rets])
                ens_lr    = self._stacking_ridge.predict(meta_X)[0]
            else:
                ens_lr    = np.mean(dl_log_rets, axis=0)
        else:
            ens_lr = np.mean(dl_log_rets, axis=0)

        # Step 2 — Blend in Prophet / SARIMA as log-returns
        # Convert price-space predictions to log-returns relative to base_price
        # and blend into the ensemble with equal weight alongside the Ridge output.
        blend_log_rets = [ens_lr]

        if result.prophet_preds is not None and base_price > 0:
            ph_lr = np.array([
                float(np.log(max(p, 1e-8) / max(base_price, 1e-8)))
                if not np.isnan(p) else ens_lr[i]
                for i, p in enumerate(result.prophet_preds)
            ])
            blend_log_rets.append(ph_lr)

        if result.sarima_preds is not None and base_price > 0:
            sa_lr = np.array([
                float(np.log(max(p, 1e-8) / max(base_price, 1e-8)))
                if (not np.isnan(p) and i <= SARIMA_MAX_HORIZON_IDX) else ens_lr[i]
                for i, p in enumerate(result.sarima_preds)
            ])
            blend_log_rets.append(sa_lr)

        # Final blended log-return → convert to price
        final_lr              = np.mean(blend_log_rets, axis=0)
        result.ensemble_preds = base_price * np.exp(final_lr)

        # ── Confidence interval (always positive) ─────────────────────────
        uncertainties = []
        if result.lstm_uncertainty is not None:
            uncertainties.append(result.lstm_uncertainty)
        if result.transformer_uncertainty is not None:
            uncertainties.append(result.transformer_uncertainty)

        if uncertainties and result.ensemble_preds is not None:
            combined_std           = np.mean(uncertainties, axis=0)
            lower                  = result.ensemble_preds - 1.96 * combined_std
            upper                  = result.ensemble_preds + 1.96 * combined_std
            # FIX: Clamp lower bound so it is never below 1 % of current price
            min_price              = max(base_price * 0.01, 1e-4)
            result.ensemble_lower  = np.maximum(lower, min_price)
            result.ensemble_upper  = np.maximum(upper, result.ensemble_preds)

        result.metrics = self.validate()
        return result

    # ------------------------------------------------------------------
    def walk_forward_backtest(
        self,
        df:           pd.DataFrame,
        X:            np.ndarray,
        y:            np.ndarray,
        n_test_steps: int = 60,
        step_size:    int = 7,
    ) -> Dict:
        """Walk-forward backtest. Primary metric is directional accuracy."""
        if not TORCH_AVAILABLE or self._lstm is None:
            log.warning("Walk-forward backtest requires PyTorch LSTM.")
            return {}

        log.info("Running walk-forward backtest for %s (%d steps)…", self.coin, n_test_steps)
        n_total   = len(X)
        train_end = n_total - n_test_steps

        if train_end < 100:
            log.warning("Not enough data for walk-forward backtest (need 100+ train samples)")
            return {}

        all_true:  List[np.ndarray] = []
        all_preds: List[np.ndarray] = []

        for step in range(0, n_test_steps - max(HORIZONS), step_size):
            t = train_end + step
            if t >= n_total - 1:
                break

            X_tr, y_tr = X[:t], y[:t]
            X_te, y_te = X[t: t + 1], y[t: t + 1]

            n_v          = max(1, int(len(X_tr) * 0.1))
            X_tr_s, y_tr_s, xsc, ysc = scale_features(X_tr[:-n_v], y_tr[:-n_v])
            n, w, f      = X_te.shape
            X_te_s       = xsc.transform(X_te.reshape(-1, f)).reshape(n, w, f)

            n_val_v, w_val, f_val = X_tr[-n_v:].shape
            X_val_wf_s = xsc.transform(X_tr[-n_v:].reshape(-1, f_val)).reshape(n_val_v, w_val, f_val)
            y_val_wf_s = ysc.transform(y_tr[-n_v:])

            hp       = self._lstm_hparams
            tmp_lstm = LSTMModel(
                input_size=f,
                hidden_size=hp.get("hidden_size", 128),
                num_layers=hp.get("num_layers", 2),
                output_size=len(HORIZONS),
                dropout=hp.get("dropout", 0.2),
                bidirectional=hp.get("bidirectional", False),
            )
            tmp_lstm = _train_torch_model(
                tmp_lstm, X_tr_s, y_tr_s, X_val_wf_s, y_val_wf_s,
                epochs=25, batch_size=32, lr=1e-3, device=self.device, patience=6,
            )

            tmp_lstm.eval()
            Xt = torch.tensor(X_te_s, dtype=torch.float32)
            with torch.no_grad():
                pred = ysc.inverse_transform(tmp_lstm(Xt).numpy())

            all_true.append(y_te[0])
            all_preds.append(pred[0])

        if not all_true:
            return {}

        true_arr = np.array(all_true)
        pred_arr = np.array(all_preds)

        metrics: Dict = {}
        for i, label in enumerate(HORIZON_LABELS):
            if i < true_arr.shape[1]:
                metrics.setdefault("lstm_walkforward", {})[label] = compute_metrics(
                    true_arr[:, i], pred_arr[:, i], f"walkforward/lstm/{label}"
                )
        return metrics