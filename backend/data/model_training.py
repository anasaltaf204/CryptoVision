

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
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

HORIZONS = [7, 30, 90, 180, 365]
HORIZON_LABELS = ["1w", "1m", "3m", "6m", "1y"]

# Number of MC Dropout forward passes for uncertainty estimation
MC_SAMPLES = 50


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict[str, float]:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    direction = (
        float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
        if len(y_true) > 1 else 0.0
    )
    log.info(
        "Metrics [%s]: MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  DirAcc=%.2f",
        label, mae, rmse, mape, direction,
    )
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "dir_acc": direction}


# ---------------------------------------------------------------------------
# PyTorch models with MC Dropout
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        """
        2-layer LSTM with MC Dropout.
        Dropout is active at inference too (via mc_forward) for uncertainty estimation.
        """
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            output_size: int = 5,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.fc(self.dropout(out[:, -1, :]))

        def mc_forward(self, x: "torch.Tensor", n_samples: int = MC_SAMPLES) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Run n_samples forward passes with dropout active → mean + std."""
            self.train()  # keep dropout active
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
            self.eval()
            return preds.mean(dim=0), preds.std(dim=0)

    class TransformerModel(nn.Module):
        """
        Encoder-only Transformer with MC Dropout uncertainty.
        """
        def __init__(
            self,
            input_size: int,
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 2,
            output_size: int = 5,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x)
            x = self.encoder(x)
            return self.fc(self.dropout(x.mean(dim=1)))

        def mc_forward(self, x: "torch.Tensor", n_samples: int = MC_SAMPLES) -> Tuple["torch.Tensor", "torch.Tensor"]:
            self.train()
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
            self.eval()
            return preds.mean(dim=0), preds.std(dim=0)

    def _train_torch_model(
        model: "nn.Module",
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
        patience: int = 10,
    ) -> "nn.Module":
        """Train with early stopping on validation loss."""
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
        yt = torch.tensor(y_train, dtype=torch.float32).to(device)
        Xv = torch.tensor(X_val,   dtype=torch.float32).to(device)
        yv = torch.tensor(y_val,   dtype=torch.float32).to(device)

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                optim.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                epoch_loss += loss.item() * len(xb)
            avg_train = epoch_loss / len(Xt)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xv), yv).item()
            model.train()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
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
# Optuna hyperparameter search (free)
# ---------------------------------------------------------------------------

def tune_lstm_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 15,
    device: str = "cpu",
) -> Dict:
    """Use Optuna to search LSTM hidden_size, num_layers, dropout, lr."""
    if not OPTUNA_AVAILABLE or not TORCH_AVAILABLE:
        log.info("Optuna/Torch not available — using default hyperparams")
        return {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32}

    n_feat = X_train.shape[2]

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        num_layers  = trial.suggest_int("num_layers", 1, 3)
        dropout     = trial.suggest_float("dropout", 0.1, 0.4)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = LSTMModel(
            input_size=n_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=len(HORIZONS),
            dropout=dropout,
        )
        model = _train_torch_model(
            model, X_train, y_train, X_val, y_val,
            epochs=20, batch_size=batch_size, lr=lr, device=device, patience=5,
        )
        model.eval()
        Xv = torch.tensor(X_val, dtype=torch.float32)
        with torch.no_grad():
            pred = model(Xv).numpy()
        return float(mean_squared_error(y_val, pred))

    study = optuna.create_study(direction="minimize")
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
    df = pd.DataFrame({
        "ds": close_series.index.tz_localize(None)
              if close_series.index.tz is not None else close_series.index,
        "y": close_series.values,
    })
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = m.predict(future)
    return m, forecast


def _prophet_predict_horizons(close_series: pd.Series) -> np.ndarray:
    if not PROPHET_AVAILABLE:
        return np.full(len(HORIZONS), np.nan)
    try:
        m, forecast = _train_prophet(close_series, max(HORIZONS))
        tail = forecast.tail(max(HORIZONS))
        return np.array([float(tail.iloc[h - 1]["yhat"]) for h in HORIZONS])
    except Exception as exc:
        log.warning("Prophet predict failed: %s", exc)
        return np.full(len(HORIZONS), np.nan)


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

def _train_sarima(close_series: pd.Series):
    if not PMDARIMA_AVAILABLE:
        raise RuntimeError("pmdarima not installed")
    log_close = np.log(close_series.values)
    model = pm.auto_arima(
        log_close, start_p=1, start_q=1, max_p=3, max_q=3,
        d=1, seasonal=False, information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore",
    )
    log.info("SARIMA order: %s", model.order)
    return model


def _sarima_predict_horizons(model) -> np.ndarray:
    if not PMDARIMA_AVAILABLE:
        return np.full(len(HORIZONS), np.nan)
    try:
        return np.array([float(np.exp(model.predict(n_periods=h)[-1])) for h in HORIZONS])
    except Exception as exc:
        log.warning("SARIMA predict failed: %s", exc)
        return np.full(len(HORIZONS), np.nan)


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def scale_features(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    n, w, f = X.shape
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
# ForecastResult (now includes uncertainty intervals)
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    coin: str
    horizons: List[str] = field(default_factory=lambda: HORIZON_LABELS)
    prophet_preds:      Optional[np.ndarray] = None
    sarima_preds:       Optional[np.ndarray] = None
    lstm_preds:         Optional[np.ndarray] = None
    lstm_uncertainty:   Optional[np.ndarray] = None   # MC Dropout std
    transformer_preds:  Optional[np.ndarray] = None
    transformer_uncertainty: Optional[np.ndarray] = None
    ensemble_preds:     Optional[np.ndarray] = None   # stacking ridge
    ensemble_lower:     Optional[np.ndarray] = None   # ensemble - 1.96*std
    ensemble_upper:     Optional[np.ndarray] = None   # ensemble + 1.96*std
    metrics:            Dict = field(default_factory=dict)
    current_price:      Optional[float] = None

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
                    p = float(arr[i])
                    change = ((p - self.current_price) / self.current_price * 100
                              if self.current_price and self.current_price > 0 else None)
                    entry = {
                        "price": round(p, 4),
                        "change_pct": round(float(change), 2) if change is not None else None,
                    }
                    if unc is not None and i < len(unc):
                        entry["uncertainty_std"] = round(float(unc[i]), 4)
                        entry["ci_lower"] = round(p - 1.96 * float(unc[i]), 4)
                        entry["ci_upper"] = round(p + 1.96 * float(unc[i]), 4)
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
    coin: str
    cache_dir: str = ".model_cache"
    epochs: int = 50
    window: int = 30
    device: str = "cpu"
    enable_prophet:     bool = True
    enable_sarima:      bool = True
    enable_lstm:        bool = True
    enable_transformer: bool = True
    run_optuna:         bool = True
    optuna_trials:      int = 15

    def __post_init__(self):
        self.cache = ModelCache(self.cache_dir)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            log.info("Using GPU: %s", torch.cuda.get_device_name(0))

    def data_fingerprint(self, df: pd.DataFrame) -> str:
        key = f"{self.coin}_{len(df)}_{df.index[-1].isoformat()}"
        return hashlib.sha256(key.encode()).hexdigest()

    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        val_split: float = 0.15,
        force_retrain: bool = False,
    ) -> "ModelTrainer":
        self._df = df
        self._fp = self.data_fingerprint(df)
        self._feature_names = feature_names
        n = len(X)
        split = int(n * (1 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        self._y_val_raw = y_val          # raw prices for metric computation
        self._y_train_raw = y_train

        X_train_s, y_train_s, self._x_scaler, self._y_scaler = scale_features(X_train, y_train)
        self._X_train_s = X_train_s

        n_v, w_v, f_v = X_val.shape
        X_val_s = self._x_scaler.transform(X_val.reshape(-1, f_v)).reshape(n_v, w_v, f_v)
        y_val_s = self._y_scaler.transform(y_val)
        self._X_val_s = X_val_s
        self._y_val_s = y_val_s

        n_feat = X.shape[2]
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

        # ---- Optuna hyperparameter search for LSTM ----
        self._lstm_hparams = {"hidden_size": 128, "num_layers": 2, "dropout": 0.2,
                              "lr": 1e-3, "batch_size": 32}
        if TORCH_AVAILABLE and self.enable_lstm:
            hp_cache_key = f"lstm_hp_{self.coin}_{self._fp[:8]}"
            hp_cache_path = Path(self.cache_dir) / f"{hp_cache_key}.pkl"
            if not force_retrain and hp_cache_path.exists():
                with open(hp_cache_path, "rb") as fh:
                    self._lstm_hparams = pickle.load(fh)
                log.info("LSTM hyperparams loaded from cache: %s", self._lstm_hparams)
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
            )
            if not force_retrain and self.cache.torch_exists(self.coin, "lstm", self._fp):
                self._lstm = self.cache.load_torch(self._lstm, self.coin, "lstm", self._fp)
                log.info("LSTM loaded from cache for %s", self.coin)
            else:
                log.info("Training LSTM for %s…", self.coin)
                self._lstm = _train_torch_model(
                    self._lstm, X_train_s, y_train_s, X_val_s, y_val_s,
                    epochs=self.epochs,
                    batch_size=hp.get("batch_size", 32),
                    lr=hp.get("lr", 1e-3),
                    device=self.device,
                    patience=10,
                )
                self.cache.save_torch(self._lstm, self.coin, "lstm", self._fp)

        # ---- Transformer ----
        self._transformer = None
        if TORCH_AVAILABLE and self.enable_transformer:
            self._transformer = TransformerModel(input_size=n_feat, output_size=len(HORIZONS))
            if not force_retrain and self.cache.torch_exists(self.coin, "transformer", self._fp):
                self._transformer = self.cache.load_torch(self._transformer, self.coin, "transformer", self._fp)
                log.info("Transformer loaded from cache for %s", self.coin)
            else:
                log.info("Training Transformer for %s…", self.coin)
                self._transformer = _train_torch_model(
                    self._transformer, X_train_s, y_train_s, X_val_s, y_val_s,
                    epochs=self.epochs, batch_size=32, lr=1e-3,
                    device=self.device, patience=10,
                )
                self.cache.save_torch(self._transformer, self.coin, "transformer", self._fp)

        # ---- Stacking ensemble (Ridge meta-learner) ----
        self._stacking_ridge = None
        self._stacking_ridge = self._train_stacking(
            X_val_s, y_val, force_retrain=force_retrain,
        )

        # Save scalers
        self.cache.save((self._x_scaler, self._y_scaler), self.coin, "scalers", self._fp)
        return self

    def _train_stacking(
        self,
        X_val_s: np.ndarray,
        y_val_raw: np.ndarray,
        force_retrain: bool = False,
    ) -> Optional[Ridge]:
        """
        Stacking ensemble: collect each base model's val predictions,
        then fit a Ridge regression to learn optimal weights.
        Far better than naive mean averaging.
        """
        stack_cache_path = Path(self.cache_dir) / f"{self.coin}_stacking_{self._fp[:16]}.pkl"
        if not force_retrain and stack_cache_path.exists():
            with open(stack_cache_path, "rb") as fh:
                ridge = pickle.load(fh)
            log.info("Stacking meta-learner loaded from cache for %s", self.coin)
            return ridge

        val_preds_list = []  # each entry: (n_val, n_horizons)

        if TORCH_AVAILABLE:
            Xv = torch.tensor(X_val_s, dtype=torch.float32)
            if self._lstm is not None:
                self._lstm.eval()
                with torch.no_grad():
                    p = self._y_scaler.inverse_transform(self._lstm(Xv).numpy())
                val_preds_list.append(p)
            if self._transformer is not None:
                self._transformer.eval()
                with torch.no_grad():
                    p = self._y_scaler.inverse_transform(self._transformer(Xv).numpy())
                val_preds_list.append(p)

        if self._prophet_model is not None and hasattr(self, "_df"):
            ph = _prophet_predict_horizons(self._df["close"].dropna())
            if not np.all(np.isnan(ph)):
                # Broadcast single forecast to all val samples
                val_preds_list.append(np.tile(ph, (len(X_val_s), 1)))

        if self._sarima_model is not None:
            sa = _sarima_predict_horizons(self._sarima_model)
            if not np.all(np.isnan(sa)):
                val_preds_list.append(np.tile(sa, (len(X_val_s), 1)))

        if len(val_preds_list) < 2:
            log.info("Not enough models for stacking — falling back to mean ensemble")
            return None

        # Build meta-feature matrix: shape (n_val, n_models * n_horizons)
        meta_X = np.hstack(val_preds_list)              # (n_val, n_models * n_horizons)
        meta_y = y_val_raw                              # (n_val, n_horizons)

        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_X, meta_y)
        log.info("Stacking Ridge fitted for %s (meta_X=%s)", self.coin, meta_X.shape)

        with open(stack_cache_path, "wb") as fh:
            pickle.dump(ridge, fh, protocol=5)
        return ridge

    # ------------------------------------------------------------------
    def validate(self) -> Dict:
        metrics: Dict = {}
        if not hasattr(self, "_y_val_raw") or len(self._y_val_raw) < 2:
            return metrics

        y_true = self._y_val_raw
        Xv = None
        if TORCH_AVAILABLE and hasattr(self, "_X_val_s"):
            Xv = torch.tensor(self._X_val_s, dtype=torch.float32)

        for i, label in enumerate(HORIZON_LABELS):
            if self._lstm is not None and Xv is not None:
                with torch.no_grad():
                    pred = self._y_scaler.inverse_transform(self._lstm(Xv).numpy())
                metrics.setdefault("lstm", {})[label] = compute_metrics(
                    y_true[:, i], pred[:, i], f"lstm/{label}"
                )
            if self._transformer is not None and Xv is not None:
                with torch.no_grad():
                    pred = self._y_scaler.inverse_transform(self._transformer(Xv).numpy())
                metrics.setdefault("transformer", {})[label] = compute_metrics(
                    y_true[:, i], pred[:, i], f"transformer/{label}"
                )
        return metrics

    # ------------------------------------------------------------------
    def forecast(
        self,
        latest_window: np.ndarray,
        current_price: float = 0.0,
        live_price: Optional[float] = None,
    ) -> ForecastResult:
        result = ForecastResult(coin=self.coin, current_price=live_price or current_price)

        if self._prophet_model is not None and hasattr(self, "_df"):
            result.prophet_preds = _prophet_predict_horizons(self._df["close"].dropna())

        if self._sarima_model is not None:
            result.sarima_preds = _sarima_predict_horizons(self._sarima_model)

        base_preds_for_stack = []

        if TORCH_AVAILABLE and hasattr(self, "_x_scaler"):
            n, w, f = latest_window.shape
            win_s = self._x_scaler.transform(latest_window.reshape(-1, f)).reshape(n, w, f)
            Xw = torch.tensor(win_s, dtype=torch.float32)

            if self._lstm is not None:
                # MC Dropout: get mean + std
                mean_t, std_t = self._lstm.mc_forward(Xw, n_samples=MC_SAMPLES)
                result.lstm_preds = self._y_scaler.inverse_transform(mean_t.detach().numpy())[0]
                # Scale std back to price space (approximate using y_scaler scale)
                y_scale = self._y_scaler.scale_  # shape: (n_horizons,)
                result.lstm_uncertainty = std_t.detach().numpy()[0] / (y_scale + 1e-8)
                base_preds_for_stack.append(result.lstm_preds)

            if self._transformer is not None:
                mean_t, std_t = self._transformer.mc_forward(Xw, n_samples=MC_SAMPLES)
                result.transformer_preds = self._y_scaler.inverse_transform(mean_t.detach().numpy())[0]
                result.transformer_uncertainty = std_t.detach().numpy()[0] / (self._y_scaler.scale_ + 1e-8)
                base_preds_for_stack.append(result.transformer_preds)

        if result.prophet_preds is not None and not np.all(np.isnan(result.prophet_preds)):
            base_preds_for_stack.append(result.prophet_preds)
        if result.sarima_preds is not None and not np.all(np.isnan(result.sarima_preds)):
            base_preds_for_stack.append(result.sarima_preds)

        # ---- Stacking ensemble ----
        if self._stacking_ridge is not None and len(base_preds_for_stack) >= 2:
            # For single sample forecast, meta_X shape: (1, n_models * n_horizons)
            # We need to pad to match training if some models missing; use available
            n_models_train = self._stacking_ridge.n_features_in_ // len(HORIZONS)
            current_n = len(base_preds_for_stack)
            if current_n == n_models_train:
                meta_X = np.hstack([p[np.newaxis, :] for p in base_preds_for_stack])
                result.ensemble_preds = self._stacking_ridge.predict(meta_X)[0]
            else:
                # Fallback to mean if model count mismatch
                result.ensemble_preds = np.nanmean(base_preds_for_stack, axis=0)
        elif len(base_preds_for_stack) > 0:
            result.ensemble_preds = np.nanmean(base_preds_for_stack, axis=0)

        # Ensemble confidence interval: propagate uncertainties
        uncertainties = []
        if result.lstm_uncertainty is not None:
            uncertainties.append(result.lstm_uncertainty)
        if result.transformer_uncertainty is not None:
            uncertainties.append(result.transformer_uncertainty)
        if uncertainties and result.ensemble_preds is not None:
            combined_std = np.mean(uncertainties, axis=0)
            result.ensemble_lower = result.ensemble_preds - 1.96 * combined_std
            result.ensemble_upper = result.ensemble_preds + 1.96 * combined_std

        result.metrics = self.validate()
        return result

    # ------------------------------------------------------------------
    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        n_test_steps: int = 60,
        step_size: int = 7,
    ) -> Dict:
        """
        Proper walk-forward backtest:
        - Train on data up to step t
        - Predict next horizons
        - Advance t by step_size days
        - Repeat n_test_steps // step_size times

        This simulates real deployment and gives honest performance metrics.
        Unlike the original static backtest which tested a pre-trained model.
        """
        if not TORCH_AVAILABLE or self._lstm is None:
            log.warning("Walk-forward backtest requires PyTorch LSTM.")
            return {}

        log.info("Running walk-forward backtest for %s (%d steps)…", self.coin, n_test_steps)
        n_total = len(X)
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
            X_te = X[t: t + 1]
            y_te = y[t: t + 1]

            n_v = max(1, int(len(X_tr) * 0.1))
            X_tr_s, y_tr_s, xsc, ysc = scale_features(X_tr[:-n_v], y_tr[:-n_v])
            n, w, f = X_te.shape
            X_te_s = xsc.transform(X_te.reshape(-1, f)).reshape(n, w, f)

            n_val_v, w_val, f_val = X_tr[-n_v:].shape
            X_val_wf_s = xsc.transform(X_tr[-n_v:].reshape(-1, f_val)).reshape(n_val_v, w_val, f_val)
            y_val_wf_s = ysc.transform(y_tr[-n_v:])

            # Quick retrain (fewer epochs for speed)
            tmp_lstm = LSTMModel(
                input_size=f,
                hidden_size=self._lstm_hparams.get("hidden_size", 128),
                num_layers=self._lstm_hparams.get("num_layers", 2),
                output_size=len(HORIZONS),
                dropout=self._lstm_hparams.get("dropout", 0.2),
            )
            tmp_lstm = _train_torch_model(
                tmp_lstm, X_tr_s, y_tr_s, X_val_wf_s, y_val_wf_s,
                epochs=20, batch_size=32, lr=1e-3, device=self.device, patience=5,
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
