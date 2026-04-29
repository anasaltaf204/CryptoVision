# 🔮 CryptoVision AI

> AI-powered cryptocurrency price forecasting with an ensemble of machine learning models — Prophet, SARIMA, LSTM, and Transformer — served through a FastAPI backend and a React frontend.

---

## 📸 Overview

CryptoPredict AI lets users view live crypto market data, explore technical indicators, and get multi-horizon price forecasts with 95% confidence intervals for **BTC, ETH, BNB, SOL, and XRP**.

| Feature | Details |
|---|---|
| 📊 Dashboard | Live prices, market cap, trending coins, charts |
| 🔮 Prediction | AI ensemble forecast — 1W / 1M / 3M / 6M / 1Y |
| 📐 Model Accuracy | Directional accuracy & MAPE per model |
| 📡 Live Market | Real-time Binance ticker data |
| 🔒 Auth | Firebase email/password authentication |
| 📱 Responsive | Mobile-friendly layout |

---

## 🧠 AI Models

The prediction engine uses a **stacking ensemble** (Ridge meta-learner) over four base models:

| Model | Type | Notes |
|---|---|---|
| **Prophet** | Statistical | Facebook's time-series model, handles seasonality |
| **SARIMA** | Statistical | Auto-tuned via `pmdarima` |
| **LSTM** | Deep Learning | PyTorch, MC Dropout uncertainty estimation |
| **Transformer** | Deep Learning | PyTorch, MC Dropout uncertainty estimation |
| **Ensemble** | Meta-learner | Ridge regression stacking all four models |

Hyperparameters are tuned automatically using **Optuna** (Bayesian search).
Models are **cached after first training** — subsequent startups are instant.

---

## 🗂️ Project Structure

```
crypto-price-prediction/
│
├── fastapi_backend/               # Python AI backend
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/
│   │   ├── BTC.csv                # Historical OHLCV data
│   │   ├── ETH.csv
│   │   ├── BNB.csv
│   │   ├── SOL.csv
│   │   ├── XRP.csv
│   │   ├── data_pipeline.py       # Data loading & feature engineering
│   │   └── model_training.py      # Model training, forecasting & backtesting
│   └── app/
│       ├── core/
│       │   ├── config.py          # Settings from environment variables
│       │   └── trainer_state.py   # Singleton model registry (loaded on startup)
│       ├── routers/
│       │   ├── health.py          # GET /api/health
│       │   └── prediction.py      # POST /api/predict
│       ├── services/
│       │   └── prediction_service.py  # Forecast business logic
│       └── models/
│           └── schemas.py         # Pydantic request/response models
│
└── frontend_updated/              # React frontend
    ├── index.html
    ├── vite.config.js
    ├── package.json
    ├── .env.example
    └── src/
        ├── App.jsx                # Router & layout
        ├── pages/
        │   ├── Dashboard.jsx      # Market overview page
        │   ├── Prediction.jsx     # AI prediction page
        │   ├── CoinDetail.jsx     # Individual coin page
        │   └── LoginSignUp.jsx    # Auth page
        ├── hooks/
        │   ├── usePrediction.js   # All prediction state & API logic
        │   └── useDashboard.js    # Dashboard data fetching
        ├── components/
        │   ├── prediction/        # Prediction page components
        │   │   ├── PredictionHeader.jsx
        │   │   ├── CoinSelector.jsx
        │   │   ├── MarketSummary.jsx
        │   │   ├── PredictionControls.jsx
        │   │   ├── ForecastChart.jsx
        │   │   ├── ForecastTable.jsx
        │   │   ├── ModelAccuracyCards.jsx
        │   │   ├── IndicatorsGrid.jsx
        │   │   └── WarmUpBanner.jsx
        │   └── dashboard/         # Dashboard page components
        │       ├── StatCard.jsx
        │       ├── ChartPanel.jsx
        │       ├── MarketBanner.jsx
        │       └── TrendingBar.jsx
        ├── services/
        │   ├── api.js             # Axios instance pointing to FastAPI
        │   └── coingecko.js       # CoinGecko API calls
        └── Styles/
            ├── Dashboard.css
            ├── Prediction.css
            └── ...
```

---

## ⚙️ Tech Stack

**Backend**
- Python 3.10+
- FastAPI + Uvicorn
- PyTorch (LSTM & Transformer)
- Prophet, pmdarima (SARIMA)
- Optuna (hyperparameter tuning)
- Pandas, NumPy, scikit-learn
- Binance REST API (live prices)
- CoinGecko API (market data)

**Frontend**
- React 19 + Vite
- React Router v7
- Chart.js + react-chartjs-2
- Axios
- Firebase (Authentication + Firestore)
- Bootstrap 5
- FontAwesome

---

## 🚀 Running Locally

You need **two terminals** — one for the backend, one for the frontend.

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- npm

---

### Terminal 1 — Backend

**1. Go to the backend folder**
```bash
cd fastapi_backend
```

**2. Create and activate a virtual environment**
```bash
# Create
python -m venv .venv

# Activate — Mac/Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
> ⏳ This takes a few minutes — it installs PyTorch, Prophet, Optuna, etc.

**4. Set up your environment file**
```bash
cp .env.example .env
```
The `.env` already has working API keys. No changes needed for local development.

**5. Start the server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**What to expect on first run:**
```
🚀 Starting up — training models for all coins…
Training BTC…  ✅
Training ETH…  ✅
...
✅ Startup training complete.
Uvicorn running on http://0.0.0.0:8000
```
> ⏱️ **First run:** 2–5 minutes (trains all 5 models)
> ⚡ **After that:** near-instant (models are cached in `.model_cache/`)

**Verify it's working:**
- API docs → http://localhost:8000/docs
- Health check → http://localhost:8000/api/health

---

### Terminal 2 — Frontend

**1. Go to the frontend folder**
```bash
cd frontend_updated
```

**2. Install dependencies**
```bash
npm install
```

**3. Check your `.env` file contains:**
```
VITE_API_URL=http://127.0.0.1:8000
```

**4. Start the dev server**
```bash
npm run dev
```

Open **http://localhost:5173** in your browser.

---

### Restarting After Shutdown

Every time you want to run the project again:

```bash
# Terminal 1 — Backend
cd fastapi_backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend_updated
npm run dev
```

---

## 🔑 Environment Variables

### Backend (`fastapi_backend/.env`)

| Variable | Description | Default |
|---|---|---|
| `COINGECKO_API_KEY` | CoinGecko Demo API key | Provided |
| `CRYPTOPANIC_TOKEN` | CryptoPanic token (optional) | — |
| `NEWSAPI_KEY` | NewsAPI key (optional) | Provided |
| `DATA_DIR` | Path to folder with CSV files | `./data` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `localhost:5173` |
| `EPOCHS` | Training epochs for deep models | `50` |
| `OPTUNA_TRIALS` | Bayesian HP search trials | `15` |
| `ENABLE_PROPHET` | Enable Prophet model | `true` |
| `ENABLE_SARIMA` | Enable SARIMA model | `true` |
| `ENABLE_LSTM` | Enable LSTM model | `true` |
| `ENABLE_TRANSFORMER` | Enable Transformer model | `true` |

**Speed up training during development:**
```env
EPOCHS=10
OPTUNA_TRIALS=3
ENABLE_TRANSFORMER=false
```

### Frontend (`frontend_updated/.env`)

| Variable | Description |
|---|---|
| `VITE_API_URL` | FastAPI backend URL |
| `VITE_FIREBASE_API_KEY` | Firebase project API key |
| `VITE_FIREBASE_AUTH_DOMAIN` | Firebase auth domain |
| `VITE_FIREBASE_PROJECT_ID` | Firebase project ID |
| `VITE_FIREBASE_STORAGE_BUCKET` | Firebase storage bucket |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Firebase sender ID |
| `VITE_FIREBASE_APP_ID` | Firebase app ID |

---

## 📡 API Reference

### `GET /api/health`
Returns backend status and list of trained coins.

```json
{
  "status": "ready",
  "trained_coins": ["BTC", "ETH", "BNB", "SOL", "XRP"],
  "supported_coins": ["BTC", "ETH", "BNB", "SOL", "XRP"]
}
```

### `POST /api/predict`

**Request body:**
```json
{
  "coin": "BTC",
  "run_backtest": false
}
```

**Response:**
```json
{
  "coin": "BTC",
  "generated_at": "2025-01-01T12:00:00+00:00",
  "live_market": {
    "price": 95000.0,
    "price_change_pct_24h": 2.3,
    "volume_24h": 28000000000.0,
    "bid": 94995.0,
    "ask": 95005.0,
    "spread": 10.0
  },
  "indicators": {
    "rsi_14": 54.32,
    "macd": 120.45,
    "sma_20": 92000.0
  },
  "forecast": {
    "horizons": ["1w", "1m", "3m", "6m", "1y"],
    "current_price": 95000.0,
    "ensemble": [96500.0, 102000.0, 115000.0, 130000.0, 160000.0],
    "ensemble_lower": [92000.0, 94000.0, 100000.0, 110000.0, 130000.0],
    "ensemble_upper": [101000.0, 110000.0, 130000.0, 150000.0, 190000.0],
    "prophet": [...],
    "sarima": [...],
    "lstm": [...],
    "transformer": [...]
  },
  "validation_metrics": { ... }
}
```

---

## 🛠️ Troubleshooting

**Models are warming up / training forever**
→ Check Terminal 1 for logs. Wait for `✅ Startup training complete.`

**"Failed to fetch prediction" on Prediction page**
→ Make sure the backend is running in Terminal 1 and `VITE_API_URL=http://127.0.0.1:8000` is in your frontend `.env`.

**Prophet install fails**
```bash
pip install cmdstanpy
python -m cmdstanpy.install_cmdstan
pip install prophet
```

**PyTorch install fails on Windows**
→ Install manually from https://pytorch.org/get-started/locally (select CPU), then re-run `pip install -r requirements.txt`.

**Frontend blank page or import error**
```bash
rm -rf node_modules
npm install
npm run dev
```

---

## 📦 Deployment

### Backend → Railway
1. Push `fastapi_backend/` to a GitHub repo
2. Create a new project at https://railway.app → Deploy from GitHub
3. Add environment variables (including `ALLOWED_ORIGINS=https://your-app.vercel.app`)
4. Railway auto-detects Python and deploys

### Frontend → Vercel
1. Push `frontend_updated/` to a GitHub repo
2. Import at https://vercel.com → New Project
3. Set `VITE_API_URL=https://your-railway-url.up.railway.app` in environment variables
4. Deploy

---

## 📄 License

This project is for educational purposes.

---

> Built with ❤️ using FastAPI, React, PyTorch, and Prophet.
