/**
 * constants/prediction.js
 *
 * Static configuration for the prediction feature.
 *
 * Previously these lived inside usePrediction.js, which meant any component
 * that needed COINS or MODELS had to import the hook — pulling in all of its
 * state machinery even if it only needed a label map.
 *
 * Extracting them here:
 *  - Makes imports explicit and lightweight
 *  - Allows CoinSelector, ForecastChart, etc. to import constants directly
 *  - usePrediction re-exports them for backward compatibility
 */

export const ALL_HORIZONS = ["1w", "1m", "3m", "6m", "1y"];

export const HORIZON_LABELS = {
    "1w": "1 Week",
    "1m": "1 Month",
    "3m": "3 Months",
    "6m": "6 Months",
    "1y": "1 Year",
};

export const COINS = [
    { symbol: "BTC", name: "Bitcoin",      icon: "₿",  color: "#f7931a" },
    { symbol: "ETH", name: "Ethereum",     icon: "Ξ",  color: "#627eea" },
    { symbol: "BNB", name: "Binance Coin", icon: "⬡",  color: "#f3ba2f" },
    { symbol: "SOL", name: "Solana",       icon: "◎",  color: "#9945ff" },
    { symbol: "XRP", name: "XRP",          icon: "✕",  color: "#346aa9" },
];

export const MODELS = [
    { key: "ensemble",    label: "Ensemble",    color: "#4f7ef7", primary: true  },
    { key: "prophet",     label: "Prophet",     color: "#10b981", primary: false },
    { key: "sarima",      label: "SARIMA",      color: "#f59e0b", primary: false },
    { key: "lstm",        label: "LSTM",        color: "#a78bfa", primary: false },
    { key: "transformer", label: "Transformer", color: "#f472b6", primary: false },
];
