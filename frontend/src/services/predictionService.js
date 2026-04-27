/**
 * services/predictionService.js
 *
 * All prediction-backend API calls in one place.
 * Components and hooks import from here — never call api directly.
 *
 * Why a dedicated service module?
 *  - Swapping the backend URL or auth scheme requires one file change.
 *  - Mocking in tests is trivial (import the service, mock its exports).
 *  - Keeps hooks focused on state logic, not HTTP concerns.
 */

import api from "./api";

/**
 * Checks backend health and returns { status, trained_coins }.
 * @returns {Promise<{ status: string, trained_coins: string[] }>}
 */
export async function checkHealth() {
    const res = await api.get("/api/health");
    return res.data;
}

/**
 * Fetches a price forecast for the given coin symbol.
 * @param {string} coin  e.g. "BTC", "ETH"
 * @returns {Promise<object>}  Full prediction payload from the backend
 */
export async function fetchPrediction(coin) {
    const res = await api.post("/api/predict", { coin });
    return res.data;
}
