/**
 * services/coingecko.js
 *
 * CoinGecko API service layer.
 *
 * Changes from original:
 *  - Replaced bare axios.get() calls with a dedicated coingeckoClient
 *    Axios instance that has its own baseURL, timeout, and interceptors.
 *    This isolates CoinGecko concerns from the backend api instance.
 *  - Response interceptor on coingeckoClient normalises rate-limit errors
 *    (HTTP 429) into a clear message instead of a raw AxiosError.
 *  - Cache helpers and fetchWithCache are unchanged — they were already solid.
 */

import axios from "axios";

// ── CoinGecko Axios instance ──────────────────────────────────────────────────
const coingeckoClient = axios.create({
    baseURL: "https://api.coingecko.com/api/v3",
    timeout: 10_000,
    headers: { Accept: "application/json" },
});

coingeckoClient.interceptors.response.use(
    (res) => res,
    (error) => {
        if (error.response?.status === 429) {
            return Promise.reject(
                new Error("CoinGecko rate limit reached. Please wait a moment.")
            );
        }
        const msg =
            error.response?.data?.error ||
            error.message ||
            "CoinGecko request failed";
        return Promise.reject(new Error(msg));
    }
);

// ── Cache helpers (localStorage with TTL) ────────────────────────────────────
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function cacheSet(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify({ ts: Date.now(), data }));
    } catch {
        // localStorage quota exceeded — fail silently
    }
}

function cacheGet(key) {
    try {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        const { ts, data } = JSON.parse(raw);
        return { data, stale: Date.now() - ts > CACHE_TTL };
    } catch {
        return null;
    }
}

// ── Cache-then-network fetcher ────────────────────────────────────────────────
// Returns cached data immediately so the UI never blocks on a spinner.
// Callers receive { data, fromCache, stale } to show staleness indicators.
async function fetchWithCache(cacheKey, fetcher) {
    const cached = cacheGet(cacheKey);

    if (cached && !cached.stale) {
        return { data: cached.data, fromCache: true, stale: false };
    }

    try {
        const fresh = await fetcher();
        cacheSet(cacheKey, fresh);
        return { data: fresh, fromCache: false, stale: false };
    } catch (err) {
        // Network failed — serve stale cache rather than crash the UI
        if (cached) {
            return { data: cached.data, fromCache: true, stale: true };
        }
        throw err;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Fetches top-N coins with sparkline data. */
export async function fetchCoins(perPage = 20) {
    return fetchWithCache(`cg_coins_${perPage}`, async () => {
        const res = await coingeckoClient.get("/coins/markets", {
            params: {
                vs_currency: "usd",
                order: "market_cap_desc",
                per_page: perPage,
                page: 1,
                sparkline: true,
                price_change_percentage: "1h,24h,7d",
            },
        });
        return res.data;
    });
}

/** Fetches global market stats (market cap, dominance, volume, etc.) */
export async function fetchGlobalStats() {
    return fetchWithCache("cg_global", async () => {
        const res = await coingeckoClient.get("/global");
        return res.data.data;
    });
}

/** Fetches trending coins (hot searches on CoinGecko). */
export async function fetchTrending() {
    return fetchWithCache("cg_trending", async () => {
        const res = await coingeckoClient.get("/search/trending");
        return res.data.coins.slice(0, 5).map((c) => c.item);
    });
}

/** Fetches full coin detail — price, market data, description, links, community. */
export async function fetchCoinDetail(coinId) {
    return fetchWithCache(`cg_coin_${coinId}`, async () => {
        const res = await coingeckoClient.get(`/coins/${coinId}`, {
            params: {
                localization: false,
                tickers: false,
                market_data: true,
                community_data: true,
                developer_data: false,
                sparkline: true,
            },
        });
        return res.data;
    });
}

/** Fetches OHLC price chart data. days: 1 | 7 | 14 | 30 | 90 | 180 | 365 */
export async function fetchCoinOHLC(coinId, days = 7) {
    return fetchWithCache(`cg_ohlc_${coinId}_${days}`, async () => {
        const res = await coingeckoClient.get(`/coins/${coinId}/ohlc`, {
            params: { vs_currency: "usd", days },
        });
        return res.data;
    });
}

/** Fetches historical market chart (prices + volumes) for a coin. */
export async function fetchCoinChart(coinId, days = 7) {
    return fetchWithCache(`cg_chart_${coinId}_${days}`, async () => {
        const res = await coingeckoClient.get(`/coins/${coinId}/market_chart`, {
            params: {
                vs_currency: "usd",
                days,
                interval: days <= 1 ? "minutely" : "daily",
            },
        });
        return res.data;
    });
}
