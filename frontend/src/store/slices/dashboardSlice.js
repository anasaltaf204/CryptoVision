/**
 * store/slices/dashboardSlice.js
 *
 * Redux slice for global dashboard data (coins, global stats, trending).
 *
 * Why Redux here (and not just local hook state)?
 *  - The same coin list is read by Dashboard AND Navbar (search feature).
 *  - GlobalStats is consumed by both MarketBanner and StatCards — if we kept
 *    it in a hook, both components would trigger duplicate API calls.
 *  - With Redux the data is fetched once and shared; any component can
 *    subscribe without causing new network requests.
 *
 * Why NOT Redux for prediction state?
 *  - Prediction data is only ever used by the Prediction page — there's no
 *    cross-route sharing benefit, so a custom hook stays cleaner.
 */

import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import {
    fetchCoins,
    fetchGlobalStats,
    fetchTrending,
} from "../../services/coingecko";

// ── Async thunks ──────────────────────────────────────────────────────────────

export const loadCoins = createAsyncThunk(
    "dashboard/loadCoins",
    async (perPage = 20, { rejectWithValue }) => {
        try {
            return await fetchCoins(perPage);
        } catch (err) {
            return rejectWithValue(err.message);
        }
    }
);

export const loadGlobalStats = createAsyncThunk(
    "dashboard/loadGlobalStats",
    async (_, { rejectWithValue }) => {
        try {
            return await fetchGlobalStats();
        } catch (err) {
            return rejectWithValue(err.message);
        }
    }
);

export const loadTrending = createAsyncThunk(
    "dashboard/loadTrending",
    async (_, { rejectWithValue }) => {
        try {
            return await fetchTrending();
        } catch (err) {
            return rejectWithValue(err.message);
        }
    }
);

// ── Slice ─────────────────────────────────────────────────────────────────────

const dashboardSlice = createSlice({
    name: "dashboard",
    initialState: {
        coins:       [],
        globalStats: null,
        trending:    [],

        // Per-slice status: "idle" | "loading" | "ready" | "error"
        coinsStatus:   "idle",
        globalStatus:  "idle",
        trendingStatus:"idle",

        coinsError:   null,
        globalError:  null,
        trendingError:null,

        isStale:      false,
        lastUpdated:  null,
    },
    reducers: {
        setStale(state, action) {
            state.isStale = action.payload;
        },
        setLastUpdated(state) {
            state.lastUpdated = new Date().toISOString();
        },
    },
    extraReducers: (builder) => {
        // ── Coins ──────────────────────────────────────────────────────────
        builder
            .addCase(loadCoins.pending, (state) => {
                if (state.coinsStatus !== "ready") state.coinsStatus = "loading";
                state.coinsError = null;
            })
            .addCase(loadCoins.fulfilled, (state, { payload }) => {
                state.coins = payload.data;
                state.coinsStatus = "ready";
                if (payload.stale) state.isStale = true;
            })
            .addCase(loadCoins.rejected, (state, { payload }) => {
                state.coinsStatus = "error";
                state.coinsError = payload;
            });

        // ── Global stats ───────────────────────────────────────────────────
        builder
            .addCase(loadGlobalStats.pending, (state) => {
                if (state.globalStatus !== "ready") state.globalStatus = "loading";
                state.globalError = null;
            })
            .addCase(loadGlobalStats.fulfilled, (state, { payload }) => {
                state.globalStats = payload.data;
                state.globalStatus = "ready";
                if (payload.stale) state.isStale = true;
            })
            .addCase(loadGlobalStats.rejected, (state, { payload }) => {
                state.globalStatus = "error";
                state.globalError = payload;
            });

        // ── Trending ───────────────────────────────────────────────────────
        builder
            .addCase(loadTrending.pending, (state) => {
                if (state.trendingStatus !== "ready") state.trendingStatus = "loading";
                state.trendingError = null;
            })
            .addCase(loadTrending.fulfilled, (state, { payload }) => {
                state.trending = payload.data;
                state.trendingStatus = "ready";
                if (payload.stale) state.isStale = true;
            })
            .addCase(loadTrending.rejected, (state, { payload }) => {
                state.trendingStatus = "error";
                state.trendingError = payload;
            });
    },
});

export const { setStale, setLastUpdated } = dashboardSlice.actions;
export default dashboardSlice.reducer;

// ── Selectors ─────────────────────────────────────────────────────────────────
export const selectCoins        = (s) => s.dashboard.coins;
export const selectGlobalStats  = (s) => s.dashboard.globalStats;
export const selectTrending     = (s) => s.dashboard.trending;
export const selectDashboardMeta = (s) => ({
    coinsStatus:    s.dashboard.coinsStatus,
    globalStatus:   s.dashboard.globalStatus,
    trendingStatus: s.dashboard.trendingStatus,
    coinsError:     s.dashboard.coinsError,
    globalError:    s.dashboard.globalError,
    trendingError:  s.dashboard.trendingError,
    isStale:        s.dashboard.isStale,
    lastUpdated:    s.dashboard.lastUpdated,
});
