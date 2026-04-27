/**
 * hooks/usePrediction.js
 *
 * Changes from original:
 *  - HTTP calls are now delegated to predictionService (checkHealth,
 *    fetchPrediction) instead of calling api directly. The hook owns
 *    state logic only.
 *  - Error handling now distinguishes ApiError types (network / timeout /
 *    server) to surface more useful messages to the user.
 *  - Constants (COINS, MODELS, ALL_HORIZONS, HORIZON_LABELS) are moved to
 *    a separate constants file so they can be imported anywhere without
 *    pulling in the hook's state logic.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { checkHealth, fetchPrediction } from "../services/predictionService";
import { ApiError } from "../services/api";

export { ALL_HORIZONS, HORIZON_LABELS, COINS, MODELS } from "../constants/prediction";

const POLL_INTERVAL_MS  = 3_000;
const MAX_POLL_ATTEMPTS = 40;

export function usePrediction() {
    const [selectedCoin, setSelectedCoin]     = useState("BTC");
    const [activeHorizons, setActiveHorizons] = useState(new Set(["1w","1m","3m","6m","1y"]));
    const [visibleModels, setVisibleModels]   = useState(new Set(["ensemble","prophet","sarima","lstm","transformer"]));
    const [showCI, setShowCI]                 = useState(true);
    const [loading, setLoading]               = useState(false);
    const [error, setError]                   = useState(null);
    const [data, setData]                     = useState(null);
    const [backendReady, setBackendReady]     = useState(null);
    const [trainedCoins, setTrainedCoins]     = useState([]);
    const [lastUpdated, setLastUpdated]       = useState(null);

    const pollRef   = useRef(null);
    const pollCount = useRef(0);

    // ── Health check ────────────────────────────────────────────────────────
    const doCheckHealth = useCallback(async () => {
        try {
            const { status, trained_coins } = await checkHealth();
            setTrainedCoins(trained_coins || []);
            if (status === "ready") {
                setBackendReady(true);
                return true;
            }
        } catch {
            // Backend not reachable yet — polling will retry
        }
        return false;
    }, []);

    // Poll until backend is ready
    useEffect(() => {
        let stopped = false;
        const startPolling = async () => {
            const ready = await doCheckHealth();
            if (ready || stopped) return;
            setBackendReady(false);
            pollRef.current = setInterval(async () => {
                pollCount.current += 1;
                if (pollCount.current >= MAX_POLL_ATTEMPTS) {
                    clearInterval(pollRef.current);
                    setError("Backend took too long to start. Please refresh the page.");
                    return;
                }
                const r = await doCheckHealth();
                if (r) clearInterval(pollRef.current);
            }, POLL_INTERVAL_MS);
        };
        startPolling();
        return () => {
            stopped = true;
            clearInterval(pollRef.current);
        };
    }, [doCheckHealth]);

    // ── Prediction fetch ────────────────────────────────────────────────────
    const runPrediction = useCallback(async (coin) => {
        setLoading(true);
        setError(null);
        try {
            const result = await fetchPrediction(coin);
            setData(result);
            setLastUpdated(new Date());
        } catch (err) {
            // Give users a more meaningful message based on error type
            if (err instanceof ApiError) {
                const msg = {
                    timeout: "The model is taking too long. Try again in a moment.",
                    network: "Cannot reach the prediction server. Check your connection.",
                    server:  err.message,
                }[err.type] ?? err.message;
                setError(msg);
            } else {
                setError(err.message || "Failed to fetch prediction.");
            }
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (backendReady) runPrediction(selectedCoin);
    }, [selectedCoin, backendReady, runPrediction]);

    // ── Derived forecast filtered by active horizons ─────────────────────────
    const filteredForecast = useMemo(() => {
        if (!data?.forecast) return null;
        const f = data.forecast;
        const indices = f.horizons
            .map((h, i) => ({ h, i }))
            .filter(({ h }) => activeHorizons.has(h));
        const pick = (arr) => (arr ? indices.map(({ i }) => arr[i] ?? null) : []);
        return {
            ...f,
            horizons:       indices.map(({ h }) => h),
            ensemble:       pick(f.ensemble),
            ensemble_lower: pick(f.ensemble_lower),
            ensemble_upper: pick(f.ensemble_upper),
            prophet:        pick(f.prophet),
            sarima:         pick(f.sarima),
            lstm:           pick(f.lstm),
            transformer:    pick(f.transformer),
        };
    }, [data, activeHorizons]);

    // ── Toggle helpers ───────────────────────────────────────────────────────
    const toggleHorizon = useCallback((h) => {
        setActiveHorizons((prev) => {
            const next = new Set(prev);
            if (next.has(h)) { if (next.size > 1) next.delete(h); }
            else next.add(h);
            return next;
        });
    }, []);

    const toggleModel = useCallback((key) => {
        if (key === "ensemble") return; // ensemble is always visible
        setVisibleModels((prev) => {
            const next = new Set(prev);
            if (next.has(key)) { if (next.size > 1) next.delete(key); }
            else next.add(key);
            return next;
        });
    }, []);

    const selectCoin = useCallback(
        (coin) => { if (!loading) setSelectedCoin(coin); },
        [loading]
    );

    const refresh = useCallback(() => {
        if (backendReady && !loading) runPrediction(selectedCoin);
    }, [backendReady, loading, selectedCoin, runPrediction]);

    return {
        selectedCoin, selectCoin, trainedCoins,
        activeHorizons, toggleHorizon,
        visibleModels, toggleModel,
        showCI, setShowCI,
        loading, error, data, filteredForecast, backendReady, lastUpdated,
        refresh,
    };
}
