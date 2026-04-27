/**
 * hooks/useDashboard.js
 *
 * Thin hook over the Redux dashboard slice.
 *
 * Changes from original:
 *  - Data fetching is now done via dispatched thunks (loadCoins, loadGlobalStats,
 *    loadTrending) instead of calling service functions directly.
 *  - Component state (coins, globalStats, trending, status) is read from the
 *    Redux store via useSelector, so any component can subscribe without
 *    triggering duplicate fetches.
 *  - The auto-refresh interval and the manual refresh() function remain here
 *    because they are side-effects, not store state.
 *  - isRefreshing is tracked locally — it's transient UI feedback, not shared
 *    data that other components care about.
 */

import { useEffect, useCallback, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
    loadCoins,
    loadGlobalStats,
    loadTrending,
    setLastUpdated,
    selectCoins,
    selectGlobalStats,
    selectTrending,
    selectDashboardMeta,
} from "../store/slices/dashboardSlice";

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 min

export function useDashboard() {
    const dispatch     = useDispatch();
    const coins        = useSelector(selectCoins);
    const globalStats  = useSelector(selectGlobalStats);
    const trending     = useSelector(selectTrending);
    const meta         = useSelector(selectDashboardMeta);

    const [isRefreshing, setIsRefreshing] = useState(false);
    const intervalRef = useRef(null);

    const loadAll = useCallback(
        async (isManualRefresh = false) => {
            if (isManualRefresh) setIsRefreshing(true);

            // Fire all three thunks in parallel — each updates its own slice
            await Promise.allSettled([
                dispatch(loadCoins(20)),
                dispatch(loadGlobalStats()),
                dispatch(loadTrending()),
            ]);

            dispatch(setLastUpdated());
            if (isManualRefresh) setIsRefreshing(false);
        },
        [dispatch]
    );

    // Initial load + auto-refresh
    useEffect(() => {
        loadAll();
        intervalRef.current = setInterval(() => loadAll(), REFRESH_INTERVAL);
        return () => clearInterval(intervalRef.current);
    }, [loadAll]);

    const refresh = useCallback(() => loadAll(true), [loadAll]);

    // Convenience flag: true only during the very first load with no cached data
    const isInitialLoad =
        meta.coinsStatus === "loading" && coins.length === 0;

    // Unified status object matching the shape the original hook returned,
    // so Dashboard.jsx requires zero changes to its destructuring.
    const status = {
        coins:    meta.coinsStatus,
        global:   meta.globalStatus,
        trending: meta.trendingStatus,
    };

    return {
        coins,
        globalStats,
        trending,
        status,
        isStale:      meta.isStale,
        isRefreshing,
        lastUpdated:  meta.lastUpdated ? new Date(meta.lastUpdated) : null,
        isInitialLoad,
        refresh,
    };
}
