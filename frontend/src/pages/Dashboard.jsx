

import React from "react";
import {
    Chart as ChartJS,
    CategoryScale, LinearScale, PointElement, LineElement,
    BarElement, ArcElement, Tooltip, Legend, Filler,
} from "chart.js";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowsRotate } from "@fortawesome/free-solid-svg-icons";
import { useDashboard } from "../hooks/useDashboard";
import MarketBanner   from "../components/dashboard/MarketBanner";
import TrendingBar    from "../components/dashboard/TrendingBar";
import StatCard       from "../components/dashboard/StatCard";
import ChartPanel     from "../components/dashboard/ChartPanel";
import CryptoTable    from "../components/CryptoTable";
import "../Styles/Dashboard.css";

ChartJS.register(
    CategoryScale, LinearScale, PointElement, LineElement,
    BarElement, ArcElement, Tooltip, Legend, Filler
);

function fmtT(n) { return n ? "$" + (n / 1e12).toFixed(2) + "T" : "—"; }
function fmtB(n) { return n ? "$" + (n / 1e9).toFixed(1) + "B" : "—"; }

export default function Dashboard() {
    const {
        coins, globalStats, trending,
        status, isStale, isRefreshing, lastUpdated,
        isInitialLoad, refresh,
    } = useDashboard();


    // ── Derive stat card data ─────────────────────────────────────────────────
    const mcap        = globalStats?.total_market_cap?.usd || 0;
    const vol         = globalStats?.total_volume?.usd || 0;
    const btcDom      = globalStats?.market_cap_percentage?.btc || 0;
    const activeCoin  = globalStats?.active_cryptocurrencies || 0;
    const mcapChange  = globalStats?.market_cap_change_percentage_24h_usd || 0;

    const mcapPrices = (() => {
        const top10 = coins.slice(0, 10);
        if (!top10.length || !top10[0]?.sparkline_in_7d?.price?.length) return [];
        const len = top10[0].sparkline_in_7d.price.length;
        return Array.from({ length: len }, (_, i) =>
            top10.reduce((sum, c) => sum + (c.sparkline_in_7d?.price?.[i] ?? 0), 0)
        );
    })();


    const volPrices = (() => {
        const top3 = coins.slice(0, 3);
        if (!top3.length || !top3[0]?.sparkline_in_7d?.price?.length) return [];
        const len = top3[0].sparkline_in_7d.price.length;
        return Array.from({ length: len }, (_, i) =>
            top3.reduce((sum, c) => sum + (c.sparkline_in_7d?.price?.[i] ?? 0), 0)
        );
    })();


    const btcCoin     = coins.find(c => c.id === "bitcoin");
    const btcPrices   = btcCoin?.sparkline_in_7d?.price || [];

    const top20Changes = coins.map(c => c.price_change_percentage_24h ?? 0);

    return (
        <div className="dashboard">

            {/* ── Top stats banner (CoinGecko-style header strip) ── */}
            <MarketBanner globalStats={globalStats} isStale={isStale} />

            {/* ── Page header + refresh ── */}
            <div className="dashboard__header container-fluid px-lg-5">
                <div className="dashboard__title-block">
                    <h1 className="dashboard__title">Cryptocurrency Prices by Market Cap</h1>
                    <p className="dashboard__subtitle">
                        The global crypto market cap is{" "}
                        <strong>{fmtT(mcap)}</strong>
                        {mcapChange !== 0 && (
                            <span className={mcapChange >= 0 ? "text-success" : "text-danger"}>
                                {" "}{mcapChange >= 0 ? "▲" : "▼"} {Math.abs(mcapChange).toFixed(1)}%
                            </span>
                        )}{" "}
                        in the last 24 hours.
                    </p>
                </div>
                <div className="dashboard__actions">
                    {lastUpdated && (
                        <span className="dashboard__updated">
                            Updated {lastUpdated.toLocaleTimeString()}
                        </span>
                    )}
                    <button
                        className={`dashboard__refresh-btn ${isRefreshing ? "spinning" : ""}`}
                        onClick={refresh}
                        disabled={isRefreshing}
                        title="Refresh all data"
                    >
                        <FontAwesomeIcon icon={faArrowsRotate} />
                        {isRefreshing ? "Refreshing…" : "Refresh"}
                    </button>
                </div>
            </div>

            {/* ── Trending coins strip ── */}
            <div className="container-fluid px-lg-5">
                <TrendingBar
                    trending={trending}
                    loading={status.trending === "loading" && trending.length === 0}
                />
            </div>

            {/* ── Stat cards (no duplication: only one source of truth each) ── */}
            <div className="container-fluid px-lg-5">
                <div className="stat-cards-grid">
                    <StatCard
                        
                        label="Total Market Cap"
                        value={fmtT(mcap)}
                        change={mcapChange}
                        chartPrices={mcapPrices}
                        chartColor="#667eea"
                        loading={status.global === "loading" && !globalStats}
                    />
                    <StatCard
                        
                        label="24h Trading Volume"
                        value={fmtB(vol)}
                        subValue="USD"
                        chartPrices={volPrices}
                        chartColor="#10b981"
                        loading={status.global === "loading" && !globalStats}
                    />
                    <StatCard
                        
                        label="BTC Dominance"
                        value={btcDom.toFixed(2) + "%"}
                        subValue={btcDom > 50 ? "Market leader" : "Contested market"}
                        chartPrices={btcPrices}
                        chartColor="#f7931a"
                        formatTooltip={(v) => "$" + v.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                        loading={status.global === "loading" && !globalStats}
                    />
                    <StatCard
                        
                        label="Active Cryptocurrencies"
                        value={activeCoin.toLocaleString()}
                        subValue="Top-20 24h performance"
                        chartPrices={top20Changes}
                        chartColor="#8b5cf6"
                        formatTooltip={(v) => (v >= 0 ? "+" : "") + v.toFixed(2) + "%"}
                        loading={status.global === "loading" && !globalStats}
                    />
                </div>
            </div>

            {/* ── Charts: each sub-chart is isolated, only re-renders on its data ── */}
            <div className="container-fluid px-lg-5">
                <ChartPanel
                    coins={coins}
                    globalStats={globalStats}
                    coinsLoading={status.coins === "loading" && coins.length === 0}
                    statsLoading={status.global === "loading" && !globalStats}
                />
            </div>

            {/* ── Coin table with skeleton loading and sortable columns ── */}
            <div className="container-fluid px-lg-5">
                <div className="dashboard__table-header">
                    <h2 className="dashboard__table-title">
                        💎 Top {coins.length || 20} Cryptocurrencies by Market Cap
                    </h2>
                    {isStale && (
                        <span className="dashboard__stale-notice">
                            ⚡ Showing cached data — fetching fresh data…
                        </span>
                    )}
                </div>
                <CryptoTable
                    coins={coins}
                    loading={status.coins === "loading"}
                />
            </div>

            {/* ── Coin detail modal ── */}

        </div>
    );
}
