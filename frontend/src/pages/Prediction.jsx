/**
 * Prediction.jsx — thin orchestrator page
 * All logic lives in usePrediction hook.
 * All sections are isolated, reusable components.
 */
import React from "react";
import {
    Chart as ChartJS, CategoryScale, LinearScale,
    PointElement, LineElement, Tooltip, Legend, Filler,
} from "chart.js";

import { usePrediction } from "../hooks/usePrediction";
import PredictionHeader    from "../components/prediction/PredictionHeader";
import CoinSelector        from "../components/prediction/CoinSelector";
import MarketSummary       from "../components/prediction/MarketSummary";
import PredictionControls  from "../components/prediction/PredictionControls";
import ForecastChart       from "../components/prediction/ForecastChart";
import ForecastTable       from "../components/prediction/ForecastTable";
import ModelAccuracyCards  from "../components/prediction/ModelAccuracyCards";
import IndicatorsGrid      from "../components/prediction/IndicatorsGrid";
import WarmUpBanner        from "../components/prediction/WarmUpBanner";
import "../Styles/Prediction.css";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

function SkeletonCard({ height = 260 }) {
    return (
        <div className="pred-card">
            <div className="pred-skeleton-bar" style={{ width: "40%", height: 18, marginBottom: 20 }} />
            <div className="pred-skeleton-bar" style={{ height }} />
        </div>
    );
}

export default function Prediction() {
    const {
        selectedCoin, selectCoin, trainedCoins,
        activeHorizons, toggleHorizon,
        visibleModels, toggleModel,
        showCI, setShowCI,
        loading, error, data, filteredForecast, backendReady, lastUpdated,
        refresh,
    } = usePrediction();

    return (
        <div className="pred-page">
            <div className="container-fluid px-lg-5">

                {/* ── Page header ── */}
                <PredictionHeader
                    selectedCoin={selectedCoin}
                    lastUpdated={lastUpdated}
                    loading={loading}
                    backendReady={backendReady}
                    onRefresh={refresh}
                />

                {/* ── Error ── */}
                {error && (
                    <div className="pred-error" role="alert">
                        <span className="pred-error-icon">⚠️</span>
                        {error}
                    </div>
                )}

                {/* ── Coin selector ── */}
                <CoinSelector
                    selected={selectedCoin}
                    onSelect={selectCoin}
                    disabled={loading || !backendReady}
                    trainedCoins={trainedCoins}
                />

                {/* ── Backend warming up ── */}
                {backendReady === false && !error && (
                    <WarmUpBanner trainedCoins={trainedCoins} />
                )}

                {/* ── Live market bar ── */}
                {(data?.live_market || loading) && (
                    <MarketSummary market={data?.live_market} coin={selectedCoin} />
                )}

                {/* ── Controls ── */}
                {backendReady && (
                    <PredictionControls
                        activeHorizons={activeHorizons}
                        toggleHorizon={toggleHorizon}
                        visibleModels={visibleModels}
                        toggleModel={toggleModel}
                        showCI={showCI}
                        setShowCI={setShowCI}
                    />
                )}

                {/* ── Main content ── */}
                {backendReady && (
                    <div className="pred-main-grid">
                        {/* Left column — chart + table */}
                        <div className="pred-col-main">
                            {loading ? (
                                <>
                                    <SkeletonCard height={320} />
                                    <SkeletonCard height={260} />
                                </>
                            ) : (
                                <>
                                    <ForecastChart
                                        forecast={filteredForecast}
                                        visibleModels={visibleModels}
                                        showCI={showCI}
                                    />
                                    <ForecastTable
                                        forecast={filteredForecast}
                                        visibleModels={visibleModels}
                                    />
                                </>
                            )}
                        </div>

                        {/* Right column — accuracy + indicators */}
                        <div className="pred-col-side">
                            {loading ? (
                                <>
                                    <SkeletonCard height={240} />
                                    <SkeletonCard height={300} />
                                </>
                            ) : (
                                <>
                                    {/* <ModelAccuracyCards
                                        validationMetrics={data?.validation_metrics}
                                    /> */}
                                    <IndicatorsGrid
                                        indicators={data?.indicators}
                                    />
                                </>
                            )}
                        </div>
                    </div>
                )}

                {/* ── Empty state ── */}
                {backendReady && !loading && !data && !error && (
                    <div className="pred-empty">
                        <span className="pred-empty-icon">🔮</span>
                        <p>Select a coin above to run the ensemble forecast.</p>
                    </div>
                )}

                {/* ── Timestamp ── */}
                {data?.generated_at && !loading && (
                    <p className="pred-footer-ts">
                        Forecast generated at {new Date(data.generated_at).toLocaleString()}
                    </p>
                )}

            </div>
        </div>
    );
}
