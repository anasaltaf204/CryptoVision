import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowsRotate, faBrain } from "@fortawesome/free-solid-svg-icons";
import { COINS } from "../../constants/prediction";

export default function PredictionHeader({ selectedCoin, lastUpdated, loading, backendReady, onRefresh }) {
    const coin = COINS.find(c => c.symbol === selectedCoin);
    return (
        <div className="pred-header">
            <div className="pred-header-left">
                <div className="pred-title-row">
                    <span className="pred-title-icon"><FontAwesomeIcon icon={faBrain} /></span>
                    <h1 className="pred-title">AI Price Prediction</h1>
                    {coin && (
                        <span className="pred-title-badge" style={{ background: coin.color + "22", color: coin.color, borderColor: coin.color + "55" }}>
                            <span style={{ color: coin.color }}>{coin.icon}</span> {coin.symbol}
                        </span>
                    )}
                </div>
                <p className="pred-subtitle">
                    Ensemble model · Prophet · SARIMA · LSTM · Transformer 
                </p>
            </div>
            <div className="pred-header-right">
                {lastUpdated && (
                    <span className="pred-updated">
                        Updated {lastUpdated.toLocaleTimeString()}
                    </span>
                )}
                <button
                    className={`pred-refresh-btn${loading ? " spinning" : ""}`}
                    onClick={onRefresh}
                    disabled={loading || !backendReady}
                >
                    <FontAwesomeIcon icon={faArrowsRotate} />
                    {loading ? "Fetching…" : "Refresh"}
                </button>
            </div>
        </div>
    );
}
