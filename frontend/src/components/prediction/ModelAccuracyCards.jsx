import React, { memo } from "react";
import { MODELS } from "../../constants/prediction";

const HORIZON_ORDER = ["1w", "1m", "3m", "6m", "1y"];

function AccuracyRing({ pct }) {
    const r = 20, circ = 2 * Math.PI * r;
    const dash = (pct / 100) * circ;
    const color = pct >= 65 ? "#10b981" : pct >= 50 ? "#f59e0b" : "#ef4444";
    return (
        <svg width="52" height="52" viewBox="0 0 52 52">
            <circle cx="26" cy="26" r={r} fill="none" stroke="#e8eaf0" strokeWidth="4" />
            <circle
                cx="26" cy="26" r={r} fill="none"
                stroke={color} strokeWidth="4"
                strokeDasharray={`${dash} ${circ - dash}`}
                strokeLinecap="round"
                transform="rotate(-90 26 26)"
            />
            <text x="26" y="30" textAnchor="middle" fontSize="10" fontWeight="700" fill={color} fontFamily="'IBM Plex Mono',monospace">
                {Math.round(pct)}%
            </text>
        </svg>
    );
}

const ModelAccuracyCards = memo(({ validationMetrics }) => {
    if (!validationMetrics || !Object.keys(validationMetrics).length) return null;

    return (
        <div className="pred-card">
            <div className="pred-card-header">
                <h3 className="pred-card-title">📐 Model Accuracy</h3>
                <span className="pred-card-hint">Directional accuracy on validation set</span>
            </div>
            <div className="pred-accuracy-grid">
                {MODELS.map(model => {
                    const metrics = validationMetrics[model.key];
                    if (!metrics) return null;

                    // Average directional accuracy across horizons
                    const horizonKeys = Object.keys(metrics);
                    const avgDirAcc = horizonKeys.length
                        ? horizonKeys.reduce((sum, h) => sum + (metrics[h]?.dir_acc ?? 0), 0) / horizonKeys.length * 100
                        : null;
                    const avgMape = horizonKeys.length
                        ? horizonKeys.reduce((sum, h) => sum + (metrics[h]?.mape ?? 0), 0) / horizonKeys.length
                        : null;

                    return (
                        <div key={model.key} className="pred-accuracy-card">
                            <div className="pred-acc-header">
                                <span className="pred-acc-dot" style={{ background: model.color }} />
                                <span className="pred-acc-name">{model.label}</span>
                            </div>
                            <div className="pred-acc-body">
                                {avgDirAcc != null && <AccuracyRing pct={avgDirAcc} />}
                                <div className="pred-acc-stats">
                                    {avgMape != null && (
                                        <div className="pred-acc-stat">
                                            <span className="pred-acc-stat-label">Avg MAPE</span>
                                            <span className="pred-acc-stat-val">{avgMape.toFixed(2)}%</span>
                                        </div>
                                    )}
                                    {HORIZON_ORDER.filter(h => metrics[h]).slice(0, 3).map(h => (
                                        <div key={h} className="pred-acc-stat">
                                            <span className="pred-acc-stat-label">{h} dir</span>
                                            <span className={`pred-acc-stat-val ${metrics[h].dir_acc >= 0.55 ? "good" : "warn"}`}>
                                                {(metrics[h].dir_acc * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
});

ModelAccuracyCards.displayName = "ModelAccuracyCards";
export default ModelAccuracyCards;
