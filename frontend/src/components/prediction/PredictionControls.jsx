import React from "react";
import { ALL_HORIZONS, HORIZON_LABELS, MODELS } from "../../constants/prediction";

export default function PredictionControls({ activeHorizons, toggleHorizon, visibleModels, toggleModel, showCI, setShowCI }) {
    return (
        <div className="pred-controls-bar">
            {/* Horizon toggles */}
            <div className="pred-controls-group">
                <span className="pred-controls-label">Horizons</span>
                <div className="pred-toggle-group">
                    {ALL_HORIZONS.map(h => (
                        <button
                            key={h}
                            className={`pred-toggle-btn${activeHorizons.has(h) ? " active" : ""}`}
                            onClick={() => toggleHorizon(h)}
                            title={HORIZON_LABELS[h]}
                        >
                            {h}
                        </button>
                    ))}
                </div>
            </div>

            <div className="pred-controls-divider" />

            {/* Model toggles */}
            <div className="pred-controls-group">
                <span className="pred-controls-label">Models</span>
                <div className="pred-toggle-group">
                    {MODELS.map(m => (
                        <button
                            key={m.key}
                            className={`pred-toggle-btn pred-model-btn${visibleModels.has(m.key) ? " active" : ""}${m.primary ? " primary" : ""}`}
                            style={visibleModels.has(m.key) ? { "--model-color": m.color } : {}}
                            onClick={() => toggleModel(m.key)}
                            title={m.key === "ensemble" ? "Always visible" : `Toggle ${m.label}`}
                            disabled={m.key === "ensemble"}
                        >
                            <span className="pred-model-dot" style={{ background: m.color }} />
                            {m.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="pred-controls-divider" />

            {/* CI toggle */}
            <div className="pred-controls-group">
                <span className="pred-controls-label">Options</span>
                <button
                    className={`pred-toggle-btn${showCI ? " active" : ""}`}
                    onClick={() => setShowCI(v => !v)}
                >
                    95% CI Band
                </button>
            </div>
        </div>
    );
}
