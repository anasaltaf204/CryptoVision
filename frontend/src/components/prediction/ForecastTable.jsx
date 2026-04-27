import React, { memo } from "react";
import { MODELS, HORIZON_LABELS } from "../../constants/prediction";
import { fmtPrice, fmtPct, calcChange } from "./utils";

function ChangeCell({ current, forecast }) {
    const chg = calcChange(current, forecast);
    if (chg === null) return <span className="pred-na">—</span>;
    return (
        <span className={`pred-change-badge${chg >= 0 ? " up" : " down"}`}>
            {chg >= 0 ? "▲" : "▼"} {Math.abs(chg).toFixed(2)}%
        </span>
    );
}

function ConfidenceBar({ lower, upper, current }) {
    if (lower == null || upper == null || !current) return <span className="pred-na">—</span>;
    const lo = calcChange(current, lower);
    const hi = calcChange(current, upper);
    if (lo == null || hi == null) return <span className="pred-na">—</span>;
    return (
        <div className="pred-ci-bar-wrap" title={`${fmtPrice(lower)} – ${fmtPrice(upper)}`}>
            <div className="pred-ci-range">
                <span className="pred-ci-low">{fmtPct(lo)}</span>
                <div className="pred-ci-track">
                    <div
                        className="pred-ci-fill"
                        style={{
                            left:  `${Math.max(0, Math.min(100, (lo + 50)))}%`,
                            right: `${Math.max(0, Math.min(100, (50 - hi)))}%`,
                        }}
                    />
                    <div className="pred-ci-zero" />
                </div>
                <span className="pred-ci-high">{fmtPct(hi)}</span>
            </div>
            <div className="pred-ci-prices">{fmtPrice(lower)} – {fmtPrice(upper)}</div>
        </div>
    );
}

const ForecastTable = memo(({ forecast, visibleModels }) => {
    if (!forecast?.horizons?.length) return null;
    const cur = forecast.current_price;

    return (
        <div className="pred-card">
            <div className="pred-card-header">
                <h3 className="pred-card-title"> Price Forecast</h3>
                {cur && (
                    <div className="pred-current-price-block">
                        <span className="pred-current-label">Current</span>
                        <span className="pred-current-value">{fmtPrice(cur)}</span>
                    </div>
                )}
            </div>
            <div className="pred-table-scroll">
                <table className="pred-table">
                    <thead>
                        <tr>
                            <th className="col-horizon">Horizon</th>
                            {MODELS.filter(m => visibleModels.has(m.key)).map(m => (
                                <th key={m.key} className={m.primary ? "col-ensemble" : ""}>
                                    <span className="pred-model-dot-sm" style={{ background: m.color }} />
                                    {m.label}
                                </th>
                            ))}
                            <th className="col-change">Change</th>
                            <th className="col-ci">95% Confidence Interval</th>
                        </tr>
                    </thead>
                    <tbody>
                        {forecast.horizons.map((h, i) => {
                            const ep = forecast.ensemble?.[i];
                            return (
                                <tr key={h} className="pred-table-row">
                                    <td className="pred-horizon-cell">
                                        <span className="pred-horizon-tag">{h}</span>
                                        <span className="pred-horizon-full">{HORIZON_LABELS[h]}</span>
                                    </td>
                                    {MODELS.filter(m => visibleModels.has(m.key)).map(m => {
                                        const v = forecast[m.key]?.[i];
                                        return (
                                            <td key={m.key} className={`pred-price-cell${m.primary ? " ensemble-col" : ""}`}>
                                                {v != null ? (
                                                    <span style={m.primary ? { color: "#4f7ef7" } : {}}>{fmtPrice(v)}</span>
                                                ) : <span className="pred-na">—</span>}
                                            </td>
                                        );
                                    })}
                                    <td className="pred-change-cell">
                                        <ChangeCell current={cur} forecast={ep} />
                                    </td>
                                    <td className="pred-ci-cell">
                                        <ConfidenceBar
                                            lower={forecast.ensemble_lower?.[i]}
                                            upper={forecast.ensemble_upper?.[i]}
                                            current={cur}
                                        />
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
});

ForecastTable.displayName = "ForecastTable";
export default ForecastTable;
