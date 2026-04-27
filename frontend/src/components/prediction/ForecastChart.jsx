import React, { memo, useMemo } from "react";
import {
    Chart as ChartJS, CategoryScale, LinearScale,
    PointElement, LineElement, Tooltip, Legend, Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { MODELS } from "../../constants/prediction";
import { fmtPrice } from "./utils";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const tooltipBase = {
    backgroundColor: "rgba(13,17,23,0.95)",
    padding: 14,
    cornerRadius: 8,
    titleFont: { family: "'Sora', system-ui", size: 12, weight: "600" },
    bodyFont: { family: "'IBM Plex Mono', monospace", size: 12 },
    borderColor: "rgba(255,255,255,0.08)",
    borderWidth: 1,
};

const ForecastChart = memo(({ forecast, visibleModels, showCI }) => {
    if (!forecast?.horizons?.length) return null;

    const datasets = useMemo(() => {
        const sets = [];

        // CI bands first (drawn behind lines)
        if (showCI && forecast.ensemble_lower?.some(v => v != null)) {
            sets.push({
                label: "_ci_upper",
                data: forecast.ensemble_upper,
                borderColor: "transparent",
                backgroundColor: "rgba(79,126,247,0.08)",
                fill: "+1",
                tension: 0.4,
                pointRadius: 0,
                spanGaps: true,
                order: 10,
            });
            sets.push({
                label: "_ci_lower",
                data: forecast.ensemble_lower,
                borderColor: "transparent",
                backgroundColor: "rgba(79,126,247,0.08)",
                fill: false,
                tension: 0.4,
                pointRadius: 0,
                spanGaps: true,
                order: 10,
            });
        }

        // Model lines
        MODELS.forEach((m, idx) => {
            if (!visibleModels.has(m.key)) return;
            const vals = forecast[m.key];
            if (!vals?.some(v => v != null)) return;
            sets.push({
                label: m.label,
                data: vals,
                borderColor: m.color,
                backgroundColor: m.primary ? m.color + "14" : "transparent",
                fill: false,
                borderWidth: m.primary ? 3 : 1.8,
                borderDash: m.primary ? [] : [5, 4],
                tension: 0.4,
                pointRadius: m.primary ? 6 : 4,
                pointBackgroundColor: m.color,
                pointBorderColor: "#fff",
                pointBorderWidth: 2,
                pointHoverRadius: m.primary ? 8 : 6,
                spanGaps: true,
                order: m.primary ? 1 : idx + 2,
            });
        });

        return sets;
    }, [forecast, visibleModels, showCI]);

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        animation: { duration: 400, easing: "easeOutQuart" },
        plugins: {
            legend: {
                position: "bottom",
                labels: {
                    filter: item => !item.text.startsWith("_"),
                    font: { family: "'Sora', system-ui", size: 12 },
                    padding: 20,
                    usePointStyle: true,
                    pointStyleWidth: 28,
                },
            },
            tooltip: {
                ...tooltipBase,
                callbacks: {
                    title: (items) => {
                        const h = items[0]?.label;
                        const labels = { "1w": "1 Week", "1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "1y": "1 Year" };
                        return labels[h] || h;
                    },
                    label: (ctx) => {
                        if (ctx.dataset.label?.startsWith("_")) return null;
                        const v = ctx.parsed.y;
                        return v != null ? ` ${ctx.dataset.label}: ${fmtPrice(v)}` : null;
                    },
                    afterBody: (items) => {
                        if (!showCI || !forecast.ensemble_lower) return [];
                        const i = items[0]?.dataIndex;
                        const lo = forecast.ensemble_lower[i];
                        const hi = forecast.ensemble_upper[i];
                        if (lo == null || hi == null) return [];
                        return [``, ` 95% CI: ${fmtPrice(lo)} – ${fmtPrice(hi)}`];
                    },
                },
            },
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: {
                    font: { family: "'Sora', system-ui", size: 12 },
                    color: "#9aa0b0",
                },
            },
            y: {
                beginAtZero: false,
                grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
                ticks: {
                    font: { family: "'IBM Plex Mono', monospace", size: 11 },
                    color: "#9aa0b0",
                    callback: v => v >= 1000 ? "$" + (v / 1000).toFixed(0) + "K" : "$" + v.toFixed(2),
                },
            },
        },
    }), [forecast, showCI]);

    return (
        <div className="pred-card">
            <div className="pred-card-header">
                <h3 className="pred-card-title"> Forecast Comparison</h3>
                <span className="pred-card-hint">Hover for details</span>
            </div>
            <div className="pred-chart-wrap">
                <Line data={{ labels: forecast.horizons, datasets }} options={options} />
            </div>
        </div>
    );
});

ForecastChart.displayName = "ForecastChart";
export default ForecastChart;
