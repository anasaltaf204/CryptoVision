import React, { memo, useMemo } from "react";
import { Bar, Line, Doughnut } from "react-chartjs-2";

const COLORS = ["#667eea", "#764ba2", "#3b82f6", "#10b981", "#f43f5e"];
const COLORS_10 = [...COLORS, "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#14b8a6"];

const tooltipBase = {
    backgroundColor: "rgba(15,15,30,0.92)",
    padding: 12,
    cornerRadius: 8,
    titleFont: { weight: 600, size: 13 },
    bodyFont: { size: 12 },
    borderColor: "rgba(255,255,255,0.08)",
    borderWidth: 1,
};

function SkeletonChart({ height = 260 }) {
    return <div className="chart-skeleton" style={{ height }} />;
}

// ── Bar: Top 5 Market Caps ────────────────────────────────────────────────────
const MarketCapChart = memo(({ coins }) => {
    const top5 = coins.slice(0, 5);
    const data = useMemo(() => ({
        labels: top5.map(c => c.symbol.toUpperCase()),
        datasets: [{
            label: "Market Cap",
            data: top5.map(c => c.market_cap),
            backgroundColor: COLORS.map(c => c + "CC"),
            borderColor: COLORS,
            borderWidth: 2,
            borderRadius: 6,
        }],
    }), [top5.map(c => c.id).join()]);

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                ...tooltipBase,
                callbacks: {
                    label: ctx => " $" + (ctx.parsed.y / 1e9).toFixed(1) + "B",
                },
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
                ticks: { font: { size: 11 }, callback: v => "$" + (v / 1e9).toFixed(0) + "B" },
            },
            x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        },
    }), []);

    return <Bar data={data} options={options} />;
});
MarketCapChart.displayName = "MarketCapChart";

// ── Line: 7-Day Trend ─────────────────────────────────────────────────────────
const TrendChart = memo(({ coins }) => {
    const top5 = coins.slice(0, 5);
    const data = useMemo(() => ({
        labels: top5[0]?.sparkline_in_7d?.price.map((_, i) => {
            // Label every 24th point as a day marker
            const day = Math.floor(i / 24);
            return i % 24 === 0 ? `Day ${day + 1}` : "";
        }) || [],
        datasets: top5.map((c, idx) => ({
            label: c.symbol.toUpperCase(),
            data: c.sparkline_in_7d?.price || [],
            borderColor: COLORS[idx],
            backgroundColor: "transparent",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 1.8,
        })),
    }), [top5.map(c => c.id).join()]);

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
            legend: {
                position: "bottom",
                labels: { font: { size: 11 }, padding: 10, usePointStyle: true, boxWidth: 6 },
            },
            tooltip: {
                ...tooltipBase,
                callbacks: {
                    label: ctx => ` ${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString("en-US", {
                        minimumFractionDigits: 2, maximumFractionDigits: 2,
                    })}`,
                },
            },
        },
        scales: {
            y: {
                grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
                ticks: {
                    font: { size: 10 },
                    callback: v => "$" + v.toLocaleString("en-US", { maximumFractionDigits: 0 }),
                },
            },
            x: {
                grid: { display: false },
                ticks: { font: { size: 10 }, maxTicksLimit: 8, autoSkip: true },
            },
        },
    }), []);

    return <Line data={data} options={options} />;
});
TrendChart.displayName = "TrendChart";

// ── Doughnut: Market Dominance ────────────────────────────────────────────────
const DominanceChart = memo(({ globalStats }) => {
    const dominanceMap = globalStats?.market_cap_percentage || {};
    const top10Symbols = Object.entries(dominanceMap)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

    const data = useMemo(() => ({
        labels: top10Symbols.map(([sym]) => sym.toUpperCase()),
        datasets: [{
            label: "Dominance %",
            data: top10Symbols.map(([, pct]) => parseFloat(pct.toFixed(2))),
            backgroundColor: COLORS_10,
            borderColor: "#fff",
            borderWidth: 2,
            hoverOffset: 8,
        }],
    }), [JSON.stringify(dominanceMap)]);

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "bottom",
                labels: { font: { size: 10 }, padding: 8, usePointStyle: true, boxWidth: 8 },
            },
            tooltip: {
                ...tooltipBase,
                callbacks: {
                    label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(2)}%`,
                },
            },
        },
    }), []);

    return <Doughnut data={data} options={options} />;
});
DominanceChart.displayName = "DominanceChart";

// ── ChartPanel: parent grid ───────────────────────────────────────────────────
const ChartPanel = memo(({ coins, globalStats, coinsLoading, statsLoading }) => {
    return (
        <div className="chart-panel">
            <div className="chart-card">
                <div className="chart-card__title"> Top 5 Market Caps</div>
                <div className="chart-card__body">
                    {coinsLoading
                        ? <SkeletonChart />
                        : <MarketCapChart coins={coins} />
                    }
                </div>
            </div>

            <div className="chart-card">
                <div className="chart-card__title"> 7-Day Price Trends</div>
                <div className="chart-card__body">
                    {coinsLoading
                        ? <SkeletonChart />
                        : <TrendChart coins={coins} />
                    }
                </div>
            </div>

            <div className="chart-card">
                <div className="chart-card__title"> Market Dominance</div>
                <div className="chart-card__body">
                    {statsLoading
                        ? <SkeletonChart />
                        : <DominanceChart globalStats={globalStats} />
                    }
                </div>
            </div>
        </div>
    );
});

ChartPanel.displayName = "ChartPanel";
export default ChartPanel;
