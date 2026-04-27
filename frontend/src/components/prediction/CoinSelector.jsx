import React from "react";
import { COINS } from "../../constants/prediction";

export default function CoinSelector({ selected, onSelect, disabled, trainedCoins }) {
    return (
        <div className="pred-coin-strip">
            {COINS.map((coin) => {
                const isActive  = selected === coin.symbol;
                const isReady   = !trainedCoins.length || trainedCoins.includes(coin.symbol);
                return (
                    <button
                        key={coin.symbol}
                        className={`pred-coin-btn${isActive ? " active" : ""}${!isReady ? " not-ready" : ""}`}
                        style={isActive ? { "--coin-color": coin.color, borderColor: coin.color, boxShadow: `0 0 0 3px ${coin.color}22` } : {}}
                        onClick={() => onSelect(coin.symbol)}
                        disabled={disabled || !isReady}
                    >
                        <span className="pred-coin-icon" style={isActive ? { color: coin.color } : { color: coin.color }}>{coin.icon}</span>
                        <div className="pred-coin-text">
                            <span className="pred-coin-name">{coin.name}</span>
                            <span className="pred-coin-sym">{coin.symbol}</span>
                        </div>
                        {!isReady && <span className="pred-training-dot" title="Training…" />}
                    </button>
                );
            })}
        </div>
    );
}
