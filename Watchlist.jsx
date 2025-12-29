// frontend/src/components/Watchlist.jsx
import React, { useEffect, useState } from "react";
import { addWatch, listWatch, delWatch, setSymbol } from "../services/api";

// Tiny inline sparkline for the Trend column
function Sparkline({ data }) {
  if (!Array.isArray(data) || data.length === 0) {
    return (
      <span style={{ fontSize: 11, opacity: 0.5 }}>
        —
      </span>
    );
  }

  const w = 70;
  const h = 22;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / Math.max(data.length - 1, 1)) * w;
      const y = h - ((v - min) / span) * h;
      return `${x},${y}`;
    })
    .join(" ");

  const first = data[0];
  const last = data[data.length - 1];
  const isUp = last >= first;
  const stroke = isUp ? "#22c55e" : "#ef4444";

  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      style={{ display: "block" }}
    >
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        points={points}
      />
    </svg>
  );
}

export default function Watchlist({ onSelect }) {
  const [rows, setRows] = useState([]);
  const [input, setInput] = useState("");

  const refresh = async () => {
    try {
      const data = await listWatch();
      // Expecting each row like:
      // { symbol, price, change_pct, sparkline: [ ... ] }
      setRows(Array.isArray(data) ? data : []);
      console.log("Watchlist rows:", data); // helpful debug in dev tools
    } catch (e) {
      console.error("listWatch failed:", e);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleAdd = async () => {
    const sym = input.trim().toUpperCase();
    if (!sym) return;
    try {
      await addWatch(sym);
      setInput("");
      await refresh();
    } catch (e) {
      console.error("addWatch failed:", e);
    }
  };

  const handleDelete = async (sym) => {
    try {
      await delWatch(sym);
      await refresh();
    } catch (e) {
      console.error("delWatch failed:", e);
    }
  };

  const handleSelect = async (sym) => {
    try {
      await setSymbol(sym);
    } catch (e) {
      console.error("setSymbol failed:", e);
    }
    if (onSelect) onSelect(sym);
  };

  return (
    <>
      {/* Small helper text above input */}
      <div className="watchlist-header">
        <span className="text-muted">Autosyncs with live prices</span>
      </div>

      {/* Input row */}
      <div className="watchlist-input-row">
        <input
          className="watchlist-input"
          placeholder="Add ticker (AAPL, TSLA…)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAdd()}
        />
        <button className="watchlist-add-btn" onClick={handleAdd}>
          +
        </button>
      </div>

      {/* Column headers: Symbol / Price / % / Trend / trash */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.1fr 0.9fr 0.7fr 1fr 32px",
          fontSize: 11,
          opacity: 0.65,
          marginTop: 8,
          marginBottom: 4,
        }}
      >
        <span>Symbol</span>
        <span style={{ textAlign: "right" }}>Price</span>
        <span style={{ textAlign: "right" }}>%</span>
        <span style={{ textAlign: "center" }}>Trend</span>
        <span />
      </div>

      {/* Rows */}
      <div className="watchlist-list">
        {rows.length === 0 && (
          <div className="text-muted">No symbols yet. Add one above.</div>
        )}

        {rows.map((r) => {
          const price =
            typeof r.price === "number" ? r.price.toFixed(2) : "--";
          const change =
            typeof r.change_pct === "number"
              ? `${r.change_pct.toFixed(2)}%`
              : "--";
          const changeClass =
            typeof r.change_pct !== "number"
              ? "badge-neutral"
              : r.change_pct >= 0
              ? "badge-up"
              : "badge-down";

          return (
            <div
              key={r.symbol}
              onClick={() => handleSelect(r.symbol)}
              className="watchlist-row"
              style={{
                display: "grid",
                gridTemplateColumns: "1.1fr 0.9fr 0.7fr 1fr 32px",
                alignItems: "center",
                columnGap: 8,
                cursor: "pointer",
              }}
            >
              {/* Symbol */}
              <div className="watch-symbol">{r.symbol}</div>

              {/* Price */}
              <div style={{ textAlign: "right" }}>${price}</div>

              {/* % change */}
              <div style={{ textAlign: "right" }}>
                <span className={changeClass}>{change}</span>
              </div>

              {/* Trend sparkline – THIS is the important part */}
              <div style={{ justifySelf: "center" }}>
                <Sparkline data={r.sparkline || []} />
              </div>

              {/* Delete button */}
              <button
                className="watch-delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(r.symbol);
                }}
              >
                🗑
              </button>
            </div>
          );
        })}
      </div>
    </>
  );
}
