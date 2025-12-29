// frontend/src/components/Portfolio.jsx
import React, { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";
import { useTheme } from "../theme/ThemeContext";

const Portfolio = () => {
  const { theme } = useTheme();
  const accent = theme.accent || "#22d3ee";

  const [account, setAccount] = useState(null);
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [acctRes, posRes] = await Promise.all([
          fetch(`${API_BASE_URL}/portfolio/account`),
          fetch(`${API_BASE_URL}/portfolio/positions`),
        ]);
        if (!acctRes.ok) throw new Error(`Account error ${acctRes.status}`);
        if (!posRes.ok) throw new Error(`Positions error ${posRes.status}`);
        const acctData = await acctRes.json();
        const posData = await posRes.json();
        setAccount(acctData);
        setPositions(Array.isArray(posData) ? posData : []);
      } catch (e) {
        console.error("Portfolio load error", e);
        setError(e.message || "Failed to load portfolio.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const fmtCash = (v) =>
    v == null || isNaN(v) ? "$0.00" : `$${Number(v).toFixed(2)}`;

  const fmtPrice = (v) =>
    v == null || isNaN(v) ? "$0.00" : `$${Number(v).toFixed(2)}`;

  return (
    <div
      style={{
        padding: "16px",
        color: "#e5e7eb",
      }}
    >
      {/* Title like screenshot */}
      <div
        style={{
          fontSize: "22px",
          fontWeight: 700,
          color: accent,
          marginBottom: "12px",
        }}
      >
        Portfolio
      </div>

      <div
        style={{
          borderRadius: "18px",
          background: "#020617",
          border: "1px solid #111827",
          padding: "14px 20px",
          minHeight: "120px",
        }}
      >
        {loading && (
          <div style={{ fontSize: "13px", opacity: 0.8 }}>Loading…</div>
        )}
        {error && (
          <div style={{ fontSize: "13px", color: "#fca5a5" }}>{error}</div>
        )}

        {/* Cash row */}
        {account && (
          <div
            style={{
              fontSize: "14px",
              marginBottom: "10px",
            }}
          >
            <span style={{ opacity: 0.85 }}>Cash: </span>
            <span style={{ fontWeight: 600 }}>{fmtCash(account.cash)}</span>
          </div>
        )}

        {/* Table header */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr",
            fontSize: "13px",
            fontWeight: 600,
            marginBottom: "6px",
          }}
        >
          <span>Symbol</span>
          <span style={{ textAlign: "right" }}>Qty</span>
          <span style={{ textAlign: "right" }}>Avg</span>
          <span style={{ textAlign: "right" }}>Price</span>
          <span style={{ textAlign: "right" }}>P&amp;L</span>
        </div>

        {/* Positions */}
        {positions.length === 0 && !loading && (
          <div style={{ fontSize: "12px", opacity: 0.7 }}>No positions.</div>
        )}

        {positions.map((p) => {
          const qty = Number(p.qty || 0);
          const avg = Number(p.avg_entry_price || 0);
          const price = Number(p.current_price || p.market_price || p.lastday_price || avg);
          const pl =
            p.unrealized_pl != null
              ? Number(p.unrealized_pl)
              : (price - avg) * qty;
          const plColor = pl >= 0 ? "#22c55e" : "#ef4444";

          return (
            <div
              key={p.symbol}
              style={{
                display: "grid",
                gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr",
                fontSize: "13px",
                padding: "2px 0",
              }}
            >
              <span>{p.symbol}</span>
              <span style={{ textAlign: "right" }}>{qty}</span>
              <span style={{ textAlign: "right" }}>{fmtPrice(avg)}</span>
              <span style={{ textAlign: "right" }}>{fmtPrice(price)}</span>
              <span style={{ textAlign: "right", color: plColor }}>
                {fmtPrice(pl)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Portfolio;
