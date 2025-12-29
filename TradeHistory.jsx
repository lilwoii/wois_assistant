// frontend/src/components/TradeHistory.jsx
import React, { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";
import { useTheme } from "../theme/ThemeContext";

const TradeHistory = () => {
  const { theme } = useTheme();
  const accent = theme.accent || "#22d3ee";

  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadOrders = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE_URL}/orders/history?status=all&limit=200`
        );
        if (!res.ok) throw new Error(`Orders error ${res.status}`);
        const data = await res.json();
        setOrders(Array.isArray(data) ? data : []);
      } catch (e) {
        console.error("Orders load error", e);
        setError(e.message || "Failed to load orders.");
      } finally {
        setLoading(false);
      }
    };
    loadOrders();
  }, []);

  const fmtTs = (t) => {
    if (!t) return "--";
    try {
      return new Date(t).toLocaleString();
    } catch {
      return String(t);
    }
  };

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
        Trade History
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

        {/* Header row */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "2.6fr 1.2fr 1fr 0.8fr 1fr",
            fontSize: "13px",
            fontWeight: 600,
            marginBottom: "6px",
          }}
        >
          <span>Time</span>
          <span>Symbol</span>
          <span>Side</span>
          <span style={{ textAlign: "right" }}>Qty</span>
          <span style={{ textAlign: "right" }}>Price</span>
        </div>

        {/* Rows */}
        {orders.length === 0 && !loading && (
          <div style={{ fontSize: "12px", opacity: 0.7 }}>No orders.</div>
        )}

        {orders.map((o) => {
          const side = (o.side || "").toUpperCase();
          const qty = Number(o.filled_qty || o.qty || 0);
          const price =
            o.filled_avg_price != null
              ? Number(o.filled_avg_price)
              : o.limit_price != null
              ? Number(o.limit_price)
              : 0;
          const sideColor =
            side === "BUY" ? "#22c55e" : side === "SELL" ? "#ef4444" : "#e5e7eb";

          return (
            <div
              key={o.id || `${o.symbol}-${o.submitted_at}`}
              style={{
                display: "grid",
                gridTemplateColumns: "2.6fr 1.2fr 1fr 0.8fr 1fr",
                fontSize: "13px",
                padding: "2px 0",
              }}
            >
              <span>{fmtTs(o.submitted_at)}</span>
              <span>{o.symbol}</span>
              <span style={{ color: sideColor }}>{side}</span>
              <span style={{ textAlign: "right" }}>{qty}</span>
              <span style={{ textAlign: "right" }}>{fmtPrice(price)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TradeHistory;
