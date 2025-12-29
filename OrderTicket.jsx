import React, { useState, useEffect } from "react";
import { placeOrder } from "../services/api";

export default function OrderTicket({ symbol, clickInfo, onClose }) {
  const [side, setSide] = useState("buy");
  const [qty, setQty] = useState("1");
  const [status, setStatus] = useState("");

  useEffect(() => {
    setStatus("");
  }, [symbol, clickInfo]);

  if (!clickInfo) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setStatus("Placing order…");
      const body = {
        symbol: symbol.toUpperCase(),
        side,
        qty: Number(qty) || 1,
        type: "market",
        time_in_force: "day",
        entry_price: clickInfo.price,
      };
      const res = await placeOrder(body);
      setStatus("Order placed (see backend / broker).");
      console.log("Order response:", res);
    } catch (err) {
      console.error("placeOrder failed:", err);
      setStatus("Order failed – see console / backend logs.");
    }
  };

  return (
    <div className="card" style={{ marginTop: "10px" }}>
      <div className="card-title">Quick order from chart</div>
      <p className="text-muted">
        Clicked price: <strong>{clickInfo.price.toFixed(2)}</strong>
      </p>
      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", gap: "8px", marginTop: "6px", flexWrap: "wrap" }}
      >
        <select
          value={side}
          onChange={(e) => setSide(e.target.value)}
          style={{
            borderRadius: "999px",
            border: "1px solid rgba(148,163,184,0.4)",
            background: "rgba(15,23,42,0.85)",
            color: "#f9fafb",
            padding: "4px 8px",
            fontSize: "0.8rem",
          }}
        >
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
        <input
          type="number"
          min="1"
          value={qty}
          onChange={(e) => setQty(e.target.value)}
          style={{
            width: "70px",
            borderRadius: "999px",
            border: "1px solid rgba(148,163,184,0.4)",
            background: "rgba(15,23,42,0.85)",
            color: "#f9fafb",
            padding: "4px 8px",
            fontSize: "0.8rem",
          }}
        />
        <button type="submit" className="btn-primary">
          Place {side.toUpperCase()}
        </button>
        <button
          type="button"
          onClick={onClose}
          style={{
            borderRadius: "999px",
            border: "none",
            padding: "4px 10px",
            fontSize: "0.8rem",
            background: "rgba(148,163,184,0.2)",
            color: "#e5e7eb",
            cursor: "pointer",
          }}
        >
          Close
        </button>
      </form>
      {status && <p className="text-muted" style={{ marginTop: "6px" }}>{status}</p>}
    </div>
  );
}
