import React from "react";

export default function Sidebar({ active, onChange }) {
  const items = [
    { key: "dashboard", label: "Dashboard" },
    { key: "portfolio", label: "Portfolio" },
    { key: "history", label: "Trade History" },
    { key: "ai", label: "AI Trainer" },
    { key: "settings", label: "Settings" },
    { key: "backtest", label: "Backtesting" },
  ];

  return (
    <aside
      style={{
        width: 220,
        background: "#020617",
        borderRight: "1px solid #111827",
        padding: "16px 12px",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "#22d3ee",
            marginBottom: 2,
          }}
        >
          WOIS Assistant
        </div>
        <div style={{ fontSize: 11, color: "#6b7280" }}>AI Trading Suite</div>
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {items.map((item) => (
          <button
            key={item.key}
            onClick={() => onChange(item.key)}
            style={{
              textAlign: "left",
              borderRadius: 999,
              border:
                active === item.key
                  ? "1px solid #22d3ee66"
                  : "1px solid #111827",
              background:
                active === item.key
                  ? "rgba(34,211,238,0.1)"
                  : "transparent",
              color: active === item.key ? "#e5e7eb" : "#9ca3af",
              fontSize: 12,
              padding: "6px 12px",
              cursor: "pointer",
            }}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  );
}
