// frontend/src/App.jsx
import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";
import Settings from "./components/Settings";
import Portfolio from "./components/Portfolio";
import TradeHistory from "./components/TradeHistory";
import AILab from "./components/AILab";
import { ThemeProvider, useTheme } from "./theme/ThemeContext";
import Backtesting from "./components/Backtesting";
import Alerts from "./components/Alerts";

const TABS = [
  { id: "dashboard", label: "Dashboard", icon: "📊" },
  { id: "portfolio", label: "Portfolio", icon: "💼" },
  { id: "history", label: "Trade History", icon: "📜" },
  { id: "ai-lab", label: "AI Trainer", icon: "🧪" },
  { id: "backtest", label: "Backtesting", icon: "⏪" },
  { id: "alerts", label: "Alerts", icon: "🚨" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

const AppShell = () => {
  const [activeTab, setActiveTab] = useState("dashboard");
  const { theme } = useTheme();

  // simple robo head bob
  const [roboPhase, setRoboPhase] = useState(0);
  useEffect(() => {
    const id = setInterval(() => {
      setRoboPhase((p) => (p + 1) % 4);
    }, 700);
    return () => clearInterval(id);
  }, []);

  const bg =
    theme.mode === "neon"
      ? "radial-gradient(circle at top,#020617,#22c55e,#0ea5e9)"
      : theme.mode === "cyber"
      ? "radial-gradient(circle at bottom,#020617,#22d3ee,#a855f7)"
      : theme.mode === "plain"
      ? "#020617"
      : "radial-gradient(circle at top,#020617,#111827)";

  const accent = theme.accent || "#2563eb";

  const roboTransform =
    roboPhase === 0
      ? "translateY(0px)"
      : roboPhase === 1
      ? "translateY(-2px)"
      : roboPhase === 2
      ? "translateX(2px)"
      : "translateX(-2px)";

  const renderTab = () => {
    switch (activeTab) {
      case "dashboard":
        return <Dashboard />;
      case "portfolio":
        return <Portfolio />;
      case "history":
        return <TradeHistory />;
      case "settings":
        return <Settings />;
      case "ai-lab":
        return <AILab />;
      case "backtest":
        return <Backtesting />;
      case "alerts":
        return <Alerts />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div
      className="app-root"
      style={{
        minHeight: "100vh",
        display: "flex",
        background: bg,
        color: "#e5e7eb",
      }}
    >
      {/* SIDEBAR */}
      <aside
        style={{
          width: "230px",
          borderRight: "1px solid rgba(15,23,42,0.9)",
          background: "rgba(15,23,42,0.96)",
          display: "flex",
          flexDirection: "column",
          padding: "12px",
          boxShadow: "0 0 30px rgba(15,23,42,0.9)",
        }}
      >
        {/* Logo / robot */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            padding: "8px 8px 12px",
            borderBottom: "1px solid #111827",
            marginBottom: "10px",
          }}
        >
          <div
            style={{
              width: "34px",
              height: "34px",
              borderRadius: "999px",
              background: "radial-gradient(circle,#1f2937,#020617)",
              border: `2px solid ${accent}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "18px",
              transform: roboTransform,
              transition: "transform 0.25s ease-out",
            }}
          >
            🤖
          </div>
          <div>
            <div style={{ fontSize: "13px", opacity: 0.7 }}>
              Woi&apos;s Assistant
            </div>
            <div style={{ fontSize: "14px", fontWeight: 600 }}>
              AI Trading Suite
            </div>
          </div>
        </div>

        {/* Tabs */}
        <nav
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "4px",
            flex: 1,
          }}
        >
          {TABS.map((tab) => {
            const active = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  border: "none",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "8px 10px",
                  borderRadius: "8px",
                  cursor: "pointer",
                  background: active ? "rgba(15,23,42,0.9)" : "transparent",
                  color: active ? "#f9fafb" : "#9ca3af",
                  fontSize: "13px",
                  transition: "background 0.15s, color 0.15s",
                }}
              >
                <span
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </span>
                {active && (
                  <span
                    style={{
                      width: "6px",
                      height: "18px",
                      borderRadius: "999px",
                      background: accent,
                    }}
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Status footer */}
        <div
          style={{
            marginTop: "8px",
            padding: "8px",
            borderRadius: "8px",
            background: "rgba(15,23,42,0.9)",
            fontSize: "11px",
          }}
        >
          <div style={{ marginBottom: "4px" }}>
            <span
              style={{
                display: "inline-block",
                width: "7px",
                height: "7px",
                borderRadius: "999px",
                background: "#22c55e",
                marginRight: "6px",
              }}
            />
            AI Engine
          </div>
          <div style={{ opacity: 0.7 }}>
            Dashboard, AI Lab & signals are active.
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main
        style={{
          flex: 1,
          overflow: "auto",
        }}
      >
        {renderTab()}
      </main>
    </div>
  );
};

const App = () => (
  <ThemeProvider>
    <AppShell />
  </ThemeProvider>
);

export default App;
