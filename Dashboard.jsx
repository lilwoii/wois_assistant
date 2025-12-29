import React, { useEffect, useRef, useState } from "react";
import html2canvas from "html2canvas";
import { API_BASE_URL, WS_URL } from "../config";
import Chart from "./Chart";

const Dashboard = () => {
  const [symbol, setSymbol] = useState("TSLA");
  const [inputSymbol, setInputSymbol] = useState("TSLA");

  const [price, setPrice] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [aiInsight, setAiInsight] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);

  const [watchlist, setWatchlist] = useState([]);
  const [watchlistUpdating, setWatchlistUpdating] = useState(false);

  const [notifications, setNotifications] = useState([]);
  const [wsStatus, setWsStatus] = useState("disconnected");

  const [newWatchSymbol, setNewWatchSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("1D");

  const [rightPanelTab, setRightPanelTab] = useState("insight");

  const [quickLoading, setQuickLoading] = useState(false);

  // settings (paper vs live)
  const [usePaper, setUsePaper] = useState(true);
  const [settingsLoading, setSettingsLoading] = useState(false);

  // alerts
  const [alerts, setAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(false);
  const [newAlertPrice, setNewAlertPrice] = useState("");
  const [newAlertDirection, setNewAlertDirection] = useState("above");

  const wsRef = useRef(null);
  const chartRef = useRef(null);

  // ---------------------------
  // Helpers
  // ---------------------------
  const pushNotification = (message) => {
    setNotifications((prev) => [
      { id: Date.now(), message },
      ...prev.slice(0, 49),
    ]);
  };

  // ---------------------------
  // WebSocket handling
  // ---------------------------
  useEffect(() => {
    if (!WS_URL) return;

    setWsStatus("connecting");
    const ws = new WebSocket(`${WS_URL}?symbol=${symbol}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus("connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "price_update" || data.type === "tick") {
          if (data.price != null) setPrice(data.price);
        }

        if (data.type === "notification") {
          setNotifications((prev) => [
            { id: Date.now(), message: data.message },
            ...prev.slice(0, 49),
          ]);
        }

        if (data.type === "ai_insight") {
          setAiInsight(data.payload || data);
        }
      } catch (err) {
        console.error("WS message parse error:", err);
      }
    };

    ws.onerror = () => {
      setWsStatus("error");
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
    };

    return () => {
      ws.close();
    };
  }, [symbol]);

  // ---------------------------
  // Initial settings + alerts
  // ---------------------------
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/settings`);
        if (res.ok) {
          const data = await res.json();
          if (typeof data.use_paper === "boolean") {
            setUsePaper(data.use_paper);
          }
        }
      } catch (err) {
        console.error("Error loading settings", err);
      }
    };
    loadSettings();
  }, []);

  useEffect(() => {
    const loadAlerts = async () => {
      setAlertsLoading(true);
      try {
        const res = await fetch(
          `${API_BASE_URL}/alerts?symbol=${encodeURIComponent(symbol)}`
        );
        if (res.ok) {
          const data = await res.json();
          setAlerts(Array.isArray(data) ? data : []);
        }
      } catch (err) {
        console.error("Error loading alerts", err);
      } finally {
        setAlertsLoading(false);
      }
    };

    loadAlerts();
  }, [symbol]);

  // ---------------------------
  // Load initial dashboard data
  // ---------------------------
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        // price
        const priceRes = await fetch(
          `${API_BASE_URL}/price?symbol=${encodeURIComponent(symbol)}`
        );
        if (priceRes.ok) {
          const data = await priceRes.json();
          if (data.price != null) setPrice(data.price);
        }

        // watchlist
        const wlRes = await fetch(`${API_BASE_URL}/watchlist`);
        if (wlRes.ok) {
          const data = await wlRes.json();
          setWatchlist(Array.isArray(data) ? data : []);
        }

        // notifications (optional)
        const notifRes = await fetch(`${API_BASE_URL}/notifications`);
        if (notifRes.ok) {
          const data = await notifRes.json();
          if (Array.isArray(data)) {
            setNotifications(
              data
                .map((n, idx) => ({
                  id: n.id ?? idx,
                  message: n.message ?? String(n),
                }))
                .slice(-50)
                .reverse()
            );
          }
        }
      } catch (err) {
        console.error("Error loading initial dashboard data", err);
      }
    };

    fetchInitial();
  }, [symbol]);

  // ---------------------------
  // Symbol handling
  // ---------------------------
  const handleSymbolSubmit = (e) => {
    e.preventDefault();
    const clean = inputSymbol.trim().toUpperCase();
    if (!clean) return;
    setSymbol(clean);
  };

  // ---------------------------
  // AI insight helper (Scan button)
  // ---------------------------
  const runAIInsight = async (mode = "default") => {
    if (!symbol) return;

    setAiLoading(true);
    setAiInsight(null);
    try {
      const res = await fetch(
        `${API_BASE_URL}/ai/insight?symbol=${encodeURIComponent(
          symbol
        )}&mode=${encodeURIComponent(mode)}`
      );
      if (res.ok) {
        const data = await res.json();
        setAiInsight(data);
      } else {
        setAiInsight({
          summary: "AI analysis failed. Please try again.",
        });
      }
    } catch (err) {
      console.error("AI insight error", err);
      setAiInsight({
        summary: "AI analysis error. Check backend logs.",
      });
    } finally {
      setAiLoading(false);
    }
  };

  // ---------------------------
  // Paper vs Live toggle
  // ---------------------------
  const togglePaperMode = async () => {
    if (settingsLoading) return;
    const next = !usePaper;
    setSettingsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_paper: next }),
      });
      if (res.ok) {
        const data = await res.json();
        const value =
          typeof data.use_paper === "boolean" ? data.use_paper : next;
        setUsePaper(value);
        pushNotification(
          value
            ? "🧪 Switched to PAPER trading."
            : "💵 Switched to LIVE trading."
        );
      } else {
        pushNotification("❌ Failed to update trading mode.");
      }
    } catch (err) {
      console.error("Toggle paper/live error", err);
      pushNotification("❌ Failed to update trading mode.");
    } finally {
      setSettingsLoading(false);
    }
  };

  // ---------------------------
  // Chart screenshot helper
  // ---------------------------
  const captureChartBlob = async () => {
    if (!chartRef.current) return null;
    try {
      const canvas = await html2canvas(chartRef.current, {
        useCORS: true,
        backgroundColor: "#020617",
      });
      const blob = await new Promise((resolve) =>
        canvas.toBlob((b) => resolve(b), "image/png")
      );
      return blob;
    } catch (err) {
      console.error("Chart capture failed", err);
      return null;
    }
  };

  // ---------------------------
  // Quick trade helpers
  // ---------------------------
  const handleQuickBuyAI = async () => {
    if (!symbol || quickLoading) return;
    setQuickLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/orders/quick_buy_ai`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });

      let data = null;
      try {
        data = await res.json();
      } catch {
        // ignore
      }

      if (res.ok && data?.ok) {
        pushNotification(`🧠 Quick BUY AI executed for ${symbol}.`);
      } else {
        const reason =
          data?.reason || data?.error || "Signal not strong BUY.";
        pushNotification(`⚠️ Quick BUY skipped for ${symbol}: ${reason}`);
      }
    } catch (err) {
      console.error("Quick BUY error", err);
      pushNotification(`❌ Quick BUY failed for ${symbol}.`);
    } finally {
      setQuickLoading(false);
    }
  };

  const handleQuickSellAll = async () => {
    if (!symbol || quickLoading) return;
    setQuickLoading(true);
    try {
      // 1) Trend analysis for snapshot
      let trendData = null;
      try {
        const trendRes = await fetch(`${API_BASE_URL}/ai/trend`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol,
            tf: "1Min",
            limit: 400,
            heavy: false,
            send_to_discord: false,
          }),
        });
        if (trendRes.ok) {
          trendData = await trendRes.json();
        }
      } catch (err) {
        console.error("Trend analysis error", err);
      }

      // 2) Capture chart screenshot
      const blob = await captureChartBlob();

      // 3) Send snapshot to Discord (chart + analysis)
      if (trendData || blob) {
        const form = new FormData();
        form.append("symbol", symbol);
        form.append("analysis", JSON.stringify(trendData || {}));
        if (blob) {
          form.append("file", blob, `${symbol}_chart.png`);
        }

        try {
          const snapRes = await fetch(`${API_BASE_URL}/ai/trend/snapshot`, {
            method: "POST",
            body: form,
          });
          if (!snapRes.ok) {
            console.warn("Trend snapshot failed");
          }
        } catch (err) {
          console.error("Snapshot upload error", err);
        }
      }

      // 4) Actually close the position
      const res = await fetch(`${API_BASE_URL}/orders/quick_sell_all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });

      let data = null;
      try {
        data = await res.json();
      } catch {
        // ignore
      }

      if (res.ok && data?.ok) {
        pushNotification(`🔴 Quick SELL + snapshot executed for ${symbol}.`);
      } else {
        const reason =
          data?.error || "No open position or broker error.";
        pushNotification(
          `⚠️ Quick SELL snapshot skipped for ${symbol}: ${reason}`
        );
      }
    } catch (err) {
      console.error("Quick SELL snapshot error", err);
      pushNotification(`❌ Quick SELL + snapshot failed for ${symbol}.`);
    } finally {
      setQuickLoading(false);
    }
  };

  // ---------------------------
  // Watchlist
  // ---------------------------
  const handleAddWatchlist = async (e) => {
    e.preventDefault();
    const clean = newWatchSymbol.trim().toUpperCase();
    if (!clean) return;

    setWatchlistUpdating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/watchlist/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: clean }),
      });

      if (res.ok) {
        const updated = await res.json();
        setWatchlist((prev) => {
          const others = prev.filter((w) => w.symbol !== updated.symbol);
          return [...others, updated];
        });
        setNewWatchSymbol("");
      }
    } catch (err) {
      console.error("Error adding to watchlist", err);
    } finally {
      setWatchlistUpdating(false);
    }
  };

  const handleDeleteWatchItem = async (item) => {
    setWatchlistUpdating(true);
    try {
      const res = await fetch(
        `${API_BASE_URL}/watchlist/${encodeURIComponent(item.symbol)}`,
        {
          method: "DELETE",
        }
      );
      if (res.ok) {
        setWatchlist((prev) =>
          prev.filter((w) => w.symbol !== item.symbol)
        );
      }
    } catch (err) {
      console.error("Error deleting watchlist item", err);
    } finally {
      setWatchlistUpdating(false);
    }
  };

  // ---------------------------
  // Alerts
  // ---------------------------
  const handleCreateAlert = async (e) => {
    e.preventDefault();
    const priceVal = parseFloat(newAlertPrice);
    if (!symbol || Number.isNaN(priceVal)) return;

    try {
      const res = await fetch(`${API_BASE_URL}/alerts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          direction: newAlertDirection,
          price: priceVal,
          enabled: true,
        }),
      });

      if (res.ok) {
        const rule = await res.json();
        setAlerts((prev) => [...prev, rule]);
        setNewAlertPrice("");
        pushNotification(
          `🔔 Alert set for ${symbol}: ${newAlertDirection} ${priceVal.toFixed(
            2
          )}`
        );
      }
    } catch (err) {
      console.error("Create alert error", err);
      pushNotification("❌ Failed to create alert.");
    }
  };

  const handleDeleteAlert = async (id) => {
    try {
      const res = await fetch(`${API_BASE_URL}/alerts/${id}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setAlerts((prev) => prev.filter((a) => a.id !== id));
      }
    } catch (err) {
      console.error("Delete alert error", err);
    }
  };

  // ---------------------------
  // Timeframe
  // ---------------------------
  const timeframes = ["1D", "1W", "1M", "3M", "1Y"];

  // ---------------------------
  // AI display helper
  // ---------------------------
  const signal = (aiInsight && aiInsight.signal) || "NEUTRAL";
  const confidence =
    aiInsight && typeof aiInsight.confidence === "number"
      ? aiInsight.confidence.toFixed(2)
      : null;

  let signalColor = "#38bdf8";
  if (signal === "BUY") signalColor = "#22c55e";
  if (signal === "SELL") signalColor = "#ef4444";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "2.5fr 1.4fr",
        gap: "16px",
        height: "100%",
        padding: "16px",
        background: "#020617",
        color: "#e5e7eb",
      }}
    >
      {/* LEFT SIDE */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "12px",
          minWidth: 0,
        }}
      >
        {/* Top bar: symbol + ws status + paper/live */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <form
            onSubmit={handleSymbolSubmit}
            style={{ display: "flex", gap: "8px", flex: 1 }}
          >
            <input
              value={inputSymbol}
              onChange={(e) => setInputSymbol(e.target.value)}
              placeholder="Symbol (TSLA, AAPL, BTCUSD...)"
              style={{
                flex: 1,
                borderRadius: "999px",
                border: "1px solid #1f2933",
                background: "#020617",
                color: "#e5e7eb",
                padding: "8px 14px",
                outline: "none",
              }}
            />
            <button
              type="submit"
              style={{
                borderRadius: "999px",
                border: "1px solid #1d4ed8",
                background:
                  "radial-gradient(circle at top left, #22c55e, #0ea5e9, #6366f1)",
                color: "#0b1120",
                fontWeight: 600,
                padding: "8px 16px",
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              Load
            </button>
          </form>

          <div
            style={{
              fontSize: "12px",
              padding: "4px 10px",
              borderRadius: "999px",
              border: "1px solid #1f2933",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              background: "#020617",
            }}
          >
            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "999px",
                background:
                  wsStatus === "connected"
                    ? "#22c55e"
                    : wsStatus === "connecting"
                    ? "#eab308"
                    : "#ef4444",
              }}
            />
            <span style={{ textTransform: "capitalize" }}>
              {wsStatus}
            </span>
          </div>

          <button
            onClick={togglePaperMode}
            disabled={settingsLoading}
            style={{
              borderRadius: "999px",
              border: "1px solid #111827",
              padding: "4px 12px",
              fontSize: "12px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: usePaper ? "#020617" : "#111827",
              boxShadow: usePaper
                ? "0 0 0 1px rgba(148, 163, 184, 0.4)"
                : "0 0 0 1px rgba(248, 113, 113, 0.5)",
            }}
          >
            <span>{usePaper ? "🧪" : "💵"}</span>
            <span>{usePaper ? "Paper" : "Live"}</span>
          </button>
        </div>

        {/* Price + timeframe + quick trade */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "8px",
          }}
        >
          <div style={{ fontSize: "14px", opacity: 0.9 }}>
            <span style={{ fontWeight: 500 }}>{symbol}</span>
            {price != null && (
              <span style={{ marginLeft: "8px", fontSize: "16px" }}>
                ${price.toFixed(2)}
              </span>
            )}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <div
              style={{
                display: "flex",
                gap: "6px",
                fontSize: "12px",
              }}
            >
              {timeframes.map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "999px",
                    border:
                      timeframe === tf
                        ? "1px solid #38bdf8"
                        : "1px solid #1f2933",
                    background:
                      timeframe === tf ? "#020617" : "transparent",
                    boxShadow:
                      timeframe === tf
                        ? "0 0 0 1px rgba(56, 189, 248, 0.45)"
                        : "none",
                    color: "#e5e7eb",
                    cursor: "pointer",
                  }}
                >
                  {tf}
                </button>
              ))}
            </div>

            <div
              style={{
                display: "flex",
                gap: "6px",
              }}
            >
              <button
                onClick={handleQuickBuyAI}
                disabled={quickLoading}
                style={{
                  borderRadius: "999px",
                  border: "none",
                  padding: "6px 14px",
                  fontSize: "12px",
                  cursor: "pointer",
                  background: "#16a34a",
                  color: "#0b1120",
                  fontWeight: 600,
                  opacity: quickLoading ? 0.6 : 1,
                }}
              >
                🧠 Quick Buy AI
              </button>
              <button
                onClick={handleQuickSellAll}
                disabled={quickLoading}
                style={{
                  borderRadius: "999px",
                  border: "none",
                  padding: "6px 14px",
                  fontSize: "12px",
                  cursor: "pointer",
                  background: "#b91c1c",
                  color: "#f9fafb",
                  fontWeight: 600,
                  opacity: quickLoading ? 0.6 : 1,
                }}
              >
                📸 Quick Sell + Snap
              </button>
            </div>
          </div>
        </div>

        {/* Main chart */}
        <div
          ref={chartRef}
          style={{
            flex: 1,
            minHeight: "320px",
            borderRadius: "16px",
            border: "1px solid #111827",
            background:
              "radial-gradient(circle at top, #020617, #020617 40%, #020617)",
            overflow: "hidden",
          }}
        >
          <Chart
            symbol={symbol}
            timeframe={timeframe}
            autoRefresh={autoRefresh}
          />
        </div>
      </div>

      {/* RIGHT SIDE */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "12px",
          minWidth: 0,
        }}
      >
        {/* Tabs */}
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: "8px",
          }}
        >
          {[
            { id: "insight", label: "AI Insight", icon: "🧠" },
            { id: "alerts", label: "Alerts", icon: "🔔" },
            { id: "theme", label: "Theme", icon: "⚙️" },
          ].map((tab) => {
            const active = rightPanelTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setRightPanelTab(tab.id)}
                style={{
                  padding: "6px 18px",
                  borderRadius: "999px",
                  border: active
                    ? "1px solid #38bdf8"
                    : "1px solid #111827",
                  background: "#020617",
                  color: "#e5e7eb",
                  fontSize: "12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  cursor: "pointer",
                  boxShadow: active
                    ? "0 0 0 1px rgba(56, 189, 248, 0.45)"
                    : "0 0 0 0 transparent",
                }}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        {rightPanelTab === "insight" && (
          <div
            style={{
              borderRadius: "14px",
              border: "1px solid #111827",
              background:
                "radial-gradient(circle at top left, #020617, #020617 40%, #020617)",
              padding: "10px 14px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "6px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: "18px" }}>🧠</span>
                <div>
                  <div
                    style={{
                      fontSize: "13px",
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                      color: "#e5e7eb",
                    }}
                  >
                    AI Insight – {symbol}
                  </div>
                  <div
                    style={{
                      fontSize: "12px",
                      color: "#64748b",
                    }}
                  >
                    Heuristic:{" "}
                    <span style={{ color: signalColor, fontWeight: 600 }}>
                      {signal}
                    </span>
                    {confidence && (
                      <span style={{ color: "#9ca3af" }}>
                        {" "}
                        · conf {confidence}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <button
                  onClick={() => runAIInsight("default")}
                  disabled={aiLoading}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "999px",
                    border: "1px solid #38bdf8",
                    background: "#020617",
                    color: "#e5e7eb",
                    fontSize: "11px",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    cursor: "pointer",
                    opacity: aiLoading ? 0.6 : 1,
                  }}
                  title="Run AI insight scan"
                >
                  <span>🧠</span>
                  <span>{aiLoading ? "Scanning…" : "Scan"}</span>
                </button>

                <div
                  style={{
                    fontSize: "11px",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    color: "#22c55e",
                  }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 999,
                      background: "#22c55e",
                    }}
                  />
                  <span>live</span>
                </div>
              </div>
            </div>

            <div
              style={{
                fontSize: "12px",
                maxHeight: "140px",
                overflowY: "auto",
                paddingRight: "4px",
              }}
            >
              {aiLoading && (
                <div style={{ opacity: 0.7 }}>
                  🧠 AI is thinking about {symbol}…
                </div>
              )}
              {!aiLoading && aiInsight?.summary && (
                <div>{aiInsight.summary}</div>
              )}
              {!aiLoading && !aiInsight?.summary && (
                <div style={{ opacity: 0.6 }}>
                  No hybrid trend yet — hit <strong>Scan</strong> to
                  generate fresh analysis. The same insight will be
                  formatted for Discord.
                </div>
              )}
            </div>
          </div>
        )}

        {rightPanelTab === "alerts" && (
          <div
            style={{
              borderRadius: "14px",
              border: "1px solid #111827",
              background: "#020617",
              padding: "10px 14px",
            }}
          >
            <div
              style={{
                fontSize: "13px",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#e5e7eb",
                marginBottom: "6px",
              }}
            >
              🔔 Price Alerts – {symbol}
            </div>

            <form
              onSubmit={handleCreateAlert}
              style={{
                display: "flex",
                gap: "8px",
                marginBottom: "8px",
              }}
            >
              <select
                value={newAlertDirection}
                onChange={(e) => setNewAlertDirection(e.target.value)}
                style={{
                  borderRadius: "999px",
                  border: "1px solid #1f2933",
                  background: "#020617",
                  color: "#e5e7eb",
                  padding: "6px 10px",
                  fontSize: "12px",
                }}
              >
                <option value="above">📈 Above</option>
                <option value="below">📉 Below</option>
              </select>
              <input
                value={newAlertPrice}
                onChange={(e) => setNewAlertPrice(e.target.value)}
                placeholder="Price"
                type="number"
                step="0.01"
                style={{
                  flex: 1,
                  borderRadius: "999px",
                  border: "1px solid #1f2933",
                  background: "#020617",
                  color: "#e5e7eb",
                  padding: "6px 10px",
                  fontSize: "12px",
                  outline: "none",
                }}
              />
              <button
                type="submit"
                style={{
                  borderRadius: "999px",
                  border: "1px solid #38bdf8",
                  background: "#020617",
                  color: "#e5e7eb",
                  padding: "6px 16px",
                  fontSize: "12px",
                  cursor: "pointer",
                }}
              >
                Add
              </button>
            </form>

            <div
              style={{
                fontSize: "12px",
                maxHeight: "140px",
                overflowY: "auto",
                paddingRight: "4px",
              }}
            >
              {alertsLoading && (
                <div style={{ opacity: 0.7 }}>Loading alerts…</div>
              )}

              {!alertsLoading && alerts.length === 0 && (
                <div style={{ opacity: 0.6 }}>
                  No alerts yet for {symbol}. Create one above and the
                  backend will watch price and ping Discord when it
                  triggers. 🔔
                </div>
              )}

              {alerts.map((a) => (
                <div
                  key={a.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "6px 0",
                    borderBottom: "1px dashed #111827",
                    gap: 8,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                    }}
                  >
                    <span>
                      {a.direction === "above" ? "📈 Above" : "📉 Below"}{" "}
                      {Number(a.price).toFixed(2)}
                    </span>
                    <span
                      style={{
                        fontSize: "11px",
                        color: a.enabled ? "#22c55e" : "#64748b",
                      }}
                    >
                      {a.enabled ? "Active" : "Triggered / Disabled"}
                    </span>
                  </div>
                  <button
                    onClick={() => handleDeleteAlert(a.id)}
                    style={{
                      borderRadius: "999px",
                      border: "none",
                      background: "transparent",
                      color: "#e5e7eb",
                      width: "24px",
                      height: "24px",
                      fontSize: "14px",
                      cursor: "pointer",
                    }}
                    title="Delete alert"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {rightPanelTab === "theme" && (
          <div
            style={{
              borderRadius: "14px",
              border: "1px solid #111827",
              background: "#020617",
              padding: "10px 14px",
            }}
          >
            <div
              style={{
                fontSize: "13px",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#e5e7eb",
                marginBottom: "4px",
              }}
            >
              ⚙️ Theme
            </div>
            <div style={{ fontSize: "12px", color: "#9ca3af" }}>
              Future spot for theme presets and custom colors. For now
              the dashboard stays in your default dark sci-fi look. 🌌
            </div>
          </div>
        )}

        {/* Watchlist */}
        <div
          style={{
            borderRadius: "14px",
            border: "1px solid #111827",
            background: "#020617",
            padding: "10px 14px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            <div>
              <div
                style={{
                  fontSize: "13px",
                  textTransform: "uppercase",
                  letterSpacing: "0.08em",
                  color: "#e5e7eb",
                }}
              >
                Watchlist
              </div>
              <div
                style={{
                  fontSize: "12px",
                  color: "#64748b",
                  marginTop: "2px",
                }}
              >
                Live price • % change • mini trend
              </div>
            </div>
            <div
              style={{
                fontSize: "11px",
                color: "#9ca3af",
              }}
            >
              {watchlistUpdating ? "updating…" : "live"}
            </div>
          </div>

          <form
            onSubmit={handleAddWatchlist}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "10px",
            }}
          >
            <input
              value={newWatchSymbol}
              onChange={(e) =>
                setNewWatchSymbol(e.target.value.toUpperCase())
              }
              placeholder="Add symbol (e.g. NVDA)"
              style={{
                flex: 1,
                borderRadius: "999px",
                border: "1px solid #1f2933",
                background: "#020617",
                color: "#e5e7eb",
                padding: "6px 12px",
                fontSize: "12px",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={watchlistUpdating}
              style={{
                width: "30px",
                height: "30px",
                borderRadius: "999px",
                border: "1px solid #38bdf8",
                background: "#020617",
                color: "#e5e7eb",
                fontSize: "18px",
                lineHeight: "18px",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 0 0 1px rgba(56, 189, 248, 0.45)",
                opacity: watchlistUpdating ? 0.6 : 1,
              }}
            >
              +
            </button>
          </form>

          <div
            style={{
              maxHeight: "180px",
              overflowY: "auto",
              paddingRight: "4px",
              fontSize: "12px",
            }}
          >
            {watchlist.length === 0 && (
              <div style={{ opacity: 0.6 }}>
                Nothing here yet. Add a symbol above to track price,
                % change and trend.
              </div>
            )}

            {watchlist.length > 0 && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1.3fr 1.3fr 1fr 1.6fr 32px",
                  padding: "0 4px 4px",
                  marginBottom: "4px",
                  fontSize: "11px",
                  color: "#9ca3af",
                }}
              >
                <div>Symbol</div>
                <div>Price</div>
                <div>%</div>
                <div>Trend</div>
                <div />
              </div>
            )}

            {watchlist.map((item) => {
              const pct = item.change_percent ?? item.change_pct;
              let pctColor = "#e5e7eb";
              let pctPrefix = "";
              let trendEmoji = "➖";

              if (typeof pct === "number") {
                if (pct > 0) {
                  pctColor = "#22c55e";
                  pctPrefix = "+";
                  trendEmoji = "📈";
                } else if (pct < 0) {
                  pctColor = "#ef4444";
                  trendEmoji = "📉";
                }
              }

              const priceValue =
                item.price != null
                  ? item.price
                  : item.last_price != null
                  ? item.last_price
                  : null;

              return (
                <div
                  key={item.symbol}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1.3fr 1.3fr 1fr 1.6fr 32px",
                    alignItems: "center",
                    padding: "6px 4px",
                    borderRadius: "10px",
                    background:
                      "linear-gradient(135deg, #020617, #020617)",
                    border: "1px solid #111827",
                    marginBottom: "4px",
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600 }}>
                      {item.symbol}
                    </div>
                    <div
                      style={{
                        fontSize: "11px",
                        opacity: 0.7,
                      }}
                    >
                      {item.name || ""}
                    </div>
                  </div>
                  <div>
                    {priceValue != null ? (
                      <span>${Number(priceValue).toFixed(2)}</span>
                    ) : (
                      <span>—</span>
                    )}
                  </div>
                  <div style={{ color: pctColor }}>
                    {typeof pct === "number"
                      ? `${pctPrefix}${pct.toFixed(2)}%`
                      : "—"}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    <span>{trendEmoji}</span>
                    <span
                      style={{
                        fontSize: "11px",
                        opacity: 0.7,
                      }}
                    >
                      spark
                    </span>
                  </div>
                  <button
                    onClick={() => handleDeleteWatchItem(item)}
                    style={{
                      borderRadius: "999px",
                      border: "none",
                      background: "transparent",
                      color: "#e5e7eb",
                      width: "24px",
                      height: "24px",
                      fontSize: "14px",
                      cursor: "pointer",
                    }}
                    title="Remove from watchlist"
                  >
                    🗑️
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Notifications */}
        <div
          style={{
            borderRadius: "14px",
            border: "1px solid #111827",
            background: "#020617",
            padding: "10px 14px",
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minHeight: "140px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "6px",
            }}
          >
            <div
              style={{
                fontSize: "13px",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#e5e7eb",
              }}
            >
              Notifications
            </div>
            <div
              style={{
                width: 10,
                height: 18,
                borderRadius: 999,
                background: "#0ea5e9",
              }}
              title="Live feed"
            />
          </div>

          <div
            style={{
              fontSize: "12px",
              flex: 1,
              overflowY: "auto",
              paddingRight: "4px",
            }}
          >
            {notifications.length === 0 && (
              <div style={{ opacity: 0.6 }}>
                🧠 Live AI alerts, executions, and Discord messages will
                appear here.
              </div>
            )}

            {notifications.map((n) => (
              <div
                key={n.id}
                style={{
                  marginBottom: "4px",
                  padding: "4px 0",
                  borderBottom: "1px dashed #111827",
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 6,
                }}
              >
                <span>🧠</span>
                <span>{n.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
