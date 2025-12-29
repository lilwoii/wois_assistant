// frontend/src/components/Settings.jsx
import React, { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";

const SETTINGS_LOCAL_THEME_KEY = "woi_theme_v1";

const THEME_PRESETS = [
  {
    id: "neo_dark",
    name: "Neo Dark",
    emoji: "🛰️",
    accent: "#6366f1",
    bg: "radial-gradient(circle at top, #020617, #020617 50%, #020617)",
  },
  {
    id: "cyber_grid",
    name: "Cyber Grid",
    emoji: "🕸️",
    accent: "#22c55e",
    bg: "radial-gradient(circle at top, #020617, #020617 40%, #020617)",
  },
  {
    id: "terminal",
    name: "Terminal",
    emoji: "🧮",
    accent: "#22c55e",
    bg: "linear-gradient(135deg, #020617, #020617)",
  },
  {
    id: "sunset",
    name: "Sunset Fade",
    emoji: "🌅",
    accent: "#f97316",
    bg: "linear-gradient(135deg, #020617, #1e293b)",
  },
];

function applyThemeToDocument(theme) {
  if (!theme) return;
  const root = document.documentElement;
  root.style.setProperty("--woi-accent", theme.accent || "#6366f1");
  root.style.setProperty("--woi-bg", theme.bg || "#020617");
  document.body.style.backgroundImage = theme.bg || "";
  document.body.style.backgroundColor = "#020617";
}

const Settings = () => {
  const [activeTab, setActiveTab] = useState("general"); // general | risk | theme
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  // core backend settings
  const [settings, setSettings] = useState({
    discord_webhook: "",
    use_paper: true,
    use_crypto: false,
    signal_schedule_min: 0,
    risk_sl_pct: 1.5,
    risk_tp_pct: 3.0,
    risk_per_trade_pct: 1.0,
    size_mode: "risk_pct",
    horizon_min_days: 5,
    horizon_max_days: 30,
    allow_long_horizon: true,
  });

  // theme
  const [themePresetId, setThemePresetId] = useState("neo_dark");
  const [themeAccent, setThemeAccent] = useState("#6366f1");
  const [themeDirty, setThemeDirty] = useState(false);

  // --------- load settings + theme ----------

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/settings`);
        const data = await res.json().catch(() => null);
        if (!res.ok || !data) {
          setError("Failed to load settings from backend.");
        } else {
          setSettings((prev) => ({
            ...prev,
            ...data,
          }));
        }
      } catch (err) {
        console.error("Settings load error", err);
        setError("Failed to load settings (network/backend).");
      } finally {
        setLoading(false);
      }

      // theme from localStorage
      try {
        const raw = localStorage.getItem(SETTINGS_LOCAL_THEME_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          setThemePresetId(parsed.presetId || "neo_dark");
          setThemeAccent(parsed.accent || "#6366f1");
          applyThemeToDocument(parsed);
        } else {
          const preset = THEME_PRESETS[0];
          setThemePresetId(preset.id);
          setThemeAccent(preset.accent);
          applyThemeToDocument(preset);
        }
      } catch {
        const preset = THEME_PRESETS[0];
        setThemePresetId(preset.id);
        setThemeAccent(preset.accent);
        applyThemeToDocument(preset);
      }
    };

    load();
  }, []);

  // --------- backend save ----------

  const handleSaveBackendSettings = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload = {
        discord_webhook: settings.discord_webhook || null,
        use_paper: !!settings.use_paper,
        use_crypto: !!settings.use_crypto,
        signal_schedule_min: Number(settings.signal_schedule_min) || 0,
        risk_sl_pct: Number(settings.risk_sl_pct) || 1.5,
        risk_tp_pct: Number(settings.risk_tp_pct) || 3.0,
        risk_per_trade_pct: Number(settings.risk_per_trade_pct) || 1.0,
        size_mode: settings.size_mode || "risk_pct",
        horizon_min_days: Number(settings.horizon_min_days) || 5,
        horizon_max_days: Number(settings.horizon_max_days) || 30,
        allow_long_horizon: !!settings.allow_long_horizon,
      };

      const res = await fetch(`${API_BASE_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok || !data) {
        setError("Failed to save settings – backend rejected the payload.");
      } else {
        setSettings((prev) => ({
          ...prev,
          ...data,
        }));
      }
    } catch (err) {
      console.error("Settings save error", err);
      setError("Failed to save settings (network/backend).");
    } finally {
      setSaving(false);
    }
  };

  // --------- theme save (local only) ----------

  const handleApplyTheme = () => {
    const preset = THEME_PRESETS.find((p) => p.id === themePresetId);
    const theme = {
      presetId: themePresetId,
      accent: themeAccent || (preset ? preset.accent : "#6366f1"),
      bg: preset ? preset.bg : "#020617",
    };
    localStorage.setItem(SETTINGS_LOCAL_THEME_KEY, JSON.stringify(theme));
    applyThemeToDocument(theme);
    setThemeDirty(false);
  };

  const handleSelectPreset = (id) => {
    setThemePresetId(id);
    const preset = THEME_PRESETS.find((p) => p.id === id);
    if (preset) {
      setThemeAccent(preset.accent);
    }
    setThemeDirty(true);
  };

  // --------- UI helpers ----------

  const updateSetting = (key, value) => {
    setSettings((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const tabButtonStyle = (id) => ({
    padding: "6px 10px",
    borderRadius: "999px",
    border: id === activeTab ? "1px solid #6366f1" : "1px solid #1f2937",
    background: id === activeTab ? "#111827" : "transparent",
    color: "#e5e7eb",
    fontSize: "12px",
    cursor: "pointer",
  });

  // --------- tab content ----------

  const renderGeneralTab = () => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        fontSize: "12px",
      }}
    >
      <div>
        <div style={{ marginBottom: "4px", color: "#9ca3af" }}>
          Discord Webhook URL
        </div>
        <input
          value={settings.discord_webhook || ""}
          onChange={(e) =>
            updateSetting("discord_webhook", e.target.value)
          }
          placeholder="https://discord.com/api/webhooks/…"
          style={{
            width: "100%",
            borderRadius: "8px",
            border: "1px solid #1f2937",
            background: "#020617",
            color: "#e5e7eb",
            padding: "8px 10px",
            fontSize: "12px",
            outline: "none",
          }}
        />
        <div
          style={{
            marginTop: "4px",
            color: "#64748b",
            fontSize: "11px",
          }}
        >
          💬 All AI alerts, snapshots, and scheduled signals will post
          here.
        </div>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "10px",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            borderRadius: "10px",
            border: "1px solid #1f2937",
            padding: "8px 10px",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={!!settings.use_paper}
            onChange={(e) =>
              updateSetting("use_paper", e.target.checked)
            }
          />
          <span>Use paper trading (Alpaca)</span>
        </label>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            borderRadius: "10px",
            border: "1px solid #1f2937",
            padding: "8px 10px",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={!!settings.use_crypto}
            onChange={(e) =>
              updateSetting("use_crypto", e.target.checked)
            }
          />
          <span>Enable crypto (Binance)</span>
        </label>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <div>
          <div style={{ color: "#9ca3af", marginBottom: "4px" }}>
            Scheduled AI scan (minutes)
          </div>
          <input
            type="number"
            value={settings.signal_schedule_min ?? 0}
            onChange={(e) =>
              updateSetting(
                "signal_schedule_min",
                Number(e.target.value || 0)
              )
            }
            style={{
              width: "120px",
              borderRadius: "8px",
              border: "1px solid #1f2937",
              background: "#020617",
              color: "#e5e7eb",
              padding: "6px 8px",
              fontSize: "12px",
              outline: "none",
            }}
          />
          <div
            style={{
              marginTop: "4px",
              color: "#64748b",
              fontSize: "11px",
            }}
          >
            ⏱️ 0 = off. Otherwise, scans watchlist symbols on a loop
            and posts to Discord.
          </div>
        </div>
      </div>
    </div>
  );

  const renderRiskTab = () => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        fontSize: "12px",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "10px",
        }}
      >
        <div>
          <div style={{ color: "#9ca3af", marginBottom: "4px" }}>
            Risk per trade (% equity)
          </div>
          <input
            type="number"
            value={settings.risk_per_trade_pct ?? 1.0}
            onChange={(e) =>
              updateSetting(
                "risk_per_trade_pct",
                Number(e.target.value || 0)
              )
            }
            style={{
              width: "100%",
              borderRadius: "8px",
              border: "1px solid #1f2937",
              background: "#020617",
              color: "#e5e7eb",
              padding: "6px 8px",
              fontSize: "12px",
              outline: "none",
            }}
          />
        </div>

        <div>
          <div style={{ color: "#9ca3af", marginBottom: "4px" }}>
            S/L distance (%)
          </div>
          <input
            type="number"
            value={settings.risk_sl_pct ?? 1.5}
            onChange={(e) =>
              updateSetting(
                "risk_sl_pct",
                Number(e.target.value || 0)
              )
            }
            style={{
              width: "100%",
              borderRadius: "8px",
              border: "1px solid #1f2937",
              background: "#020617",
              color: "#e5e7eb",
              padding: "6px 8px",
              fontSize: "12px",
              outline: "none",
            }}
          />
        </div>

        <div>
          <div style={{ color: "#9ca3af", marginBottom: "4px" }}>
            T/P distance (%)
          </div>
          <input
            type="number"
            value={settings.risk_tp_pct ?? 3.0}
            onChange={(e) =>
              updateSetting(
                "risk_tp_pct",
                Number(e.target.value || 0)
              )
            }
            style={{
              width: "100%",
              borderRadius: "8px",
              border: "1px solid #1f2937",
              background: "#020617",
              color: "#e5e7eb",
              padding: "6px 8px",
              fontSize: "12px",
              outline: "none",
            }}
          />
        </div>

        <div>
          <div style={{ color: "#9ca3af", marginBottom: "4px" }}>
            Horizon window (days)
          </div>
          <div
            style={{
              display: "flex",
              gap: "6px",
              alignItems: "center",
            }}
          >
            <input
              type="number"
              value={settings.horizon_min_days ?? 5}
              onChange={(e) =>
                updateSetting(
                  "horizon_min_days",
                  Number(e.target.value || 0)
                )
              }
              style={{
                flex: 1,
                borderRadius: "8px",
                border: "1px solid #1f2937",
                background: "#020617",
                color: "#e5e7eb",
                padding: "6px 8px",
                fontSize: "12px",
                outline: "none",
              }}
            />
            <span>→</span>
            <input
              type="number"
              value={settings.horizon_max_days ?? 30}
              onChange={(e) =>
                updateSetting(
                  "horizon_max_days",
                  Number(e.target.value || 0)
                )
              }
              style={{
                flex: 1,
                borderRadius: "8px",
                border: "1px solid #1f2937",
                background: "#020617",
                color: "#e5e7eb",
                padding: "6px 8px",
                fontSize: "12px",
                outline: "none",
              }}
            />
          </div>
          <div
            style={{
              marginTop: "4px",
              color: "#64748b",
              fontSize: "11px",
            }}
          >
            📆 The AI&apos;s ETA label (weekly / monthly / long) uses
            this band.
          </div>
        </div>
      </div>

      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          borderRadius: "10px",
          border: "1px solid #1f2937",
          padding: "8px 10px",
          cursor: "pointer",
        }}
      >
        <input
          type="checkbox"
          checked={!!settings.allow_long_horizon}
          onChange={(e) =>
            updateSetting("allow_long_horizon", e.target.checked)
          }
        />
        <span>Allow long horizon (> 30 trading days)</span>
      </label>
    </div>
  );

  const renderThemeTab = () => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        fontSize: "12px",
      }}
    >
      <div>
        <div
          style={{
            fontSize: "13px",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            color: "#94a3b8",
            marginBottom: "2px",
          }}
        >
          Theme & Accent
        </div>
        <div style={{ color: "#64748b" }}>
          Choose a preset, tweak the accent color, and save. This is
          stored locally in your browser (no API breakage).
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
          gap: "10px",
        }}
      >
        {THEME_PRESETS.map((p) => (
          <button
            key={p.id}
            onClick={() => handleSelectPreset(p.id)}
            style={{
              textAlign: "left",
              borderRadius: "12px",
              border:
                themePresetId === p.id
                  ? "1px solid #6366f1"
                  : "1px solid #1f2937",
              background: "#020617",
              padding: "8px 10px",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              gap: "4px",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span>{p.emoji}</span>
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: 500,
                  color: "#e5e7eb",
                }}
              >
                {p.name}
              </span>
            </div>
            <div
              style={{
                height: "20px",
                borderRadius: "999px",
                border: "1px solid #111827",
                backgroundImage: p.bg,
              }}
            />
          </button>
        ))}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
        }}
      >
        <div>
          <div style={{ color: "#9ca3af", marginBottom: "4px" }}>
            Accent color
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <input
              type="color"
              value={themeAccent}
              onChange={(e) => {
                setThemeAccent(e.target.value);
                setThemeDirty(true);
              }}
              style={{
                width: "36px",
                height: "24px",
                borderRadius: "8px",
                border: "1px solid #1f2937",
                padding: 0,
                background: "transparent",
                cursor: "pointer",
              }}
            />
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            >
              {themeAccent}
            </span>
          </div>
        </div>

        <div
          style={{
            flex: 1,
            fontSize: "11px",
            color: "#64748b",
          }}
        >
          🎨 Accent is exposed as{" "}
          <code style={{ color: "#e5e7eb" }}>--woi-accent</code> on{" "}
          <code style={{ color: "#e5e7eb" }}>:root</code>. We can wire
          more components to it later (Accent Everywhere bundle).
        </div>
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          gap: "8px",
          marginTop: "4px",
        }}
      >
        <button
          onClick={handleApplyTheme}
          disabled={!themeDirty}
          style={{
            borderRadius: "999px",
            border: "none",
            background: themeDirty
              ? "linear-gradient(135deg, #22c55e, #0ea5e9)"
              : "#111827",
            color: themeDirty ? "#0b1120" : "#9ca3af",
            padding: "6px 14px",
            fontSize: "12px",
            fontWeight: 600,
            cursor: themeDirty ? "pointer" : "default",
          }}
        >
          {themeDirty ? "Apply & save theme" : "Theme is up to date"}
        </button>
      </div>
    </div>
  );

  // --------- render ----------

  return (
    <div
      style={{
        padding: "16px",
        background: "#020617",
        color: "#e5e7eb",
        height: "100%",
        overflowY: "auto",
      }}
    >
      <div
        style={{
          marginBottom: "12px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "14px",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "#94a3b8",
            }}
          >
            Settings
          </div>
          <div
            style={{
              fontSize: "12px",
              color: "#64748b",
            }}
          >
            Wire your Discord, risk engine, and theme preferences. 🧩
          </div>
        </div>
        <div
          style={{
            borderRadius: "999px",
            border: "1px solid #1f2937",
            padding: "6px 12px",
            fontSize: "11px",
            color: "#9ca3af",
          }}
        >
          {loading ? "Loading…" : "Ready"}
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: "8px",
          marginBottom: "12px",
        }}
      >
        <button
          style={tabButtonStyle("general")}
          onClick={() => setActiveTab("general")}
        >
          ⚙️ General
        </button>
        <button
          style={tabButtonStyle("risk")}
          onClick={() => setActiveTab("risk")}
        >
          🛡️ Risk
        </button>
        <button
          style={tabButtonStyle("theme")}
          onClick={() => setActiveTab("theme")}
        >
          🎨 Theme
        </button>
      </div>

      <div
        style={{
          borderRadius: "16px",
          border: "1px solid #111827",
          background:
            "radial-gradient(circle at top, #020617, #020617 40%, #020617)",
          padding: "12px 14px",
        }}
      >
        {activeTab === "general" && renderGeneralTab()}
        {activeTab === "risk" && renderRiskTab()}
        {activeTab === "theme" && renderThemeTab()}
      </div>

      {error && (
        <div
          style={{
            marginTop: "10px",
            padding: "8px 10px",
            borderRadius: "8px",
            background: "#111827",
            color: "#fecaca",
            fontSize: "12px",
          }}
        >
          ❌ {error}
        </div>
      )}

      {activeTab !== "theme" && (
        <div
          style={{
            marginTop: "10px",
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
          <button
            onClick={handleSaveBackendSettings}
            disabled={saving}
            style={{
              borderRadius: "999px",
              border: "none",
              background:
                "linear-gradient(135deg, #22c55e, #0ea5e9, #6366f1)",
              color: "#0b1120",
              padding: "8px 16px",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              opacity: saving ? 0.7 : 1,
            }}
          >
            {saving ? "Saving…" : "Save settings"}
          </button>
        </div>
      )}
    </div>
  );
};

export default Settings;
