// frontend/src/components/PriceChart.jsx
import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  Area,
  Line,
} from "recharts";
import { fetchBars } from "../api/aiTrend";

const PriceChart = ({ symbol, useCrypto }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // load bars and refresh periodically
  useEffect(() => {
    let cancelled = false;
    let intervalId;

    const load = async () => {
      if (!symbol) return;
      setLoading(true);
      setError(null);
      try {
        const bars = await fetchBars(
          symbol,
          useCrypto ? "1m" : "1Min",
          useCrypto,
          200
        );
        if (cancelled) return;

        const mapped = bars.map((b, idx) => ({
          idx,
          t: b.t,
          close: Number(b.c),
          high: Number(b.h),
          low: Number(b.l),
        }));
        setData(mapped);
      } catch (e) {
        if (!cancelled) {
          console.error("PriceChart load error", e);
          setError(e.message || "Failed to load bars");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    intervalId = setInterval(load, 10_000); // refresh every 10s

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [symbol, useCrypto]);

  return (
    <div style={{ width: "100%", height: "100%" }}>
      {error && (
        <div
          style={{
            fontSize: "11px",
            color: "#fca5a5",
            marginBottom: "4px",
          }}
        >
          {error}
        </div>
      )}
      {loading && (
        <div
          style={{
            fontSize: "11px",
            opacity: 0.7,
            marginBottom: "4px",
          }}
        >
          Loading chart...
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <XAxis
            dataKey="idx"
            hide={true}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            stroke="#4b5563"
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "#020617",
              border: "1px solid #4b5563",
              fontSize: 11,
            }}
            labelFormatter={(idx) => `Index ${idx}`}
            formatter={(val, name) =>
              [val.toFixed(2), name === "close" ? "Close" : name]
            }
          />
          {/* Price area */}
          <Area
            type="monotone"
            dataKey="close"
            stroke="#22c55e"
            fill="rgba(34,197,94,0.12)"
          />
          {/* Outline line */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#22c55e"
            strokeWidth={1.4}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
