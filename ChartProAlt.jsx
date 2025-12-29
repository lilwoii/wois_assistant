import React, { useEffect, useRef } from "react";
import * as LightweightCharts from "lightweight-charts";

const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export default function ChartProAlt({
  symbol,
  overlay,
  overlaysExpert,
  useCrypto,
  onChartClick,
}) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const candleRef = useRef(null);
  const channelTopRef = useRef(null);
  const channelBottomRef = useRef(null);
  const fibSeriesRef = useRef([]);

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = LightweightCharts.createChart(containerRef.current, {
      layout: {
        background: { type: "solid", color: "#020617" },
        textColor: "#e5e7eb",
      },
      grid: {
        vertLines: { color: "#0f172a" },
        horzLines: { color: "#0f172a" },
      },
      rightPriceScale: {
        borderColor: "#1f2937",
      },
      timeScale: {
        borderColor: "#1f2937",
      },
      crosshair: {
        mode: 1,
      },
    });

    if (typeof chart.addCandlestickSeries !== "function") {
      console.error(
        "LightweightCharts.createChart did not return a full chart object. Got:",
        chart
      );
      return;
    }

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      borderVisible: false,
    });

    const channelTop = chart.addLineSeries({
      color: "#38bdf8",
      lineWidth: 1,
      priceLineVisible: false,
    });

    const channelBottom = chart.addLineSeries({
      color: "#38bdf8",
      lineWidth: 1,
      priceLineVisible: false,
    });

    const fibColors = ["#4b5563", "#64748b", "#6b21a8", "#a855f7", "#22c55e"];
    const fibSeries = fibColors.map((c) =>
      chart.addLineSeries({
        color: c,
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
      })
    );

    candleRef.current = candleSeries;
    channelTopRef.current = channelTop;
    channelBottomRef.current = channelBottom;
    fibSeriesRef.current = fibSeries;
    chartRef.current = chart;

    const handleResize = () => {
      if (!containerRef.current || !chartRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      chartRef.current.applyOptions({
        width: rect.width,
        height: rect.height,
      });
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    // Click → order hook
    chart.subscribeClick((param) => {
      if (!onChartClick) return;
      if (!param) return;
      const price = param.price;
      const time = param.time;
      if (price == null || time == null) return;
      onChartClick({ price, time });
    });

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [onChartClick]);

  // Load bars whenever symbol / crypto changes, compute overlays
  useEffect(() => {
    if (!candleRef.current || !symbol) return;

    const controller = new AbortController();

    (async () => {
      try {
        const tf = "1Min";
        const cryptoFlag = useCrypto ? 1 : 0;
        const url = `${API}/proxy/bars?symbol=${encodeURIComponent(
          symbol
        )}&tf=${tf}&crypto=${cryptoFlag}&limit=500`;
        const res = await fetch(url, { signal: controller.signal });

        if (!res.ok) {
          console.error("proxy/bars failed", res.status);
          return;
        }

        const data = await res.json();
        if (!Array.isArray(data) || data.length === 0) return;

        const candles = data.map((b) => ({
          time: b.t,
          open: b.o,
          high: b.h,
          low: b.l,
          close: b.c,
        }));

        candleRef.current.setData(candles);

        // === AI-drawn trend "channel" (simple version) ===
        const highs = candles.map((c) => c.high);
        const lows = candles.map((c) => c.low);
        const maxHigh = Math.max(...highs);
        const minLow = Math.max(0, Math.min(...lows));
        const firstTime = candles[0].time;
        const lastTime = candles[candles.length - 1].time;

        if (channelTopRef.current && channelBottomRef.current) {
          channelTopRef.current.setData([
            { time: firstTime, value: maxHigh },
            { time: lastTime, value: maxHigh },
          ]);
          channelBottomRef.current.setData([
            { time: firstTime, value: minLow },
            { time: lastTime, value: minLow },
          ]);
        }

        // === Auto Fibonacci per swing (high-low range over current view) ===
        const swingLow = minLow;
        const swingHigh = maxHigh;
        const diff = swingHigh - swingLow;
        const fibLevels = [0, 0.382, 0.5, 0.618, 1];

        fibLevels.forEach((lvl, idx) => {
          const price = swingLow + diff * lvl;
          const series = fibSeriesRef.current[idx];
          if (!series) return;
          series.setData([
            { time: firstTime, value: price },
            { time: lastTime, value: price },
          ]);
        });
      } catch (e) {
        if (e.name !== "AbortError") {
          console.error("fetch bars error:", e);
        }
      }
    })();

    return () => controller.abort();
  }, [symbol, useCrypto]);

  return <div ref={containerRef} className="chart-inner" />;
}
