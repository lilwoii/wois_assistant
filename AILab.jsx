// frontend/src/components/AILab.jsx
import React, { useState, useEffect, useMemo, useRef } from "react";
import { API_BASE_URL } from "../config";

const AILab = () => {
  // -------------------------
  // One-shot training inputs
  // -------------------------
  const [symbolsInput, setSymbolsInput] = useState("TSLA, NVDA, SPY");
  const [epochs, setEpochs] = useState(5);
  const [timeframe, setTimeframe] = useState("1D");
  const [modelSize, setModelSize] = useState("light"); // "light" | "heavy"
  const [computeTier, setComputeTier] = useState("local"); // "local" | "runpod_24" | "runpod_80"

  // Endless knobs
  const [epochsPerCycle, setEpochsPerCycle] = useState(3);
  const [cooldownSec, setCooldownSec] = useState(2);

  // One-shot training UI
  const [isTraining, setIsTraining] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [lastResponse, setLastResponse] = useState(null);
  const [lastError, setLastError] = useState(null);

  // Endless training job UI
  const [activeJobId, setActiveJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [jobError, setJobError] = useState(null);
  const pollTimerRef = useRef(null);

  // Training history
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  // -------------------------
  // Universe Scanner (A–C blocks)
  // -------------------------
  const [scanJobId, setScanJobId] = useState(null);
  const [scanJob, setScanJob] = useState(null);
  const [scanPolling, setScanPolling] = useState(false);
  const [scanError, setScanError] = useState(null);

  const parseSymbols = () =>
    symbolsInput
      .split(",")
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s.length > 0);

  const formatDateTime = (iso) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleString();
    } catch {
      return iso;
    }
  };

  // -------------------------
  // Training history fetch
  // -------------------------
  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE_URL}/ai/training/history?limit=15`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to load training history");
      setHistory(data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // -------------------------
  // Endless training polling
  // -------------------------
  const fetchJob = async (jobId) => {
    if (!jobId) return;
    try {
      const res = await fetch(`${API_BASE_URL}/ai/train/job/${jobId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to fetch job");
      setJob(data);
      setJobError(null);
    } catch (e) {
      setJobError(e.message);
    }
  };

  useEffect(() => {
    // cleanup old timer
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    if (!activeJobId) {
      setJob(null);
      setJobError(null);
      return;
    }

    // initial fetch
    fetchJob(activeJobId);

    // poll every 2s
    pollTimerRef.current = setInterval(() => {
      fetchJob(activeJobId);
    }, 2000);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [activeJobId]);

  // -------------------------
  // One-shot training
  // -------------------------
  const handleStartTraining = async () => {
    const symbols = parseSymbols();
    if (!symbols.length) {
      setStatusMessage("Please enter at least one symbol.");
      return;
    }
    if (!epochs || Number(epochs) <= 0) {
      setStatusMessage("Epochs must be at least 1.");
      return;
    }

    setIsTraining(true);
    setLastError(null);
    setStatusMessage("Starting training job...");
    setLastResponse(null);

    try {
      const res = await fetch(`${API_BASE_URL}/ai/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbols,
          epochs: Number(epochs),
          timeframe,
          model_size: modelSize,
          compute_tier: computeTier,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Training request failed");

      setLastResponse(data);

      if (data.status === "submitted") {
        setStatusMessage(
          `Training submitted to ${
            data.mode === "runpod_80"
              ? "RunPod 80GB"
              : data.mode === "runpod_24"
              ? "RunPod 24GB"
              : "remote endpoint"
          } (job: ${data.job_id || "n/a"}).`
        );
      } else if (data.status === "completed") {
        setStatusMessage(`Local training completed (job: ${data.job_id || "n/a"}).`);
      } else {
        setStatusMessage("Training request completed.");
      }

      fetchHistory();
    } catch (err) {
      console.error(err);
      setLastError(err.message);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  // -------------------------
  // Endless training controls
  // -------------------------
  const handleStartEndless = async () => {
    const symbols = parseSymbols();
    if (!symbols.length) {
      setStatusMessage("Please enter at least one symbol.");
      return;
    }
    if (!epochsPerCycle || Number(epochsPerCycle) <= 0) {
      setStatusMessage("Epochs per cycle must be at least 1.");
      return;
    }

    setStatusMessage("Starting endless training...");
    setJobError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/ai/train/endless/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbols,
          timeframe,
          model_size: modelSize,
          compute_tier: computeTier,
          epochs_per_cycle: Number(epochsPerCycle),
          cooldown_sec: Number(cooldownSec),
          send_to_discord: true,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to start endless training");

      setActiveJobId(data.job_id);
      setStatusMessage(`Endless training started (job: ${data.job_id}).`);
      fetchHistory();
    } catch (e) {
      setStatusMessage(`Error: ${e.message}`);
      setJobError(e.message);
    }
  };

  const handleStopEndless = async () => {
    if (!activeJobId) return;

    setStatusMessage("Stopping endless training...");
    try {
      const res = await fetch(`${API_BASE_URL}/ai/train/endless/stop/${activeJobId}`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to stop job");

      setStatusMessage(`Endless training stopped (job: ${activeJobId}).`);
      fetchJob(activeJobId);
      fetchHistory();
    } catch (e) {
      setStatusMessage(`Error: ${e.message}`);
      setJobError(e.message);
    }
  };

  // -------------------------
  // Universe Scanner (A–C blocks)
  // -------------------------
  const pollScanner = async () => {
    if (!scanJobId) return;
    try {
      const res = await fetch(`${API_BASE_URL}/scanner/job/${scanJobId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to fetch scanner job");
      setScanJob(data);
      setScanError(null);
    } catch (e) {
      setScanError(e.message);
    }
  };

  // ✅ Fixed: stable polling + no “start disabled forever” bug
  useEffect(() => {
    if (!scanPolling || !scanJobId) return;

    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      await pollScanner();
    };

    tick();
    const t = setInterval(tick, 2000); // UI poll fast; scanner runs every ~5m

    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [scanPolling, scanJobId]);

  // ✅ Stop polling automatically once scanner is stopped
  useEffect(() => {
    if (scanJob?.status === "stopped" && scanPolling) {
      setScanPolling(false);
    }
  }, [scanJob?.status, scanPolling]);

  const startScanner = async () => {
    setScanError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/scanner/endless/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scan_stocks: true,
          scan_crypto: true,
          // You can tune these later; these are sane defaults
          timeframes: ["15m", "1h", "4h", "1d"],
          top_n: 10,
          interval_sec: 300, // ✅ best default
          send_to_discord: true,
          auto_add_top: false, // ✅ you asked for OFF
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to start scanner");

      setScanJobId(data.job_id);
      setScanPolling(true);
      setStatusMessage(`Universe scanner started (job: ${data.job_id}).`);
      // optionally pull immediately
      setScanJob(null);
      pollScanner();
    } catch (e) {
      setScanError(e.message);
      setStatusMessage(`Error: ${e.message}`);
    }
  };

  const stopScanner = async () => {
    if (!scanJobId) return;
    setScanError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/scanner/endless/stop/${scanJobId}`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to stop scanner");

      setScanPolling(false);
      setStatusMessage(`Universe scanner stopped (job: ${scanJobId}).`);
      pollScanner();
    } catch (e) {
      setScanError(e.message);
      setStatusMessage(`Error: ${e.message}`);
    }
  };

  const scannerRunning = useMemo(() => {
    // only treat these as running; avoids “start disabled forever” when scanJob is null/undefined
    return ["queued", "scanning"].includes(scanJob?.status);
  }, [scanJob?.status]);

  // -------------------------
  // Status pill + cosmetic bar
  // -------------------------
  const statusInfo = useMemo(() => {
    // priority: endless job state
    const st = job?.status;

    if (jobError) return { label: "Error", emoji: "🔴", color: "text-red-300", running: false };
    if (st === "queued") return { label: "Queued", emoji: "🟦", color: "text-sky-300", running: true };
    if (st === "training") return { label: "Training", emoji: "🟢", color: "text-emerald-300", running: true };
    if (st === "stopping") return { label: "Stopping", emoji: "🟡", color: "text-yellow-300", running: true };
    if (st === "stopped") return { label: "Stopped", emoji: "🔴", color: "text-red-300", running: false };
    if (st === "completed") return { label: "Completed", emoji: "✅", color: "text-emerald-300", running: false };
    if (st === "failed") return { label: "Failed", emoji: "❌", color: "text-red-300", running: false };

    // fallback to one-shot in-flight
    if (lastError) return { label: "Error", emoji: "🔴", color: "text-red-300", running: false };
    if (isTraining) return { label: "Training", emoji: "🟢", color: "text-emerald-300", running: true };
    if (lastResponse?.status === "submitted") return { label: "Queued", emoji: "🟦", color: "text-sky-300", running: true };

    return { label: "Idle", emoji: "⚪", color: "text-gray-400", running: false };
  }, [job, jobError, lastError, isTraining, lastResponse]);

  return (
    <div className="h-full w-full flex flex-col gap-6 p-6 overflow-y-auto">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">AI Lab</h1>
        <p className="text-sm text-gray-400">
          Configure training runs, choose model size, and route jobs to local hardware or your RunPod GPUs.
        </p>
      </div>

      {/* Top row: Model + Compute */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model size card */}
        <div className="bg-[#111827] border border-gray-800 rounded-2xl p-4 flex flex-col gap-3">
          <h2 className="text-lg font-semibold">Model Complexity</h2>
          <p className="text-xs text-gray-400">
            Light is quick; heavy is deeper and more expressive. Both can run locally or on RunPod.
          </p>
          <div className="flex gap-3 mt-2">
            <button
              type="button"
              className={`flex-1 px-3 py-2 rounded-xl text-sm border ${
                modelSize === "light"
                  ? "bg-emerald-500/10 border-emerald-400 text-emerald-200"
                  : "bg-black/30 border-gray-700 text-gray-300"
              }`}
              onClick={() => setModelSize("light")}
            >
              ⚡ Light
            </button>
            <button
              type="button"
              className={`flex-1 px-3 py-2 rounded-xl text-sm border ${
                modelSize === "heavy"
                  ? "bg-indigo-500/10 border-indigo-400 text-indigo-200"
                  : "bg-black/30 border-gray-700 text-gray-300"
              }`}
              onClick={() => setModelSize("heavy")}
            >
              🧠 Heavy
            </button>
          </div>
        </div>

        {/* Compute tier card */}
        <div className="bg-[#111827] border border-gray-800 rounded-2xl p-4 flex flex-col gap-3">
          <h2 className="text-lg font-semibold">Compute Tier</h2>
          <p className="text-xs text-gray-400">
            Choose where training runs: your local GPU/CPU, or serverless GPUs via RunPod.
          </p>

          <div className="flex flex-col gap-2 mt-2">
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="radio"
                className="accent-emerald-500"
                value="local"
                checked={computeTier === "local"}
                onChange={() => setComputeTier("local")}
              />
              <span>💻 Local machine</span>
            </label>

            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="radio"
                className="accent-emerald-500"
                value="runpod_24"
                checked={computeTier === "runpod_24"}
                onChange={() => setComputeTier("runpod_24")}
              />
              <span>☁️ RunPod – Standard GPU (16–24GB)</span>
            </label>

            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="radio"
                className="accent-emerald-500"
                value="runpod_80"
                checked={computeTier === "runpod_80"}
                onChange={() => setComputeTier("runpod_80")}
              />
              <span>🚀 RunPod – 80GB GPU (heavy jobs)</span>
            </label>
          </div>
        </div>
      </div>

      {/* Universe Scanner (A–C blocks UI) */}
      <div className="bg-[#111827] border border-gray-800 rounded-2xl p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Universe Scanner</h2>
            <p className="text-xs text-gray-400">
              Runs every ~5 minutes and posts top candidates + anomalies to Discord. Does NOT auto-add to watchlist.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={startScanner}
              disabled={scannerRunning}
              className="text-[12px] px-3 py-1.5 rounded-xl border border-emerald-400 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-40"
            >
              ▶️ Start
            </button>
            <button
              onClick={stopScanner}
              disabled={!scanJobId || !scannerRunning}
              className="text-[12px] px-3 py-1.5 rounded-xl border border-gray-700 bg-black/40 text-gray-200 hover:bg-black/60 disabled:opacity-40"
            >
              ⏹ Stop
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-xs">
            <span className="text-gray-400">Status: </span>
            <span
              className={
                scanJob?.status === "scanning"
                  ? "text-emerald-300"
                  : scanJob?.status === "error"
                  ? "text-red-300"
                  : scanJob?.status === "queued"
                  ? "text-yellow-300"
                  : scanJob?.status === "stopped"
                  ? "text-gray-400"
                  : "text-gray-200"
              }
            >
              {scanJob?.status || (scanJobId ? "starting..." : "idle")}
            </span>
            {scanJobId && (
              <span className="text-[11px] text-gray-500 ml-2">job: {String(scanJobId).slice(0, 8)}</span>
            )}
            {typeof scanJob?.cycles === "number" && (
              <span className="text-[11px] text-gray-500 ml-2">cycles: {scanJob.cycles}</span>
            )}
          </div>

          {(scanJob?.status === "scanning" || scanJob?.status === "queued") && (
            <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full w-1/2 bg-emerald-500/70 blur-sm animate-pulse" />
            </div>
          )}
        </div>

        {scanError && <div className="text-xs text-red-300">🔴 {scanError}</div>}

        {scanJob?.last_result && (
          <div className="mt-2 bg-black/30 border border-gray-800 rounded-xl p-3">
            <div className="text-[11px] text-gray-400 mb-2">Last run: {formatDateTime(scanJob.last_run_at)}</div>

            <div className="text-xs font-semibold mb-1">Top candidates</div>
            <div className="text-[11px] text-gray-300">
              {(scanJob.last_result.top || []).slice(0, 10).map((x, i) => {
                const sym = typeof x === "string" ? x : x.symbol || JSON.stringify(x);
                return (
                  <span key={i} className="inline-block mr-2">
                    {sym}
                  </span>
                );
              })}
            </div>

            <div className="text-xs font-semibold mt-3 mb-1">Anomalies</div>
            <div className="text-[11px] text-gray-300">
              {(scanJob.last_result.alerts || []).slice(0, 8).map((x, i) => {
                const t = typeof x === "string" ? x : x.title || JSON.stringify(x);
                return <div key={i}>• {t}</div>;
              })}
            </div>
          </div>
        )}
      </div>

      {/* Training data controls */}
      <div className="bg-[#111827] border border-gray-800 rounded-2xl p-4 flex flex-col gap-4">
        <h2 className="text-lg font-semibold">Training Data & Knobs</h2>

        {/* Symbols */}
        <div className="flex flex-col gap-1">
          <label className="text-sm font-medium">Symbols list</label>
          <p className="text-xs text-gray-400 mb-1">
            Comma-separated tickers (e.g. <code>TSLA, NVDA, SPY, BTCUSD</code>).
          </p>
          <textarea
            className="w-full bg-black/40 border border-gray-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            rows={2}
            value={symbolsInput}
            onChange={(e) => setSymbolsInput(e.target.value)}
          />
        </div>

        {/* Epochs + timeframe */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">Epochs (one-shot)</label>
            <input
              type="number"
              min={1}
              max={500}
              className="w-full bg-black/40 border border-gray-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
            />
            <p className="text-[11px] text-gray-500">One-shot trains once and finishes. Endless is below.</p>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">Timeframe</label>
            <select
              className="w-full bg-black/40 border border-gray-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="1m">1 min (scalping)</option>
              <option value="5m">5 min</option>
              <option value="15m">15 min</option>
              <option value="1H">1 hour</option>
              <option value="4H">4 hour</option>
              <option value="1D">1 day (swing/investing)</option>
            </select>
            <p className="text-[11px] text-gray-500">For investing behavior: prefer 1H / 4H / 1D.</p>
          </div>
        </div>

        {/* Endless knobs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">Endless: epochs per cycle</label>
            <input
              type="number"
              min={1}
              max={200}
              className="w-full bg-black/40 border border-gray-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              value={epochsPerCycle}
              onChange={(e) => setEpochsPerCycle(e.target.value)}
            />
            <p className="text-[11px] text-gray-500">Endless repeats cycles until you stop.</p>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium">Endless: cooldown (seconds)</label>
            <input
              type="number"
              min={0}
              max={600}
              className="w-full bg-black/40 border border-gray-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              value={cooldownSec}
              onChange={(e) => setCooldownSec(e.target.value)}
            />
            <p className="text-[11px] text-gray-500">Small pause between cycles (keeps UI responsive).</p>
          </div>
        </div>

        {/* Status + synthetic progress bar */}
        <div className="flex items-center gap-3 mt-2">
          <div className="flex items-center gap-1 text-xs">
            <span className={statusInfo.color}>{statusInfo.emoji}</span>
            <span className={`${statusInfo.color} font-medium`}>{statusInfo.label}</span>

            {activeJobId && <span className="text-[11px] text-gray-500 ml-2">job: {String(activeJobId).slice(0, 8)}</span>}
            {typeof job?.cycles_completed === "number" && (
              <span className="text-[11px] text-gray-500 ml-2">cycles: {job.cycles_completed}</span>
            )}
          </div>

          {statusInfo.running && (
            <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full w-1/2 bg-emerald-500/70 blur-sm animate-pulse" />
            </div>
          )}
        </div>

        {/* Buttons */}
        <div className="flex flex-wrap gap-3 justify-end pt-2">
          {/* One-shot */}
          <button
            type="button"
            onClick={handleStartTraining}
            disabled={isTraining || statusInfo.running}
            className={`px-5 py-2.5 rounded-2xl text-sm font-medium flex items-center gap-2 border transition ${
              isTraining || statusInfo.running
                ? "bg-gray-800 border-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-emerald-500/10 border-emerald-400 text-emerald-200 hover:bg-emerald-500/20"
            }`}
            title="Runs once then finishes"
          >
            {isTraining ? (
              <>
                <span className="animate-spin">🌀</span>
                <span>Training...</span>
              </>
            ) : (
              <>
                <span>▶️</span>
                <span>Start One-Shot</span>
              </>
            )}
          </button>

          {/* Endless start */}
          <button
            type="button"
            onClick={handleStartEndless}
            disabled={statusInfo.running}
            className={`px-5 py-2.5 rounded-2xl text-sm font-medium flex items-center gap-2 border transition ${
              statusInfo.running
                ? "bg-gray-800 border-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-indigo-500/10 border-indigo-400 text-indigo-200 hover:bg-indigo-500/20"
            }`}
          >
            <span>♾️</span>
            <span>Start Endless</span>
          </button>

          {/* Endless stop */}
          <button
            type="button"
            onClick={handleStopEndless}
            disabled={!activeJobId}
            className={`px-5 py-2.5 rounded-2xl text-sm font-medium flex items-center gap-2 border transition ${
              !activeJobId
                ? "bg-gray-800 border-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-red-500/10 border-red-400 text-red-200 hover:bg-red-500/20"
            }`}
          >
            <span>⏹️</span>
            <span>Stop</span>
          </button>
        </div>
      </div>

      {/* Status + raw response */}
      <div className="bg-[#020617] border border-gray-800 rounded-2xl p-4 flex flex-col gap-3">
        <h2 className="text-sm font-semibold">Training Status</h2>

        {statusMessage && <p className="text-xs text-gray-200">{statusMessage}</p>}
        {jobError && <p className="text-xs text-red-300">Job error: {jobError}</p>}

        {job?.last_cycle && (
          <div className="text-xs text-gray-300 bg-black/40 border border-gray-800 rounded-xl p-3">
            <div className="text-[11px] text-gray-400 mb-1">Last cycle @ {formatDateTime(job.last_cycle.at)}</div>
            <pre className="text-xs overflow-x-auto">{JSON.stringify(job.last_cycle.result, null, 2)}</pre>
          </div>
        )}

        {lastResponse && (
          <pre className="mt-2 text-xs bg-black/40 border border-gray-800 rounded-xl p-3 overflow-x-auto">
            {JSON.stringify(lastResponse, null, 2)}
          </pre>
        )}
      </div>

      {/* Training History */}
      <div className="bg-[#020617] border border-gray-800 rounded-2xl p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Recent Training Runs</h2>
          <button
            type="button"
            onClick={fetchHistory}
            disabled={loadingHistory}
            className="text-[11px] px-3 py-1 rounded-xl border border-gray-700 bg-black/40 hover:bg-black/60"
          >
            {loadingHistory ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        {history.length === 0 && !loadingHistory && (
          <p className="text-xs text-gray-500">No training runs logged yet. Start a training job above.</p>
        )}

        <div className="flex flex-col gap-2">
          {history.map((h) => {
            const isLocal = h.mode === "local";
            const modeLabel =
              h.mode === "runpod_80"
                ? "RunPod 80GB"
                : h.mode === "runpod_24"
                ? "RunPod 24GB"
                : h.mode === "endless"
                ? "Endless"
                : "Local";

            const res = Array.isArray(h.results) ? h.results : [];
            const accs = res.filter((r) => typeof r.train_accuracy === "number").map((r) => r.train_accuracy);

            const avgAcc = accs.length > 0 ? (accs.reduce((a, b) => a + b, 0) / accs.length).toFixed(3) : null;

            const statusColor =
              h.status === "completed"
                ? "text-emerald-300"
                : h.status === "training"
                ? "text-emerald-300"
                : h.status === "queued"
                ? "text-sky-300"
                : h.status === "stopped"
                ? "text-red-300"
                : h.status === "failed"
                ? "text-red-300"
                : "text-yellow-300";

            return (
              <div key={h.id} className="border border-gray-800 rounded-xl p-3 bg-black/30 flex flex-col gap-1">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-col">
                    <span className="text-xs font-semibold">
                      {modeLabel} • {h.model_size?.toUpperCase()} • {h.timeframe}
                    </span>
                    <span className="text-[11px] text-gray-400">{h.symbols?.join(", ") || "—"}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[11px] text-gray-400 block">{formatDateTime(h.created_at)}</span>
                    <span className="text-[11px] text-gray-500">job: {h.id?.slice(0, 8) || "n/a"}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between mt-1">
                  <span className="text-[11px] text-gray-400">
                    Status: <span className={statusColor}>{h.status}</span>
                    {h.epochs != null && ` • epochs=${h.epochs}`}
                    {h.epochs_per_cycle != null && ` • cycle_epochs=${h.epochs_per_cycle}`}
                    {h.cycles_completed != null && ` • cycles=${h.cycles_completed}`}
                  </span>
                  {isLocal && avgAcc && <span className="text-[11px] text-emerald-300">Avg acc: {avgAcc}</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AILab;
