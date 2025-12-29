import os
import asyncio
import json
import math
import statistics
import random
import uuid  # for alert IDs
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

import requests
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Query,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OPTIONAL: if you keep using the job-based training router
from training import router as training_router

from db import open_db
from schemas import (
    WatchItem,
    Settings,
    Signal,
    WatchItemInfo,
    DrawingPayload,
    OrderRequest,
)
from providers import alpaca_client, binance_client
from providers import ohlc as ohlc_provider
from providers import alpaca_trading
from discord_client import send_discord
from ai.signals import infer_signal
from ai.overlays import build_overlays_from_ohlc
from ai.patterns_expert import build_expert_overlays
from ai.ensemble import infer_ensemble
from ai.trend_engine import run_trend_engine

# ---------------------- ENV LOADING ----------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

for env_path in [
    ROOT_DIR / ".env",
    BACKEND_DIR / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path, override=True)

# in case you also had a plain load_dotenv() call before
load_dotenv()

WS_INTERVAL = int(os.getenv("WS_BROADCAST_INTERVAL", "2"))

# RunPod envs
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_24GB = os.getenv("RUNPOD_ENDPOINT_24GB")  # your 16GB endpoint
RUNPOD_ENDPOINT_80GB = os.getenv("RUNPOD_ENDPOINT_80GB")  # your 80GB endpoint

# Where to save trained models
MODELS_DIR = BACKEND_DIR / "models"

# ---------------------- TORCH / MODEL ----------------------

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class SimplePriceMLP(nn.Module):
    """
    Very small MLP that takes a window of closes and predicts
    probability that the next step is "up".
    """

    def __init__(self, window: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def map_timeframe_alpaca(tf: str) -> str:
    """
    Map human timeframe like '1m', '5m', '1H', '1D' to Alpaca's.
    """
    tf = tf.upper()
    mapping = {
        "1M": "1Min",
        "5M": "5Min",
        "15M": "15Min",
        "30M": "30Min",
        "1H": "1Hour",
        "2H": "2Hour",
        "4H": "4Hour",
        "1D": "1Day",
        "1W": "1Week",
    }
    return mapping.get(tf, "1Min")


def map_timeframe_binance(tf: str) -> str:
    """
    Map timeframe to Binance intervals.
    """
    tf = tf.lower()
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
    }
    return mapping.get(tf, "1m")


# ---------------------- FASTAPI APP ----------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If you have extra training routes in training.py, keep them:
app.include_router(training_router)

clients: set[WebSocket] = set()
price_cache: Dict[str, float] = {}

DEFAULT_SETTINGS = {
    "discord_webhook": os.getenv("DISCORD_WEBHOOK_URL"),
    "use_paper": True,
    "use_crypto": False,
    "signal_schedule_min": 0,
    "risk_sl_pct": 1.5,
    "risk_tp_pct": 3.0,
    "risk_per_trade_pct": 1.0,
    "size_mode": "risk_pct",
    "train_backend": os.getenv("TRAIN_BACKEND", "local"),
    # horizon preferences
    "horizon_min_days": 5,
    "horizon_max_days": 30,
    "allow_long_horizon": True,

    # -------- POWER FEATURES (NEW) --------
    # Universe scan schedule (0 disables)
    "universe_scan_schedule_min": 0,  # e.g. 60 for hourly scans
    "universe_scan_top_n": 12,
    "universe_scan_timeframes": ["15m", "1H", "4H", "1D"],
    "universe_scan_include_equity": True,
    "universe_scan_include_crypto": True,

    # Auto-watchlist from scan (default OFF)
    "auto_watchlist_from_scan": False,
    "auto_watchlist_max_add": 8,

    # Portfolio anomaly schedule (0 disables)
    "portfolio_anomaly_schedule_min": 0,  # e.g. 120 for every 2 hours
    "portfolio_max_position_pct_equity": 20.0,
    "portfolio_max_unrealized_loss_pct": 8.0,
    "portfolio_max_unrealized_gain_pct": 20.0,
}

# ====================== UNIVERSE SCANNER + FLOW DETECTION ======================

SCANNER_INTERVAL_MIN = int(os.getenv("SCANNER_INTERVAL_MIN", "15"))
SCANNER_AUTO_ADD_TOP = os.getenv("SCANNER_AUTO_ADD_TOP", "0") == "1"
SCANNER_TOP_N = int(os.getenv("SCANNER_TOP_N", "10"))
SCANNER_TIMEFRAMES = os.getenv("SCANNER_TIMEFRAMES", "15m,1h,4h,1d")
SCANNER_STRICT_REALDATA = os.getenv("SCANNER_STRICT_REALDATA", "1") == "1"

STARTER_STOCKS = [
    # Your picks
    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","TSLA","AMD","META","MARA",
    # High-volume megacaps
    "AMZN","GOOGL","GOOG","NFLX","TSM","AVGO","ORCL","CRM","ADBE","INTC",
    "JPM","BAC","WFC","GS","V","MA","PYPL",
    "XOM","CVX","COP",
    "UNH","LLY","JNJ","PFE",
    "WMT","COST","HD","LOW",
    # Broad ETFs / sector ETFs
    "DIA","VTI","VOO","IVV","RSP",
    "XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLC","XLU","XLB","XLRE",
    "SMH","SOXX","ARKK",
    # Other liquid names
    "PLTR","COIN","ROKU","SNOW","SHOP","UBER","ABNB","NIO","RIVN","LCID",
]

STARTER_CRYPTO = [
    # Your picks (we normalize BTCUSD -> BTCUSDT below)
    "BTCUSD","ETHUSD","SOLUSD","DOGEUSD",
    # Common pairs
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","AVAXUSDT",
    "LINKUSDT","MATICUSDT","DOGEUSDT","LTCUSDT","BCHUSDT","DOTUSDT","ATOMUSDT",
]

def _normalize_crypto_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    # common: BTCUSD/ETHUSD -> BTCUSDT/ETHUSDT for Binance
    if s.endswith("USD") and not s.endswith("USDT"):
        return s + "T"
    return s

def _parse_timeframes(csv: str) -> List[str]:
    return [t.strip() for t in (csv or "").split(",") if t.strip()]

def _safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0

def _safe_stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    try:
        return float(statistics.stdev(xs))
    except Exception:
        return 0.0

def _trend_strength(closes: List[float]) -> float:
    """
    Simple trend proxy: slope of closes vs time, normalized by stdev.
    Returns roughly -3..+3 typical.
    """
    n = len(closes)
    if n < 20:
        return 0.0
    # linear regression slope
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = _safe_mean(closes)
    num = 0.0
    den = 0.0
    for i, x in enumerate(xs):
        dx = x - x_mean
        dy = closes[i] - y_mean
        num += dx * dy
        den += dx * dx
    slope = (num / den) if den else 0.0
    vol = _safe_stdev(closes)
    if vol <= 1e-9:
        return 0.0
    return float(slope / vol)

def _compute_scan_features(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    bars: list of {o,h,l,c,v,...}
    returns: metrics for ranking and alerts
    """
    if not bars or len(bars) < 60:
        return {}

    closes = [float(b["c"]) for b in bars if "c" in b]
    highs  = [float(b["h"]) for b in bars if "h" in b]
    lows   = [float(b["l"]) for b in bars if "l" in b]
    vols   = [float(b.get("v", 0.0) or 0.0) for b in bars]

    if len(closes) < 60:
        return {}

    last_close = closes[-1]
    last_vol = vols[-1] if vols else 0.0

    # rolling windows
    look = 50
    v_win = vols[-look:] if len(vols) >= look else vols
    c_win = closes[-look:] if len(closes) >= look else closes
    h_win = highs[-look:] if len(highs) >= look else highs
    l_win = lows[-look:] if len(lows) >= look else lows

    v_mean = _safe_mean(v_win)
    v_std = _safe_stdev(v_win)

    # Volume spike z-score
    vol_z = ((last_vol - v_mean) / v_std) if (v_std > 1e-9) else 0.0

    # Dollar-volume (proxy for “big money”)
    dollar_vol = last_close * last_vol
    dv_win = [(c_win[i] * v_win[i]) for i in range(min(len(c_win), len(v_win)))]
    dv_mean = _safe_mean(dv_win)
    dv_std = _safe_stdev(dv_win)
    dv_z = ((dollar_vol - dv_mean) / dv_std) if (dv_std > 1e-9) else 0.0

    # Range expansion (true range-ish)
    ranges = [(h_win[i] - l_win[i]) for i in range(min(len(h_win), len(l_win)))]
    last_range = ranges[-1] if ranges else 0.0
    r_mean = _safe_mean(ranges)
    r_std = _safe_stdev(ranges)
    range_z = ((last_range - r_mean) / r_std) if (r_std > 1e-9) else 0.0

    # Breakout vs recent highs
    recent_high = max(h_win[:-1]) if len(h_win) > 2 else (h_win[-1] if h_win else last_close)
    breakout = 1.0 if (last_close > recent_high) else 0.0

    # Trend strength (normalized slope)
    t_strength = _trend_strength(closes[-200:] if len(closes) > 200 else closes)

    # Composite score (tweakable)
    score = (
        0.35 * max(0.0, vol_z) +
        0.35 * max(0.0, dv_z) +
        0.20 * max(0.0, range_z) +
        0.10 * breakout +
        0.10 * max(0.0, t_strength)
    )

    return {
        "last_close": float(last_close),
        "last_vol": float(last_vol),
        "vol_z": float(vol_z),
        "dollar_vol": float(dollar_vol),
        "dv_z": float(dv_z),
        "range_z": float(range_z),
        "breakout": float(breakout),
        "trend_strength": float(t_strength),
        "score": float(score),
    }

async def _fetch_bars_for_scan(sym: str, tf: str, use_crypto: bool, limit: int = 400) -> List[Dict[str, Any]]:
    """
    Real-data only for scanner. Raises on failure if SCANNER_STRICT_REALDATA=1.
    """
    if use_crypto:
        interval = map_timeframe_binance(tf)
        s2 = _normalize_crypto_symbol(sym)
        bars = await ohlc_provider.binance_klines(s2, interval, limit)
    else:
        alp_tf = map_timeframe_alpaca(tf)
        bars = await ohlc_provider.alpaca_bars(sym, alp_tf, limit)

    if not bars:
        if SCANNER_STRICT_REALDATA:
            raise RuntimeError(f"No bars returned for {sym} tf={tf} crypto={use_crypto}")
        return []

    return bars

def _seed_universe_if_missing():
    with open_db() as db:
        if "universe:stocks" not in db:
            db["universe:stocks"] = sorted(list(set(STARTER_STOCKS)))
        if "universe:crypto" not in db:
            db["universe:crypto"] = sorted(list(set(STARTER_CRYPTO)))
        if "scanner:settings" not in db:
            db["scanner:settings"] = {
                "timeframes": _parse_timeframes(SCANNER_TIMEFRAMES),
                "top_n": SCANNER_TOP_N,
                "auto_add_top": SCANNER_AUTO_ADD_TOP,
                "interval_min": SCANNER_INTERVAL_MIN,
                # thresholds for “flow” alerts
                "vol_z_thresh": 2.5,
                "dv_z_thresh": 2.5,
                "range_z_thresh": 2.0,
            }

class UniversePayload(BaseModel):
    stocks: List[str] = Field(default_factory=list)
    crypto: List[str] = Field(default_factory=list)

class ScannerSettingsPayload(BaseModel):
    timeframes: List[str] = Field(default_factory=lambda: ["15m","1h","4h","1d"])
    top_n: int = 10
    auto_add_top: bool = False
    interval_min: int = 15
    vol_z_thresh: float = 2.5
    dv_z_thresh: float = 2.5
    range_z_thresh: float = 2.0

class ScanRequest(BaseModel):
    scan_stocks: bool = True
    scan_crypto: bool = True
    timeframes: Optional[List[str]] = None
    top_n: Optional[int] = None
    send_to_discord: bool = True
    auto_add_top: Optional[bool] = None

async def _run_universe_scan(req: ScanRequest) -> Dict[str, Any]:
    with open_db() as db:
        settings = {**DEFAULT_SETTINGS, **db.get("settings", {})}
        webhook = settings.get("discord_webhook")

        universe_stocks = db.get("universe:stocks", [])
        universe_crypto = db.get("universe:crypto", [])
        scan_settings = db.get("scanner:settings", {})

        tfs = req.timeframes or scan_settings.get("timeframes") or _parse_timeframes(SCANNER_TIMEFRAMES)
        top_n = int(req.top_n or scan_settings.get("top_n") or SCANNER_TOP_N)
        auto_add = bool(req.auto_add_top if req.auto_add_top is not None else scan_settings.get("auto_add_top", False))

        # thresholds for flow alerts
        vol_z_thresh = float(scan_settings.get("vol_z_thresh", 2.5))
        dv_z_thresh = float(scan_settings.get("dv_z_thresh", 2.5))
        range_z_thresh = float(scan_settings.get("range_z_thresh", 2.0))

    started = datetime.utcnow().isoformat() + "Z"
    results: List[Dict[str, Any]] = []
    flow_alerts: List[Dict[str, Any]] = []

    async def scan_symbol(sym: str, tf: str, is_crypto: bool):
        bars = await _fetch_bars_for_scan(sym, tf, is_crypto, limit=500)
        feats = _compute_scan_features(bars)
        if not feats:
            return

        row = {
            "symbol": sym.upper(),
            "tf": tf,
            "is_crypto": bool(is_crypto),
            **feats,
        }
        results.append(row)

        # “Large money / abnormal activity” proxy triggers
        if feats["vol_z"] >= vol_z_thresh or feats["dv_z"] >= dv_z_thresh or feats["range_z"] >= range_z_thresh:
            flow_alerts.append({
                "symbol": sym.upper(),
                "tf": tf,
                "is_crypto": bool(is_crypto),
                "vol_z": feats["vol_z"],
                "dv_z": feats["dv_z"],
                "range_z": feats["range_z"],
                "breakout": feats["breakout"],
                "trend_strength": feats["trend_strength"],
                "last_close": feats["last_close"],
            })

    # Scan sequentially (safe). We can parallelize later if you want.
    if req.scan_stocks:
        for sym in universe_stocks:
            for tf in tfs:
                try:
                    await scan_symbol(sym.strip().upper(), tf, False)
                except Exception as e:
                    if SCANNER_STRICT_REALDATA:
                        results.append({"symbol": sym.upper(), "tf": tf, "is_crypto": False, "error": str(e)})

    if req.scan_crypto:
        for sym in universe_crypto:
            for tf in tfs:
                try:
                    await scan_symbol(sym.strip().upper(), tf, True)
                except Exception as e:
                    if SCANNER_STRICT_REALDATA:
                        results.append({"symbol": sym.upper(), "tf": tf, "is_crypto": True, "error": str(e)})

    # Rank
    scored = [r for r in results if isinstance(r, dict) and ("score" in r)]
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    top = scored[:top_n]

    # Auto-add top symbols to watchlist (disabled by default)
    if auto_add:
        with open_db() as db:
            wl = db.get("watchlist", {})
            for r in top:
                sym = r.get("symbol")
                if not sym or r.get("is_crypto"):
                    continue
                if sym not in wl:
                    wl[sym] = {"symbol": sym, "price": 0.0, "change_pct": 0.0, "sparkline": []}
            db["watchlist"] = wl

    finished = datetime.utcnow().isoformat() + "Z"
    scan_doc = {
        "id": str(uuid.uuid4()),
        "started_at": started,
        "finished_at": finished,
        "timeframes": tfs,
        "top_n": top_n,
        "counts": {
            "total_rows": len(results),
            "scored_rows": len(scored),
            "flow_alerts": len(flow_alerts),
        },
        "top": top,
        "flow_alerts": flow_alerts[:50],
    }

    # Save history
    with open_db() as db:
        hist = db.get("scanner_history", [])
        hist.append(scan_doc)
        if len(hist) > 200:
            hist = hist[-200:]
        db["scanner_history"] = hist

    # Discord summary
    if req.send_to_discord and webhook:
        lines = [
            f"🔎 [Scanner] Universe Scan complete",
            f"TFs: {', '.join(tfs)} | Top={top_n} | Rows={len(scored)} | FlowAlerts={len(flow_alerts)}",
            "",
            "🏆 Top Candidates:"
        ]
        for r in top[:min(10, len(top))]:
            sym = r["symbol"]
            tf = r["tf"]
            score = r["score"]
            volz = r["vol_z"]
            dvz = r["dv_z"]
            rngz = r["range_z"]
            brk = "🚀" if r["breakout"] >= 1.0 else ""
            trend = r["trend_strength"]
            lines.append(f"- {sym} ({tf}) score={score:.2f} volZ={volz:.2f} dvZ={dvz:.2f} rngZ={rngz:.2f} trend={trend:.2f} {brk}")

        if flow_alerts:
            lines += ["", "🧲 Abnormal Activity (Flow Proxies):"]
            for a in flow_alerts[:min(10, len(flow_alerts))]:
                sym = a["symbol"]
                tf = a["tf"]
                volz = a["vol_z"]
                dvz = a["dv_z"]
                rngz = a["range_z"]
                brk = "🚀" if a["breakout"] >= 1.0 else ""
                lines.append(f"- {sym} ({tf}) volZ={volz:.2f} dvZ={dvz:.2f} rngZ={rngz:.2f} {brk}")

        try:
            await send_discord("\n".join(lines), webhook)
        except Exception:
            pass

    return scan_doc

# ====================== ENDLESS SCANNER JOBS ======================

SCANNER_JOBS: Dict[str, Dict[str, Any]] = {}
SCANNER_JOB_TASKS: Dict[str, asyncio.Task] = {}

class EndlessScanStartRequest(BaseModel):
    scan_stocks: bool = True
    scan_crypto: bool = True
    timeframes: Optional[List[str]] = None
    top_n: Optional[int] = None
    interval_sec: int = 600  # ✅ default 10 minutes (best middle ground)
    send_to_discord: bool = True
    auto_add_top: bool = False  # ✅ keep OFF

def _job_set(job_id: str, patch: Dict[str, Any]):
    cur = SCANNER_JOBS.get(job_id, {})
    cur.update(patch)
    SCANNER_JOBS[job_id] = cur

    with open_db() as db:
        jobs = db.get("scanner_jobs", {})
        j = jobs.get(job_id, {})
        j.update(patch)
        jobs[job_id] = j
        db["scanner_jobs"] = jobs

def _job_get(job_id: str) -> Optional[Dict[str, Any]]:
    if job_id in SCANNER_JOBS:
        return SCANNER_JOBS[job_id]
    with open_db() as db:
        jobs = db.get("scanner_jobs", {})
        if job_id in jobs:
            SCANNER_JOBS[job_id] = jobs[job_id]
            return jobs[job_id]
    return None

async def _endless_scan_loop(job_id: str, req: EndlessScanStartRequest):
    _job_set(job_id, {
        "id": job_id,
        "type": "endless_scanner",
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_run_at": None,
        "last_error": None,
        "cycles": 0,
        "config": req.dict(),
    })

    # quick initial delay so UI can show queued, then run right away
    await asyncio.sleep(2)
    _job_set(job_id, {"status": "idle"})

    while True:
        j = _job_get(job_id)
        if not j:
            return
        if j.get("status") == "stopped":
            return

        try:
            _job_set(job_id, {"status": "scanning", "last_error": None})

            scan_req = ScanRequest(
                scan_stocks=req.scan_stocks,
                scan_crypto=req.scan_crypto,
                timeframes=req.timeframes,
                top_n=req.top_n,
                send_to_discord=req.send_to_discord,
                auto_add_top=req.auto_add_top,
            )

            doc = await _run_universe_scan(scan_req)

            _job_set(job_id, {
                "status": "idle",
                "last_run_at": datetime.utcnow().isoformat() + "Z",
                "cycles": int(j.get("cycles", 0)) + 1,
                "last_result": {
                    "scan_id": doc.get("id"),
                    "counts": doc.get("counts"),
                    "top": doc.get("top", [])[:10],
                    "flow_alerts": doc.get("flow_alerts", [])[:10],
                }
            })

        except Exception as e:
            _job_set(job_id, {
                "status": "error",
                "last_error": str(e),
                "last_run_at": datetime.utcnow().isoformat() + "Z",
            })

        await asyncio.sleep(max(10, int(req.interval_sec)))

@app.post("/scanner/endless/start")
async def scanner_endless_start(req: EndlessScanStartRequest):
    _seed_universe_if_missing()
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(_endless_scan_loop(job_id, req))
    SCANNER_JOB_TASKS[job_id] = task
    return {"ok": True, "job_id": job_id, "status": "queued"}

@app.get("/scanner/job/{job_id}")
async def scanner_job_status(job_id: str):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.post("/scanner/endless/stop/{job_id}")
async def scanner_endless_stop(job_id: str):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    _job_set(job_id, {"status": "stopped", "stopped_at": datetime.utcnow().isoformat() + "Z"})
    t = SCANNER_JOB_TASKS.get(job_id)
    if t and not t.done():
        t.cancel()
    return {"ok": True, "job_id": job_id, "status": "stopped"}

# ---------------------- Scanner API ----------------------

@app.get("/scanner/universe")
async def scanner_get_universe():
    _seed_universe_if_missing()
    with open_db() as db:
        return {
            "stocks": db.get("universe:stocks", []),
            "crypto": db.get("universe:crypto", []),
        }

@app.post("/scanner/universe")
async def scanner_set_universe(payload: UniversePayload):
    stocks = sorted(list({s.strip().upper() for s in payload.stocks if s.strip()}))
    crypto = sorted(list({_normalize_crypto_symbol(s) for s in payload.crypto if s.strip()}))
    with open_db() as db:
        db["universe:stocks"] = stocks
        db["universe:crypto"] = crypto
    return {"ok": True, "stocks": len(stocks), "crypto": len(crypto)}

@app.get("/scanner/settings")
async def scanner_get_settings():
    _seed_universe_if_missing()
    with open_db() as db:
        return db.get("scanner:settings", {})

@app.post("/scanner/settings")
async def scanner_set_settings(payload: ScannerSettingsPayload):
    with open_db() as db:
        db["scanner:settings"] = payload.dict()
    return {"ok": True, **payload.dict()}

@app.post("/scanner/run")
async def scanner_run(req: ScanRequest):
    _seed_universe_if_missing()
    return await _run_universe_scan(req)

@app.get("/scanner/history")
async def scanner_history(limit: int = 20):
    with open_db() as db:
        hist = db.get("scanner_history", [])
    hist_sorted = sorted(hist, key=lambda x: x.get("started_at", ""), reverse=True)
    return hist_sorted[:max(1, min(200, int(limit)))]

# ====================== INTEL BRIEF + HISTORY + ENDLESS INTEL JOBS (NEW) ======================

INTEL_JOBS: Dict[str, Dict[str, Any]] = {}
INTEL_JOB_TASKS: Dict[str, asyncio.Task] = {}

def _intel_set(job_id: str, patch: Dict[str, Any]):
    cur = INTEL_JOBS.get(job_id, {})
    cur.update(patch)
    INTEL_JOBS[job_id] = cur
    with open_db() as db:
        jobs = db.get("intel_jobs", {})
        j = jobs.get(job_id, {})
        j.update(patch)
        jobs[job_id] = j
        db["intel_jobs"] = jobs

def _intel_get(job_id: str) -> Optional[Dict[str, Any]]:
    if job_id in INTEL_JOBS:
        return INTEL_JOBS[job_id]
    with open_db() as db:
        jobs = db.get("intel_jobs", {})
        if job_id in jobs:
            INTEL_JOBS[job_id] = jobs[job_id]
            return jobs[job_id]
    return None

def _detect_regime_from_scan_top(top_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight regime proxy using SPY (preferred) if present in top list, else fallback.
    We'll do a real regime read using bars if possible (more accurate).
    """
    try:
        # Use SPY as default proxy
        return {"ok": True, "regime": "unknown", "emoji": "⚪"}
    except Exception:
        return {"ok": False, "regime": "unknown", "emoji": "⚪"}

async def _regime_from_symbol(proxy_sym: str, is_crypto: bool) -> Dict[str, Any]:
    try:
        bars = await _fetch_bars_for_scan(proxy_sym, "1D", is_crypto, limit=250)
    except Exception:
        return {"ok": False, "regime": "unknown", "emoji": "⚪"}

    if not bars or len(bars) < 80:
        return {"ok": False, "regime": "unknown", "emoji": "⚪"}

    closes = [float(b.get("c", 0.0)) for b in bars]
    highs = [float(b.get("h", 0.0)) for b in bars]
    lows = [float(b.get("l", 0.0)) for b in bars]

    n = min(120, len(closes))
    c = closes[-n:]
    h = highs[-n:]
    l = lows[-n:]

    slope = (c[-1] - c[0]) / max(1e-9, abs(c[0]))
    ranges = [(h[i] - l[i]) for i in range(n)]
    avg_range = statistics.mean(ranges) if ranges else 0.0
    net_move = abs(c[-1] - c[0])
    chop = (avg_range / max(1e-9, net_move)) if net_move > 0 else 999.0

    if abs(slope) > 0.02 and chop < 0.35:
        return {"ok": True, "regime": "trending", "emoji": "🟢", "slope": round(float(slope), 4), "chop": round(float(chop), 3)}
    if chop >= 0.55:
        return {"ok": True, "regime": "choppy", "emoji": "🟡", "slope": round(float(slope), 4), "chop": round(float(chop), 3)}
    return {"ok": True, "regime": "mixed", "emoji": "🟦", "slope": round(float(slope), 4), "chop": round(float(chop), 3)}

class IntelBriefRequest(BaseModel):
    scan_stocks: bool = True
    scan_crypto: bool = True
    timeframes: Optional[List[str]] = None
    top_n: int = 12
    send_to_discord: bool = True
    include_regime: bool = True
    include_flow: bool = True

async def _build_intel_brief(req: IntelBriefRequest) -> Dict[str, Any]:
    _seed_universe_if_missing()

    with open_db() as db:
        settings = {**DEFAULT_SETTINGS, **db.get("settings", {})}
        webhook = settings.get("discord_webhook")

    scan = await _run_universe_scan(ScanRequest(
        scan_stocks=req.scan_stocks,
        scan_crypto=req.scan_crypto,
        timeframes=req.timeframes,
        top_n=req.top_n,
        send_to_discord=False,   # we format our own intel message
        auto_add_top=False
    ))

    top = scan.get("top", [])[:req.top_n]
    flow = scan.get("flow_alerts", [])[:min(25, req.top_n * 2)]

    # regime proxy: SPY if scanning stocks, else BTCUSD
    regime = {"ok": False, "regime": "unknown", "emoji": "⚪"}
    if req.include_regime:
        if req.scan_stocks:
            regime = await _regime_from_symbol("SPY", False)
        else:
            regime = await _regime_from_symbol("BTCUSD", True)

    # risk note
    risk_note = ""
    if regime.get("regime") == "choppy":
        risk_note += "🟡 Regime is CHOPPY → reduce size / tighten stops.\n"
    if req.include_flow and len(flow) >= 8:
        risk_note += f"🐳 Many abnormal-activity hits ({len(flow)}) → expect volatility spikes.\n"

    brief = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "timeframes": scan.get("timeframes", req.timeframes or []),
        "counts": scan.get("counts", {}),
        "regime": regime,
        "risk_note": risk_note.strip(),
        "top": top,
        "flow_alerts": flow if req.include_flow else [],
        "scan_id": scan.get("id"),
    }

    # store brief
    with open_db() as db:
        hist = db.get("intel_briefs", [])
        hist.append(brief)
        if len(hist) > 300:
            hist = hist[-300:]
        db["intel_briefs"] = hist

    # discord
    if req.send_to_discord and webhook:
        lines = []
        if regime.get("ok"):
            lines.append(f"🧠 [Intel Brief] Regime: {regime.get('emoji','⚪')} {regime.get('regime','unknown')} (slope={regime.get('slope','?')} chop={regime.get('chop','?')})")
        else:
            lines.append("🧠 [Intel Brief] Regime: ⚪ unknown")

        if brief["risk_note"]:
            lines.append(brief["risk_note"])

        tfs = brief.get("timeframes") or []
        lines.append(f"TFs: {', '.join(tfs) if tfs else '—'} | top={len(top)} | flow={len(brief.get('flow_alerts', []))}")
        lines.append("")

        if top:
            lines.append("🏆 Top candidates:")
            for r in top[:min(req.top_n, 12)]:
                sym = r.get("symbol", "—")
                tf = r.get("tf", "")
                score = float(r.get("score", 0.0))
                volz = float(r.get("vol_z", 0.0))
                dvz = float(r.get("dv_z", 0.0))
                rngz = float(r.get("range_z", 0.0))
                brk = "🚀" if float(r.get("breakout", 0.0)) >= 1.0 else ""
                lines.append(f"- {sym} ({tf}) score={score:.2f} volZ={volz:.2f} dvZ={dvz:.2f} rngZ={rngz:.2f} {brk}")

        if brief.get("flow_alerts"):
            lines.append("")
            lines.append("🧲 Abnormal activity hits:")
            for a in brief["flow_alerts"][:10]:
                sym = a.get("symbol", "—")
                tf = a.get("tf", "")
                volz = float(a.get("vol_z", 0.0))
                dvz = float(a.get("dv_z", 0.0))
                rngz = float(a.get("range_z", 0.0))
                brk = "🚀" if float(a.get("breakout", 0.0)) >= 1.0 else ""
                lines.append(f"- {sym} ({tf}) volZ={volz:.2f} dvZ={dvz:.2f} rngZ={rngz:.2f} {brk}")

        try:
            await send_discord("\n".join(lines), webhook)
        except Exception:
            pass

    return brief

@app.post("/intel/brief")
async def intel_brief(req: IntelBriefRequest):
    return await _build_intel_brief(req)

@app.get("/intel/history")
async def intel_history(limit: int = 20):
    with open_db() as db:
        hist = db.get("intel_briefs", [])
    hist_sorted = sorted(hist, key=lambda x: x.get("created_at", ""), reverse=True)
    return hist_sorted[:max(1, min(200, int(limit)))]

class IntelEndlessStartRequest(BaseModel):
    scan_stocks: bool = True
    scan_crypto: bool = True
    timeframes: Optional[List[str]] = None
    top_n: int = 12
    interval_sec: int = 600  # ✅ default 10 minutes
    send_to_discord: bool = True
    include_regime: bool = True
    include_flow: bool = True

async def _intel_loop(job_id: str, req: IntelEndlessStartRequest):
    _intel_set(job_id, {
        "id": job_id,
        "type": "endless_intel",
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_run_at": None,
        "last_error": None,
        "cycles": 0,
        "config": req.dict(),
        "last_brief_id": None,
    })

    await asyncio.sleep(2)
    _intel_set(job_id, {"status": "idle"})

    while True:
        j = _intel_get(job_id)
        if not j:
            return
        if j.get("status") == "stopped":
            return

        try:
            _intel_set(job_id, {"status": "running", "last_error": None})

            brief = await _build_intel_brief(IntelBriefRequest(
                scan_stocks=req.scan_stocks,
                scan_crypto=req.scan_crypto,
                timeframes=req.timeframes,
                top_n=req.top_n,
                send_to_discord=req.send_to_discord,
                include_regime=req.include_regime,
                include_flow=req.include_flow,
            ))

            _intel_set(job_id, {
                "status": "idle",
                "last_run_at": datetime.utcnow().isoformat() + "Z",
                "cycles": int(j.get("cycles", 0)) + 1,
                "last_brief_id": brief.get("id"),
                "last_result": {
                    "brief_id": brief.get("id"),
                    "scan_id": brief.get("scan_id"),
                    "top": (brief.get("top") or [])[:10],
                    "flow_alerts": (brief.get("flow_alerts") or [])[:10],
                    "regime": brief.get("regime"),
                }
            })

        except Exception as e:
            _intel_set(job_id, {
                "status": "error",
                "last_error": str(e),
                "last_run_at": datetime.utcnow().isoformat() + "Z",
            })

        await asyncio.sleep(max(30, int(req.interval_sec)))

@app.post("/intel/endless/start")
async def intel_endless_start(req: IntelEndlessStartRequest):
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(_intel_loop(job_id, req))
    INTEL_JOB_TASKS[job_id] = task
    return {"ok": True, "job_id": job_id, "status": "queued"}

@app.get("/intel/job/{job_id}")
async def intel_job_status(job_id: str):
    job = _intel_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.post("/intel/endless/stop/{job_id}")
async def intel_endless_stop(job_id: str):
    job = _intel_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    _intel_set(job_id, {"status": "stopped", "stopped_at": datetime.utcnow().isoformat() + "Z"})
    t = INTEL_JOB_TASKS.get(job_id)
    if t and not t.done():
        t.cancel()
    return {"ok": True, "job_id": job_id, "status": "stopped"}

# ==================== END INTEL SECTION ====================

async def fetch_quote_and_spark(symbol: str, use_crypto: bool):
    """
    Get last price + sparkline for a symbol.
    Uses Alpaca or Binance with fallback on provider error.
    """
    sym = symbol.upper()
    price = None
    spark: List[float] = []

    try:
        if use_crypto:
            price = await binance_client.get_last_price(sym)
            spark = await binance_client.get_sparkline(sym)
        else:
            price = await alpaca_client.get_last_quote(sym)
            spark = await alpaca_client.get_sparkline(sym)
    except Exception as e:
        print(f"[fetch_quote_and_spark] error for {sym}: {e} – using fallback data.")
        base = price_cache.get(sym, 100.0)
        price = base + random.uniform(-1.0, 1.0)
        spark = [base + random.uniform(-2.0, 2.0) for _ in range(50)]

    final_price = float(price or 0.0)
    price_cache[sym] = final_price

    change_pct = 0.0
    if spark and spark[0]:
        try:
            first = float(spark[0])
            if first != 0:
                change_pct = (final_price - first) / first * 100.0
        except Exception:
            change_pct = 0.0

    return final_price, change_pct, spark

# ---------------------- TRAINING (for AI Lab) ----------------------

class TrainingRequest(BaseModel):
    symbols: List[str] = Field(default=["TSLA"], description="List of ticker symbols to train on.")
    epochs: int = Field(default=5, ge=1, le=500, description="Number of epochs.")
    timeframe: str = Field(default="1D", description="Timeframe like '1m', '15m', '1H', '4H', '1D'.")
    model_size: Literal["light", "heavy"] = Field(default="light", description="Lightweight or heavy model.")
    compute_tier: Literal["local", "runpod_24", "runpod_80"] = Field(default="local", description="Where to run training.")

def trigger_runpod_training(endpoint_id: str, payload: dict) -> dict:
    """
    Fire a RunPod Serverless job for training.
    """
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is not set in environment")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    resp = requests.post(url, headers=headers, json={"input": payload}, timeout=30)
    resp.raise_for_status()
    return resp.json()

async def run_local_training_job(req: TrainingRequest) -> dict:
    """
    Real local training loop using PyTorch.
    """
    if torch is None or nn is None:
        raise HTTPException(
            status_code=500,
            detail="PyTorch is not installed. Run 'pip install torch' in the backend environment.",
        )

    MODELS_DIR.mkdir(exist_ok=True)

    with open_db() as db:
        settings = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **settings}
        use_crypto = s.get("use_crypto", False)
        webhook = s.get("discord_webhook")

    window = 50 if req.model_size == "light" else 100
    symbols = [sym.strip().upper() for sym in req.symbols if sym.strip()]
    results: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            if use_crypto:
                interval = map_timeframe_binance(req.timeframe)
                bars = await ohlc_provider.binance_klines(sym, interval, 2000)
            else:
                tf = map_timeframe_alpaca(req.timeframe)
                bars = await ohlc_provider.alpaca_bars(sym, tf, 2000)

            closes = [float(b["c"]) for b in bars] if bars else []
            if len(closes) < window + 2:
                raise ValueError(f"Not enough bars for {sym} (need > {window + 2})")

            xs: List[List[float]] = []
            ys: List[float] = []
            for i in range(window, len(closes) - 1):
                window_vals = closes[i - window : i]
                next_up = 1.0 if closes[i + 1] > closes[i] else 0.0
                xs.append(window_vals)
                ys.append(next_up)

            X = torch.tensor(xs, dtype=torch.float32)
            y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimplePriceMLP(window=window).to(device)
            X = X.to(device)
            y = y.to(device)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            batch_size = 64

            for _epoch in range(req.epochs):
                perm = torch.randperm(X.size(0))
                X_shuf = X[perm]
                y_shuf = y[perm]

                for start in range(0, X_shuf.size(0), batch_size):
                    xb = X_shuf[start : start + batch_size]
                    yb = y_shuf[start : start + batch_size]
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                preds = model(X)
                preds_bin = (preds > 0.5).float()
                acc = (preds_bin == y).float().mean().item()

            model_path = MODELS_DIR / f"{sym}_mlp_{req.timeframe}_{req.model_size}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "window": window,
                    "timeframe": req.timeframe,
                    "symbol": sym,
                    "model_size": req.model_size,
                },
                model_path,
            )

            results.append(
                {
                    "symbol": sym,
                    "samples": len(xs),
                    "train_accuracy": round(acc, 4),
                    "model_path": str(model_path),
                }
            )

        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    job_id = str(uuid.uuid4())
    job_doc = {
        "id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "completed",
        "mode": "local",
        "compute_tier": "local",
        "model_size": req.model_size,
        "timeframe": req.timeframe,
        "epochs": req.epochs,
        "symbols": symbols,
        "results": results,
    }

    with open_db() as db:
        hist = db.get("training_jobs", [])
        hist.append(job_doc)
        if len(hist) > 200:
            hist = hist[-200:]
        db["training_jobs"] = hist

    if webhook:
        accs = [r["train_accuracy"] for r in results if "train_accuracy" in r]
        avg_acc = sum(accs) / len(accs) if accs else None

        lines = [
            f"[Train][local] job={job_id}",
            f"Symbols: {', '.join(symbols)}",
            f"Model: MLP-{req.model_size}  |  TF={req.timeframe}  |  epochs={req.epochs}",
        ]
        if accs:
            lines.append(f"Avg train acc: {avg_acc:.3f}")
        for r in results:
            if "train_accuracy" in r:
                lines.append(f"- {r['symbol']}: samples={r['samples']} acc={r['train_accuracy']:.3f}")
            else:
                lines.append(f"- {r['symbol']}: ERROR {r.get('error')}")

        try:
            await send_discord("\n".join(lines), webhook)
        except Exception:
            pass

    return {
        "status": "completed",
        "mode": "local",
        "job_id": job_id,
        "model_type": f"MLP-{req.model_size}",
        "timeframe": req.timeframe,
        "results": results,
    }

@app.post("/ai/train")
async def start_training(req: TrainingRequest):
    """
    Main training endpoint called by AILab.jsx.
    """
    training_payload = {
        "symbols": req.symbols,
        "epochs": req.epochs,
        "timeframe": req.timeframe,
        "model_size": req.model_size,
    }

    if req.compute_tier == "local":
        return await run_local_training_job(req)

    with open_db() as db:
        settings = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **settings}
        webhook = s.get("discord_webhook")

    if req.compute_tier == "runpod_24":
        if not RUNPOD_ENDPOINT_24GB:
            raise HTTPException(status_code=500, detail="RUNPOD_ENDPOINT_24GB not configured in .env")

        rp_result = trigger_runpod_training(endpoint_id=RUNPOD_ENDPOINT_24GB, payload=training_payload)

        job_id = str(uuid.uuid4())
        job_doc = {
            "id": job_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "submitted",
            "mode": "runpod_24",
            "compute_tier": "runpod_24",
            "model_size": req.model_size,
            "timeframe": req.timeframe,
            "epochs": req.epochs,
            "symbols": req.symbols,
            "runpod_response": rp_result,
        }

        with open_db() as db:
            hist = db.get("training_jobs", [])
            hist.append(job_doc)
            if len(hist) > 200:
                hist = hist[-200:]
            db["training_jobs"] = hist

        if webhook:
            try:
                await send_discord(
                    "\n".join(
                        [
                            f"[Train][RunPod-24GB] job={job_id}",
                            f"Symbols: {', '.join(req.symbols)}",
                            f"Model: MLP-{req.model_size}  |  TF={req.timeframe}  |  epochs={req.epochs}",
                            f"RunPod response: {rp_result.get('id', 'no-id')}",
                        ]
                    ),
                    webhook,
                )
            except Exception:
                pass

        return {"status": "submitted", "mode": "runpod_24", "job_id": job_id, "runpod_response": rp_result}

    if req.compute_tier == "runpod_80":
        if not RUNPOD_ENDPOINT_80GB:
            raise HTTPException(status_code=500, detail="RUNPOD_ENDPOINT_80GB not configured in .env")

        rp_result = trigger_runpod_training(endpoint_id=RUNPOD_ENDPOINT_80GB, payload=training_payload)

        job_id = str(uuid.uuid4())
        job_doc = {
            "id": job_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "submitted",
            "mode": "runpod_80",
            "compute_tier": "runpod_80",
            "model_size": req.model_size,
            "timeframe": req.timeframe,
            "epochs": req.epochs,
            "symbols": req.symbols,
            "runpod_response": rp_result,
        }

        with open_db() as db:
            hist = db.get("training_jobs", [])
            hist.append(job_doc)
            if len(hist) > 200:
                hist = hist[-200:]
            db["training_jobs"] = hist

        if webhook:
            try:
                await send_discord(
                    "\n".join(
                        [
                            f"[Train][RunPod-80GB] job={job_id}",
                            f"Symbols: {', '.join(req.symbols)}",
                            f"Model: MLP-{req.model_size}  |  TF={req.timeframe}  |  epochs={req.epochs}",
                            f"RunPod response: {rp_result.get('id', 'no-id')}",
                        ]
                    ),
                    webhook,
                )
            except Exception:
                pass

        return {"status": "submitted", "mode": "runpod_80", "job_id": job_id, "runpod_response": rp_result}

    raise HTTPException(status_code=400, detail="Invalid compute_tier")

@app.get("/ai/training/history")
async def get_training_history(limit: int = 20):
    with open_db() as db:
        hist = db.get("training_jobs", [])
    hist_sorted = sorted(hist, key=lambda x: x.get("created_at", ""), reverse=True)
    rows = hist_sorted[:max(1, min(200, int(limit)))]

    # UI-friendly wrapper (safe)
    return {"rows": rows, "count": len(rows)}


# =========================
# TRAINING JOB STATUS + ENDLESS CONTROL
# =========================

TRAIN_TASKS: Dict[str, asyncio.Task] = {}

class EndlessTrainRequest(BaseModel):
    symbols: List[str] = Field(default=["TSLA"])
    timeframe: str = Field(default="1D")
    model_size: Literal["light", "heavy"] = Field(default="heavy")
    compute_tier: Literal["local", "runpod_24", "runpod_80"] = Field(default="local")
    epochs_per_cycle: int = Field(default=3, ge=1, le=200)
    cooldown_sec: int = Field(default=2, ge=0, le=600)
    send_to_discord: bool = True

def _set_job_status(job_id: str, status: str, extra: Optional[dict] = None):
    with open_db() as db:
        jobs = db.get("training_jobs", [])
        for i in range(len(jobs)-1, -1, -1):
            if jobs[i].get("id") == job_id:
                jobs[i]["status"] = status
                jobs[i]["updated_at"] = datetime.utcnow().isoformat() + "Z"
                if extra:
                    jobs[i].update(extra)
                db["training_jobs"] = jobs
                return

@app.get("/ai/train/job/{job_id}")
async def get_train_job(job_id: str):
    with open_db() as db:
        jobs = db.get("training_jobs", [])
    for j in jobs:
        if j.get("id") == job_id:
            return j
    raise HTTPException(status_code=404, detail="job not found")

@app.post("/ai/train/endless/start")
async def start_endless_training(req: EndlessTrainRequest):
    with open_db() as db:
        settings = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **settings}
        webhook = s.get("discord_webhook")

    job_id = str(uuid.uuid4())
    job_doc = {
        "id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "status": "queued",
        "mode": "endless",
        "compute_tier": req.compute_tier,
        "model_size": req.model_size,
        "timeframe": req.timeframe,
        "symbols": [x.strip().upper() for x in req.symbols if x.strip()],
        "epochs_per_cycle": req.epochs_per_cycle,
        "cooldown_sec": req.cooldown_sec,
        "cycles_completed": 0,
        "last_cycle": None,
    }

    with open_db() as db:
        hist = db.get("training_jobs", [])
        hist.append(job_doc)
        if len(hist) > 200:
            hist = hist[-200:]
        db["training_jobs"] = hist

    async def runner():
        try:
            _set_job_status(job_id, "training")
            if req.send_to_discord and webhook:
                await send_discord(
                    "\n".join([
                        f"🟢 [Endless Train] job={job_id}",
                        f"Tier={req.compute_tier} | model={req.model_size} | TF={req.timeframe}",
                        f"Symbols: {', '.join(job_doc['symbols'])}",
                        "To stop: use Stop button in AI Lab."
                    ]),
                    webhook
                )

            cycles = 0
            while True:
                with open_db() as db:
                    jlist = db.get("training_jobs", [])
                    j = next((x for x in jlist if x.get("id")==job_id), None)
                    if not j:
                        break
                    if j.get("status") in ("stopping", "stopped"):
                        break

                sub_req = TrainingRequest(
                    symbols=job_doc["symbols"],
                    epochs=req.epochs_per_cycle,
                    timeframe=req.timeframe,
                    model_size=req.model_size,
                    compute_tier=req.compute_tier,
                )

                cycle_result = await start_training(sub_req)

                cycles += 1
                _set_job_status(job_id, "training", {
                    "cycles_completed": cycles,
                    "last_cycle": {
                        "at": datetime.utcnow().isoformat() + "Z",
                        "result": cycle_result,
                    }
                })

                if req.send_to_discord and webhook and (cycles % 5 == 0):
                    await send_discord(
                        f"🧠 [Endless Train] job={job_id} cycles={cycles} still running…",
                        webhook
                    )

                await asyncio.sleep(req.cooldown_sec)

            _set_job_status(job_id, "stopped")
            if req.send_to_discord and webhook:
                await send_discord(f"🔴 [Endless Train] job={job_id} stopped | cycles={cycles}", webhook)

        except asyncio.CancelledError:
            _set_job_status(job_id, "stopped")
        except Exception as e:
            _set_job_status(job_id, "failed", {"error": str(e)})
            if webhook:
                try:
                    await send_discord(f"❌ [Endless Train] job={job_id} failed: {e}", webhook)
                except Exception:
                    pass

    task = asyncio.create_task(runner())
    TRAIN_TASKS[job_id] = task
    return {"ok": True, "job_id": job_id, "status": "queued"}

@app.post("/ai/train/endless/stop/{job_id}")
async def stop_endless_training(job_id: str):
    _set_job_status(job_id, "stopping")
    t = TRAIN_TASKS.get(job_id)
    if t and not t.done():
        t.cancel()
    _set_job_status(job_id, "stopped")
    return {"ok": True, "job_id": job_id, "status": "stopped"}

# ---------------------- SETTINGS ----------------------

@app.get("/settings", response_model=Settings)
async def get_settings():
    with open_db() as db:
        s = db.get("settings", {})
        return Settings(**{**DEFAULT_SETTINGS, **s})

@app.post("/settings", response_model=Settings)
async def update_settings(s: Settings):
    with open_db() as db:
        cur = db.get("settings", {})
        cur.update({k: v for k, v in s.dict().items() if v is not None})
        db["settings"] = cur
        return Settings(**{**DEFAULT_SETTINGS, **cur})

# ---------------------- WATCHLIST ----------------------

@app.post("/watchlist/add", response_model=WatchItemInfo)
async def add_watch(item: WatchItem):
    sym = item.symbol.upper()
    with open_db() as db:
        use_crypto = db.get("settings", {}).get("use_crypto", False)

    try:
        price, change_pct, spark = await fetch_quote_and_spark(sym, use_crypto)
    except Exception as e:
        print(f"[watchlist/add] hard failure for {sym}: {e} – defaulting to zeros.")
        price, change_pct, spark = 0.0, 0.0, []

    info = {"symbol": sym, "price": price, "change_pct": change_pct, "sparkline": spark}

    with open_db() as db:
        wl = db.get("watchlist", {})
        wl[sym] = info
        db["watchlist"] = wl

    return WatchItemInfo(**info)

@app.get("/watchlist")
async def list_watch():
    with open_db() as db:
        return list(db.get("watchlist", {}).values())

@app.delete("/watchlist/{symbol}")
async def del_watch(symbol: str):
    sym = symbol.upper()
    with open_db() as db:
        wl = db.get("watchlist", {})
        wl.pop(sym, None)
        db["watchlist"] = wl
    return {"ok": True}

# ---------------------- SYMBOL ----------------------

@app.post("/symbol/{symbol}")
async def set_symbol(symbol: str):
    with open_db() as db:
        db["current_symbol"] = symbol.upper()
    return {"symbol": symbol.upper()}

@app.get("/symbol")
async def get_symbol():
    with open_db() as db:
        return {"symbol": db.get("current_symbol", "AAPL")}

# ---------------------- DRAWINGS ----------------------

@app.get("/drawings/{symbol}")
async def drawings_get(symbol: str):
    with open_db() as db:
        return db.get(f"drawings:{symbol.upper()}", [])

@app.post("/drawings")
async def drawings_save(payload: DrawingPayload):
    with open_db() as db:
        db[f"drawings:{payload.symbol.upper()}"] = payload.objects
    return {"ok": True, "count": len(payload.objects)}

# ---------------------- PORTFOLIO / ORDERS ----------------------

@app.get("/portfolio/account")
async def account():
    with open_db() as db:
        use_paper = db.get("settings", {}).get("use_paper", True)
    return await alpaca_trading.get_account(use_paper)

@app.get("/portfolio/positions")
async def positions():
    with open_db() as db:
        use_paper = db.get("settings", {}).get("use_paper", True)
    return await alpaca_trading.get_positions(use_paper)

@app.get("/orders/history")
async def orders(status: str = "all", limit: int = 200):
    with open_db() as db:
        use_paper = db.get("settings", {}).get("use_paper", True)
    return await alpaca_trading.get_orders(use_paper, status=status, limit=limit)

@app.post("/orders/place")
async def place_order(req: OrderRequest):
    with open_db() as db:
        s = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **s}
        use_paper = s.get("use_paper", True)
        webhook = s.get("discord_webhook")
    acct = await alpaca_trading.get_account(use_paper)
    equity = float(acct.get("equity", 0))
    qty = float(req.qty) if req.qty else 0.0

    if s.get("size_mode") == "risk_pct" and req.entry_price:
        risk_perc = max(0.1, float(s.get("risk_per_trade_pct", 1.0))) / 100.0
        sl_perc = max(0.1, float(s.get("risk_sl_pct", 1.5))) / 100.0
        dollars_at_risk = equity * risk_perc
        stop_distance = float(req.entry_price) * sl_perc
        qty = max(1, int(dollars_at_risk / max(0.01, stop_distance)))

    order = await alpaca_trading.place_order(
        use_paper=use_paper,
        symbol=req.symbol.upper(),
        qty=int(qty),
        side=req.side,
        type=req.type,
        time_in_force=req.time_in_force or "day",
        limit_price=req.limit_price,
        stop_price=req.stop_price,
    )

    sl_pct = float(req.sl_pct) if req.sl_pct is not None else float(s.get("risk_sl_pct", 1.5))
    tp_pct = float(req.tp_pct) if req.tp_pct is not None else float(s.get("risk_tp_pct", 3.0))

    if req.entry_price and qty:
        entry = float(req.entry_price)
        if req.side == "buy":
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)
        try:
            await alpaca_trading.place_brackets(use_paper, req.symbol.upper(), int(qty), sl, tp)
        except Exception as e:
            if webhook:
                await send_discord(f"Bracket failed: {e}", webhook)

    if webhook:
        await send_discord(
            f"Order placed: {req.side.upper()} {int(qty)} {req.symbol.upper()} ({req.type})",
            webhook,
        )
    return order

# ---------------------- PROXY / OHLC ----------------------

@app.get("/proxy/bars")
async def proxy_bars(symbol: str, tf: str = Query("1Min"), crypto: int = 0, limit: int = 500):
    try:
        if crypto:
            bars = await ohlc_provider.binance_klines(symbol, interval="1m", limit=limit)
        else:
            bars = await ohlc_provider.alpaca_bars(symbol, timeframe=tf, limit=limit)
        return bars
    except Exception as e:
        print(f"[proxy_bars] Fallback for {symbol}: {e}")
        price = 100.0
        bars = []
        for i in range(limit):
            price += random.uniform(-1.0, 1.0)
            o = price + random.uniform(-0.5, 0.5)
            h = max(o, price) + random.uniform(0.0, 0.5)
            l = min(o, price) - random.uniform(0.0, 0.5)
            bars.append({"t": i, "o": round(o, 2), "h": round(h, 2), "l": round(l, 2), "c": round(price, 2), "v": 0})
        return bars

@app.get("/overlays/{symbol}")
async def overlays(symbol: str, tf: str = Query("1Min"), crypto: int = 0, limit: int = 300):
    bars = await (ohlc_provider.binance_klines(symbol, "1m", limit) if crypto else ohlc_provider.alpaca_bars(symbol, tf, limit))
    return build_overlays_from_ohlc(bars)

@app.get("/overlays_expert/{symbol}")
async def overlays_expert(symbol: str, tf: str = Query("1Min"), crypto: int = 0, limit: int = 500):
    bars = await (ohlc_provider.binance_klines(symbol, "1m", limit) if crypto else ohlc_provider.alpaca_bars(symbol, tf, limit))
    return build_expert_overlays(bars)

# ---------------------- SIGNAL / AI ----------------------

@app.post("/signal/{symbol}", response_model=Signal)
async def compute_signal(symbol: str, heavy: bool = False):
    sym = symbol.upper()
    with open_db() as db:
        settings = db.get("settings", {})
        use_crypto = settings.get("use_crypto", False)
        webhook = settings.get("discord_webhook")
        merged = {**DEFAULT_SETTINGS, **settings}
        horizon_min = int(merged.get("horizon_min_days", 5))
        horizon_max = int(merged.get("horizon_max_days", 30))
        allow_long = bool(merged.get("allow_long_horizon", True))
        tp_pct = float(merged.get("risk_tp_pct", 3.0))

    bars = await (ohlc_provider.binance_klines(sym, "1m", 400) if use_crypto else ohlc_provider.alpaca_bars(sym, "1Min", 400))
    closes = [float(b["c"]) for b in bars] if bars else []

    ml = infer_ensemble(bars) if bars else {"class": "unknown", "probs": {}}

    _, _, spark = await fetch_quote_and_spark(sym, use_crypto)
    series_for_signal = spark if spark else closes
    action, confidence, reasoning = infer_signal(series_for_signal, heavy)

    eta_min, eta_max, horizon_label = estimate_horizon_days(closes, tp_pct, horizon_min, horizon_max, allow_long)

    if confidence >= 0.7:
        conf_tag = "HIGH"; conf_emoji = "🟢"
    elif confidence >= 0.4:
        conf_tag = "MED"; conf_emoji = "🟡"
    else:
        conf_tag = "LOW"; conf_emoji = "🔴"

    if action == "buy":
        action_emoji = "🟢"
    elif action == "sell":
        action_emoji = "🔴"
    else:
        action_emoji = "🟦"

    ml_class = ml.get("class", "unknown")
    ml_prob = max(ml.get("probs", {}).values()) if ml.get("probs") else 0.0

    if webhook:
        lines = [
            f"[Sched] {sym}: {action_emoji} Action: {action.upper()}",
            f"{conf_emoji} Confidence: {conf_tag} ({confidence:.2f})  |  ML={ml_class} (p={ml_prob:.2f})",
            f"📈 Horizon: {eta_min}–{eta_max} trading days ({horizon_label})",
            f"📝 Reasoning: {reasoning}",
        ]
        await send_discord("\n".join(lines), webhook)

    reason_str = (
        f"ML:{ml_class} p={ml_prob:.2f}; "
        f"Horizon={eta_min}-{eta_max}d ({horizon_label}); "
        f"Heuristic: {reasoning}"
    )
    return Signal(symbol=sym, action=action, confidence=confidence, reasoning=reason_str)

def estimate_horizon_days(closes: List[float], tp_pct: float, min_days: int, max_days: int, allow_long: bool) -> tuple[int, int, str]:
    if len(closes) < 10 or tp_pct <= 0:
        return min_days, max_days, "unknown"

    rets: List[float] = []
    for i in range(1, len(closes)):
        p0, p1 = closes[i - 1], closes[i]
        if p0:
            rets.append(abs((p1 - p0) / p0))

    if not rets:
        return min_days, max_days, "unknown"

    avg_vol = statistics.mean(rets)
    if avg_vol <= 0:
        return min_days, max_days, "unknown"

    target_move = tp_pct / 100.0
    est_days = target_move / avg_vol
    est_days = max(1.0, min(est_days, 120.0))

    center = int(round(est_days))
    eta_min = max(1, center - 2)
    eta_max = center + 2

    if not allow_long:
        eta_min = max(eta_min, min_days)
        eta_max = min(eta_max, max_days)

    if eta_max <= 7:
        label = "weekly"
    elif eta_max <= 30:
        label = "monthly"
    else:
        label = "long"
    return eta_min, eta_max, label

@app.get("/ai/mtf/{symbol}")
async def ai_mtf(symbol: str):
    sym = symbol.upper()
    with open_db() as db:
        s = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **s}
        use_crypto = s.get("use_crypto", False)

    if use_crypto:
        bars_1m = await ohlc_provider.binance_klines(sym, "1m", 300)
        bars_5m = await ohlc_provider.binance_klines(sym, "5m", 300)
        bars_1h = await ohlc_provider.binance_klines(sym, "1h", 300)
    else:
        bars_1m = await ohlc_provider.alpaca_bars(sym, "1Min", 300)
        bars_5m = await ohlc_provider.alpaca_bars(sym, "5Min", 300)
        bars_1h = await ohlc_provider.alpaca_bars(sym, "1Hour", 300)

    ml_1m = infer_ensemble(bars_1m)
    ml_5m = infer_ensemble(bars_5m)
    ml_1h = infer_ensemble(bars_1h)

    return {"symbol": sym, "frames": {"1m": ml_1m, "5m": ml_5m, "1h": ml_1h}}

@app.post("/ai/trend/{symbol}")
async def ai_trend(symbol: str, heavy: bool = False, send_to_discord: bool = True):
    sym = symbol.upper()

    with open_db() as db:
        settings = db.get("settings", {})
        merged = {**DEFAULT_SETTINGS, **settings}
        webhook = merged.get("discord_webhook")
        use_crypto = merged.get("use_crypto", False)

    bars = await (ohlc_provider.binance_klines(sym, "1m", 400) if use_crypto else ohlc_provider.alpaca_bars(sym, "1Min", 400))
    result = run_trend_engine(sym, bars, merged, heavy=heavy)

    if send_to_discord and webhook:
        action = result["action"]
        conf = result["confidence"]
        action_emoji = "🟢" if action == "buy" else "🔴" if action == "sell" else "🟦"
        conf_emoji = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
        msg = "\n".join(
            [
                f"[Trend] {sym}: {action_emoji} Action: {action.upper()}",
                f"{conf_emoji} Confidence: {conf:.2f} | ML={result['ml_class']} ({result['ml_prob']:.2f})",
                f"📈 Horizon: {result['eta_min_days']}–{result['eta_max_days']} trading days ({result['horizon']})",
                f"📝 Reasoning: {result['reasoning']}",
            ]
        )
        await send_discord(msg, webhook)

    return result

class TrendRequest(BaseModel):
    symbol: str
    tf: str = "1Min"
    crypto: int = 0
    limit: int = 300
    heavy: bool = False
    send_to_discord: bool = False

@app.post("/ai/trend")
async def ai_trend_hybrid(req: TrendRequest):
    sym = req.symbol.upper()
    with open_db() as db:
        s = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **s}
        settings_use_crypto = s.get("use_crypto", False)
        webhook = s.get("discord_webhook")

    use_crypto = bool(req.crypto) or settings_use_crypto

    if use_crypto:
        bars = await ohlc_provider.binance_klines(sym, "1m", req.limit)
    else:
        bars = await ohlc_provider.alpaca_bars(sym, req.tf, req.limit)

    if not bars:
        raise HTTPException(status_code=400, detail=f"No bars for {sym}")

    closes = [float(b["c"]) for b in bars]
    ml = infer_ensemble(bars)
    action, confidence, reasoning = infer_signal(closes, heavy=req.heavy)

    if len(closes) >= 2:
        diff = closes[-1] - closes[0]
        if abs(diff) < 0.001 * max(1.0, abs(closes[0])):
            direction = "range"
        else:
            direction = "up" if diff > 0 else "down"
    else:
        direction = "unknown"

    final_signal = action.upper()
    final_confidence = float(confidence) * 100.0

    result = {
        "symbol": sym,
        "direction": direction,
        "final_signal": final_signal,
        "final_confidence": final_confidence,
        "reasoning": reasoning,
        "ml_class": ml.get("class"),
        "ml_probs": ml.get("probs", {}),
    }

    if req.send_to_discord and webhook:
        msg = f"[Trend] {sym}: {final_signal} ({final_confidence:.1f}%) dir={direction}"
        await send_discord(msg, webhook)

    return result

@app.post("/ai/trend/snapshot")
async def ai_trend_snapshot(symbol: str = Form(...), analysis: str = Form(...), file: UploadFile = File(None)):
    sym = symbol.upper()
    try:
        parsed = json.loads(analysis)
    except Exception:
        parsed = None

    with open_db() as db:
        s = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **s}
        webhook = s.get("discord_webhook")

    if not webhook:
        return {"ok": False, "error": "Discord webhook not configured."}

    content = f"[Snapshot] {sym} trend analysis"
    if parsed:
        fs = parsed.get("final_signal") or parsed.get("signal") or ""
        fc = parsed.get("final_confidence") or parsed.get("confidence")
        direction = parsed.get("direction") or ""
        content += f"\nSignal: {fs} ({fc}) dir={direction}"

    files = None
    data = {"content": content}
    try:
        if file is not None:
            bytes_ = await file.read()
            files = {"file": (file.filename or "chart.png", bytes_, file.content_type or "image/png")}
        resp = requests.post(webhook, data=data, files=files, timeout=20)
        if resp.status_code >= 300:
            return {"ok": False, "error": f"Discord error {resp.status_code}: {resp.text}"}
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------- ALERTS ----------------------

class AlertCreate(BaseModel):
    symbol: str
    direction: str  # "above" or "below"
    price: float
    note: Optional[str] = None

class AlertRule(BaseModel):
    id: str
    symbol: str
    direction: str
    price: float
    enabled: bool = True
    note: Optional[str] = None

@app.get("/alerts", response_model=List[AlertRule])
async def list_alerts():
    with open_db() as db:
        raw = db.get("alerts", {})

    rules: List[AlertRule] = []
    for aid, a in raw.items():
        try:
            rules.append(
                AlertRule(
                    id=aid,
                    symbol=a.get("symbol", "").upper(),
                    direction=a.get("direction", "above"),
                    price=float(a.get("price", 0.0)),
                    enabled=bool(a.get("enabled", True)),
                    note=a.get("note"),
                )
            )
        except Exception:
            continue
    return rules

@app.post("/alerts", response_model=AlertRule)
async def create_alert(payload: AlertCreate):
    sym = payload.symbol.upper()
    direction = payload.direction.lower()
    if direction not in ("above", "below"):
        raise HTTPException(status_code=400, detail="direction must be 'above' or 'below'")

    alert_id = str(uuid.uuid4())
    alert_data = {"symbol": sym, "direction": direction, "price": float(payload.price), "enabled": True, "note": payload.note}

    with open_db() as db:
        alerts = db.get("alerts", {})
        alerts[alert_id] = alert_data
        db["alerts"] = alerts

    return AlertRule(id=alert_id, **alert_data)

@app.post("/alerts/{alert_id}/toggle", response_model=AlertRule)
async def toggle_alert(alert_id: str):
    with open_db() as db:
        alerts = db.get("alerts", {})
        if alert_id not in alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        alert = alerts[alert_id]
        alert["enabled"] = not bool(alert.get("enabled", True))
        alerts[alert_id] = alert
        db["alerts"] = alerts

    return AlertRule(
        id=alert_id,
        symbol=alert.get("symbol", "").upper(),
        direction=alert.get("direction", "above"),
        price=float(alert.get("price", 0.0)),
        enabled=bool(alert.get("enabled", True)),
        note=alert.get("note"),
    )

@app.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    with open_db() as db:
        alerts = db.get("alerts", {})
        existed = alerts.pop(alert_id, None)
        db["alerts"] = alerts
    if not existed:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"ok": True}

# ---------------------- BACKTESTING ----------------------

class BacktestRequest(BaseModel):
    symbol: str
    tf: str = "1Min"
    lookback_bars: int = 1000
    use_crypto: bool = False

@app.post("/backtest/basic")
async def backtest_basic(req: BacktestRequest):
    sym = req.symbol.upper()
    tf = req.tf or "1Min"
    lookback = max(200, int(req.lookback_bars or 1000))

    with open_db() as db:
        s = db.get("settings", {})
        s = {**DEFAULT_SETTINGS, **s}
        settings_use_crypto = s.get("use_crypto", False)

    use_crypto = bool(req.use_crypto) or settings_use_crypto

    if use_crypto:
        bars = await ohlc_provider.binance_klines(sym, "1m", lookback)
    else:
        bars = await ohlc_provider.alpaca_bars(sym, tf, lookback)

    if not bars or len(bars) < 50:
        return {"ok": False, "error": f"Not enough bars for {sym}", "symbol": sym}

    closes = [float(b["c"]) for b in bars]
    times = [b.get("t", i) for i, b in enumerate(bars)]

    equity = 1.0
    equity_curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    position = 0
    entry_price: Optional[float] = None

    window = min(100, len(closes) // 3 or 50)

    for i in range(window, len(closes)):
        window_closes = closes[i - window : i]
        action, conf, _ = infer_signal(window_closes, heavy=False)
        price_now = closes[i]

        if action == "sell" and position == 1 and entry_price is not None:
            ret = (price_now - entry_price) / entry_price
            equity *= 1.0 + ret
            trades.append(
                {
                    "side": "SELL",
                    "entry_price": entry_price,
                    "exit_price": price_now,
                    "return_pct": ret * 100.0,
                    "t_entry": times[i - 1],
                    "t_exit": times[i],
                }
            )
            position = 0
            entry_price = None
        elif action == "buy" and position == 0:
            position = 1
            entry_price = price_now

        equity_curve.append({"t": times[i], "equity": equity})

    if position == 1 and entry_price is not None:
        price_last = closes[-1]
        ret = (price_last - entry_price) / entry_price
        equity *= 1.0 + ret
        trades.append(
            {
                "side": "SELL",
                "entry_price": entry_price,
                "exit_price": price_last,
                "return_pct": ret * 100.0,
                "t_entry": times[-2] if len(times) >= 2 else times[-1],
                "t_exit": times[-1],
            }
        )

    total_trades = len(trades)
    wins = sum(1 for tr in trades if tr["return_pct"] > 0)
    losses = sum(1 for tr in trades if tr["return_pct"] < 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_return_pct = (equity - 1.0) * 100.0

    return {
        "ok": True,
        "symbol": sym,
        "use_crypto": use_crypto,
        "tf": tf,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "equity_curve": equity_curve,
        "trades": trades,
    }

# ---------------------- WS BROADCAST ----------------------

@app.websocket("/ws")
async def ws_root(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        cycle = 0
        while True:
            await asyncio.sleep(WS_INTERVAL)
            with open_db() as db:
                sym = db.get("current_symbol", "AAPL")
                settings = db.get("settings", {})
                use_crypto = settings.get("use_crypto", False)

            price, change_pct, spark = await fetch_quote_and_spark(sym, use_crypto)
            action, confidence, reasoning = infer_signal(spark)

            payload: Dict[str, Any] = {
                "type": "tick",
                "symbol": sym,
                "price": float(price or 0),
                "change_pct": change_pct,
                "sparkline": spark[-100:] if spark else [],
                "signal": {"action": action, "confidence": confidence, "reasoning": reasoning},
            }

            cycle = (cycle + 1) % 3
            if cycle == 0:
                bars = await (ohlc_provider.binance_klines(sym, "1m", 300) if use_crypto else ohlc_provider.alpaca_bars(sym, "1Min", 300))
                payload["overlays"] = build_overlays_from_ohlc(bars)
            elif cycle == 1:
                bars = await (ohlc_provider.binance_klines(sym, "1m", 400) if use_crypto else ohlc_provider.alpaca_bars(sym, "1Min", 400))
                payload["overlays_expert"] = build_expert_overlays(bars)

            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)

# ---------------------- SCHEDULERS ----------------------

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(signal_scheduler())
    asyncio.create_task(power_scheduler())

async def signal_scheduler():
    await asyncio.sleep(1)
    while True:
        with open_db() as db:
            s = db.get("settings", {})
            s = {**DEFAULT_SETTINGS, **s}
            interval = int(s.get("signal_schedule_min", 0) or 0)
            webhook = s.get("discord_webhook")
            wl = db.get("watchlist", {})
            use_crypto = s.get("use_crypto", False)
            alerts = db.get("alerts", {})

        if interval <= 0 or not wl or not webhook:
            await asyncio.sleep(5)
            continue

        for sym in list(wl.keys()):
            try:
                price, _, spark = await fetch_quote_and_spark(sym, use_crypto)
                bars = await (ohlc_provider.binance_klines(sym, "1m", 400) if use_crypto else ohlc_provider.alpaca_bars(sym, "1Min", 400))
                ml = infer_ensemble(bars) if bars else {"class": "unknown", "probs": {}}
                action, conf, reason = infer_signal(spark or [])

                if conf >= 0.7:
                    conf_tag = "HIGH"; conf_emoji = "🟢"
                elif conf >= 0.4:
                    conf_tag = "MED"; conf_emoji = "🟡"
                else:
                    conf_tag = "LOW"; conf_emoji = "🔴"

                if action == "buy":
                    action_emoji = "🟢"
                elif action == "sell":
                    action_emoji = "🔴"
                else:
                    action_emoji = "🟦"

                ml_class = ml.get("class", "unknown")

                sched_msg = "\n".join(
                    [
                        f"[Sched] {sym}: {action_emoji} {action.upper()}",
                        f"{conf_emoji} Confidence: {conf:.2f} ({conf_tag})  |  ML={ml_class}",
                        f"💵 Last price: {price:.2f}",
                        f"📝 Reasoning: {reason}",
                    ]
                )
                await send_discord(sched_msg, webhook)

                triggered_ids: List[str] = []
                for aid, alert in alerts.items():
                    if not alert.get("enabled", True):
                        continue
                    if alert.get("symbol", "").upper() != sym:
                        continue

                    direction = alert.get("direction", "above").lower()
                    level = float(alert.get("price", 0.0))

                    hit = False
                    if direction == "above" and price >= level:
                        hit = True
                    elif direction == "below" and price <= level:
                        hit = True

                    if hit:
                        triggered_ids.append(aid)
                        dir_emoji = "📈" if direction == "above" else "📉"
                        note = alert.get("note")
                        msg = (
                            f"⏰ Alert hit {dir_emoji} {sym} "
                            f"{'≥' if direction == 'above' else '≤'} {level:.2f} "
                            f"(last={price:.2f})"
                        )
                        if note:
                            msg += f"\n📝 {note}"
                        await send_discord(msg, webhook)

                if triggered_ids:
                    with open_db() as db:
                        alerts_db = db.get("alerts", {})
                        for aid in triggered_ids:
                            if aid in alerts_db:
                                alerts_db[aid]["enabled"] = False
                        db["alerts"] = alerts_db

            except Exception as e:
                try:
                    await send_discord(f"[Sched] {sym} failed: {e}", webhook)
                except Exception:
                    pass

        await asyncio.sleep(interval * 60)

async def power_scheduler():
    """
    Keeps your existing scheduler structure. (Your universe scan / anomalies logic can stay here.)
    """
    await asyncio.sleep(3)
    while True:
        await asyncio.sleep(5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
