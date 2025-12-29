const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export async function getSettings() {
  const r = await fetch(`${API}/settings`);
  return r.json();
}

export async function saveSettings(s) {
  const r = await fetch(`${API}/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(s),
  });
  return r.json();
}

export async function addWatch(symbol) {
  const r = await fetch(`${API}/watchlist/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });
  return r.json();
}

export async function listWatch() {
  const r = await fetch(`${API}/watchlist`);
  return r.json();
}

export async function delWatch(symbol) {
  const r = await fetch(`${API}/watchlist/${symbol}`, { method: "DELETE" });
  return r.json();
}

export async function setSymbol(symbol) {
  const r = await fetch(`${API}/symbol/${symbol}`, { method: "POST" });
  return r.json();
}

export async function getSymbol() {
  const r = await fetch(`${API}/symbol`);
  return r.json();
}

export async function computeSignal(symbol, heavy = false) {
  const r = await fetch(
    `${API}/signal/${symbol}?heavy=${heavy ? "true" : "false"}`,
    { method: "POST" }
  );
  return r.json();
}

/**
 * 🧾 Portfolio helpers (used by Portfolio.jsx & TradeHistory.jsx)
 */

export async function getAccount() {
  const r = await fetch(`${API}/portfolio/account`);
  return r.json();
}

export async function getPositions() {
  const r = await fetch(`${API}/portfolio/positions`);
  return r.json();
}

export async function getOrders(status = "all", limit = 200) {
  const r = await fetch(
    `${API}/orders/history?status=${encodeURIComponent(
      status
    )}&limit=${encodeURIComponent(limit)}`
  );
  return r.json();
}

/**
 * 🛒 Order placement + AI training
 */

export async function placeOrder(body) {
  const r = await fetch(`${API}/orders/place`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

export async function trainLocal() {
  const r = await fetch(`${API}/train/local`, { method: "POST" });
  return r.json();
}

export async function trainRemote() {
  const r = await fetch(`${API}/train/remote`, { method: "POST" });
  return r.json();
}
export async function getMTF(symbol) {
  const r = await fetch(
    `${API}/ai/mtf/${encodeURIComponent(symbol)}`
  );
  return r.json();
}
