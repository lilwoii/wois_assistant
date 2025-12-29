// frontend/src/config.js
const isDev = !process.env.NODE_ENV || process.env.NODE_ENV === "development";

export const API_BASE_URL = isDev
  ? "http://localhost:8000"
  : "http://localhost:8000"; // change for production if needed

export const WS_URL = isDev
  ? "ws://localhost:8000/ws"
  : "ws://localhost:8000/ws";
