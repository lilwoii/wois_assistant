// electron/preload.js
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("woiAPI", {
  // you can add IPC bridges later if needed
});
