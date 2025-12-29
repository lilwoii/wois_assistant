// electron/main.js
const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let mainWindow;
let backendProcess;

const isDev = !app.isPackaged;

function startBackend() {
  if (backendProcess) return;

  const backendPath = path.join(__dirname, "..", "backend", "main.py");
  const cwd = path.join(__dirname, "..");

  backendProcess = spawn("python", [backendPath], {
    cwd,
    stdio: "inherit",
    shell: false
  });

  backendProcess.on("close", (code) => {
    console.log(`Backend exited with code ${code}`);
    backendProcess = null;
  });

  backendProcess.on("error", (err) => {
    console.error("Failed to start backend:", err);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1300,
    height: 800,
    backgroundColor: "#020617",
    show: false, // show when ready
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  const loadURL = async () => {
    if (isDev) {
      await mainWindow.loadURL("http://localhost:3000");
    } else {
      await mainWindow.loadFile(
        path.join(__dirname, "..", "frontend", "build", "index.html")
      );
    }
  };

  loadURL()
    .then(() => {
      mainWindow.show();
    })
    .catch((err) => {
      console.error("Failed to load renderer:", err);
    });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
  }
});
