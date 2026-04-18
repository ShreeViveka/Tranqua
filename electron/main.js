/**
 * electron/main.js — Tranqua Desktop App Main Process
 * =====================================================
 * This is the heart of the Electron app. It:
 *   1. Creates the app window
 *   2. Launches the Python FastAPI backend as a hidden process
 *   3. Launches the data collector as a hidden process
 *   4. Manages app lifecycle (quit, minimize, tray)
 *   5. Handles auto-updates
 *
 * Architecture:
 *   Electron (this file) → manages everything
 *       ↓ spawns
 *   Python FastAPI backend (localhost:8000) → hidden
 *       ↓ spawns
 *   Data Collector → hidden background thread
 *       ↓
 *   React frontend → shown in Electron window
 */

const { app, BrowserWindow, Tray, Menu, ipcMain,
        shell, dialog, nativeTheme } = require('electron');
const path    = require('path');
const { spawn, execFile } = require('child_process');
const fs      = require('fs');
const log     = require('electron-log');
const Store   = require('electron-store');

// ── Logging setup ─────────────────────────────────────────────────────────────
log.transports.file.level = 'info';
log.transports.console.level = 'debug';
log.info('Tranqua starting...');

// ── Persistent settings ───────────────────────────────────────────────────────
const store = new Store({
  defaults: {
    windowBounds    : { width: 430, height: 820 },
    minimizeToTray  : true,
    firstLaunch     : true,
  }
});

// ── State ─────────────────────────────────────────────────────────────────────
let mainWindow    = null;
let tray          = null;
let backendProcess = null;
let collectorProcess = null;
let backendReady  = false;

const isDev = process.argv.includes('--dev');
const BACKEND_PORT = 8000;
const BACKEND_URL  = `http://localhost:${BACKEND_PORT}`;

// ── Paths ─────────────────────────────────────────────────────────────────────
// In development: use source files
// In production:  use bundled files inside app.asar/extraResources

const ROOT = isDev
  ? path.join(__dirname, '..')                          // D:\nlp\Tranqua
  : path.join(process.resourcesPath);                   // inside .exe

const PYTHON = isDev
  ? 'python'                                            // system python in dev
  : getPythonPath();                                    // bundled python in prod

const BACKEND_SCRIPT   = path.join(ROOT, 'backend', 'main.py');
const COLLECTOR_SCRIPT = path.join(ROOT, 'collector', 'data_collector.py');
const FRONTEND_BUILD   = path.join(ROOT, 'frontend', 'build', 'index.html');

log.info(`Root path    : ${ROOT}`);
log.info(`Python path  : ${PYTHON}`);
log.info(`Backend      : ${BACKEND_SCRIPT}`);
log.info(`Dev mode     : ${isDev}`);


// ════════════════════════════════════════════════════════════════════════════
// PYTHON PATH DETECTION
// ════════════════════════════════════════════════════════════════════════════

function getPythonPath() {
  // Look for bundled Python in extraResources/python_dist
  const bundled = path.join(process.resourcesPath, 'python_dist', 'python.exe');
  if (fs.existsSync(bundled)) {
    log.info(`Using bundled Python: ${bundled}`);
    return bundled;
  }

  // Fall back to system Python
  const candidates = [
    'python',
    'python3',
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python312', 'python.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'python.exe'),
    'C:\\Python312\\python.exe',
    'C:\\Python311\\python.exe',
  ];

  for (const p of candidates) {
    try {
      require('child_process').execSync(`"${p}" --version`, { stdio: 'ignore' });
      log.info(`Using system Python: ${p}`);
      return p;
    } catch { /* try next */ }
  }

  log.error('Python not found!');
  return 'python';   // will fail gracefully
}


// ════════════════════════════════════════════════════════════════════════════
// BACKEND LAUNCHER
// ════════════════════════════════════════════════════════════════════════════

function startBackend() {
  log.info('Starting Python backend...');

  const env = {
    ...process.env,
    PYTHONPATH: ROOT,
    PYTHONUNBUFFERED: '1',
  };

  backendProcess = spawn(PYTHON, [
    '-m', 'uvicorn',
    'backend.main:app',
    '--host', '127.0.0.1',
    '--port', String(BACKEND_PORT),
    '--no-access-log',
  ], {
    cwd       : ROOT,
    env       : env,
    stdio     : ['ignore', 'pipe', 'pipe'],
    detached  : false,
    windowsHide: true,    // hide console window on Windows
  });

  backendProcess.stdout.on('data', (data) => {
    const msg = data.toString().trim();
    log.info(`[Backend] ${msg}`);
    if (msg.includes('Application startup complete')) {
      backendReady = true;
      log.info('Backend is ready!');
      // Load the app once backend is ready
      if (mainWindow) loadApp();
    }
  });

  backendProcess.stderr.on('data', (data) => {
    log.warn(`[Backend ERR] ${data.toString().trim()}`);
  });

  backendProcess.on('error', (err) => {
    log.error(`[Backend] Failed to start: ${err.message}`);
    showBackendError(err.message);
  });

  backendProcess.on('close', (code) => {
    log.info(`[Backend] Process exited with code ${code}`);
    if (code !== 0 && mainWindow && !app.isQuitting) {
      // Auto-restart backend if it crashes
      log.info('[Backend] Restarting in 3 seconds...');
      setTimeout(startBackend, 3000);
    }
  });
}

function startCollector() {
  log.info('Starting data collector...');

  collectorProcess = spawn(PYTHON, [COLLECTOR_SCRIPT], {
    cwd       : ROOT,
    env       : { ...process.env, PYTHONPATH: ROOT, PYTHONUNBUFFERED: '1' },
    stdio     : ['ignore', 'pipe', 'pipe'],
    detached  : false,
    windowsHide: true,
  });

  collectorProcess.stdout.on('data', (data) => {
    log.info(`[Collector] ${data.toString().trim()}`);
  });

  collectorProcess.stderr.on('data', (data) => {
    // Collector writes info to stderr too — not always errors
    const msg = data.toString().trim();
    if (msg) log.debug(`[Collector] ${msg}`);
  });

  collectorProcess.on('error', (err) => {
    log.error(`[Collector] Failed to start: ${err.message}`);
  });

  collectorProcess.on('close', (code) => {
    log.info(`[Collector] Exited with code ${code}`);
  });
}

function stopAll() {
  log.info('Stopping all processes...');

  if (backendProcess) {
    backendProcess.kill('SIGTERM');
    backendProcess = null;
  }
  if (collectorProcess) {
    collectorProcess.kill('SIGTERM');
    collectorProcess = null;
  }
}


// ════════════════════════════════════════════════════════════════════════════
// WINDOW CREATION
// ════════════════════════════════════════════════════════════════════════════

function createWindow() {
  const { width, height } = store.get('windowBounds');

  mainWindow = new BrowserWindow({
    width,
    height,
    minWidth   : 380,
    minHeight  : 600,
    maxWidth   : 500,
    title      : 'Tranqua',
    icon       : path.join(__dirname, 'assets', 'icon.png'),

    // Make it look like a real app
    frame           : true,
    titleBarStyle   : 'default',
    backgroundColor : '#FDF8F2',   // matches app cream background

    webPreferences: {
      preload           : path.join(__dirname, 'preload.js'),
      nodeIntegration   : false,   // security — never enable this
      contextIsolation  : true,    // security — always keep true
      webSecurity       : true,
    },
  });

  // Show loading screen while backend starts
  mainWindow.loadFile(path.join(__dirname, 'loading.html'));

  // Hide menu bar (not needed for this app)
  mainWindow.setMenuBarVisibility(false);

  // Open DevTools in dev mode
  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: 'right' });
  }

  // Save window size on resize
  mainWindow.on('resize', () => {
    const [w, h] = mainWindow.getSize();
    store.set('windowBounds', { width: w, height: h });
  });

  // Minimize to tray instead of closing
  mainWindow.on('close', (e) => {
    if (!app.isQuitting && store.get('minimizeToTray')) {
      e.preventDefault();
      mainWindow.hide();
      showTrayNotification();
    }
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

function loadApp() {
  if (!mainWindow) return;

  if (isDev) {
    // In dev: load from React dev server
    mainWindow.loadURL('http://localhost:3000');
  } else {
    // In production: load built React files
    mainWindow.loadFile(FRONTEND_BUILD);
  }

  log.info('App UI loaded.');
}

// Wait for backend to be ready before loading the app
function waitForBackend(maxWait = 30000) {
  const http    = require('http');
  const start   = Date.now();
  const interval= setInterval(() => {
    http.get(`${BACKEND_URL}/health`, (res) => {
      if (res.statusCode === 200) {
        clearInterval(interval);
        backendReady = true;
        log.info('Backend health check passed!');
        if (mainWindow) loadApp();
      }
    }).on('error', () => {
      if (Date.now() - start > maxWait) {
        clearInterval(interval);
        log.error('Backend did not start in time!');
        showBackendError('Backend took too long to start.');
      }
    });
  }, 1000);  // check every second
}


// ════════════════════════════════════════════════════════════════════════════
// SYSTEM TRAY
// ════════════════════════════════════════════════════════════════════════════

function createTray() {
  const iconPath = path.join(__dirname, 'assets', 'tray-icon.png');
  tray = new Tray(fs.existsSync(iconPath) ? iconPath :
                  path.join(__dirname, 'assets', 'icon.png'));

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Open Tranqua',
      click: () => { mainWindow?.show(); mainWindow?.focus(); }
    },
    { type: 'separator' },
    {
      label: 'Collector Status',
      enabled: false,
      label: collectorProcess ? '🟢 Collector running' : '🔴 Collector stopped'
    },
    { type: 'separator' },
    {
      label: 'Quit Tranqua',
      click: () => {
        app.isQuitting = true;
        app.quit();
      }
    }
  ]);

  tray.setToolTip('Tranqua — Mental Health Tracker');
  tray.setContextMenu(contextMenu);
  tray.on('click', () => {
    mainWindow?.show();
    mainWindow?.focus();
  });
}

function showTrayNotification() {
  if (tray) {
    tray.displayBalloon({
      title  : 'Tranqua is still running',
      content: 'Data collection continues in the background. Right-click the tray icon to quit.',
      iconType: 'info',
    });
  }
}


// ════════════════════════════════════════════════════════════════════════════
// ERROR HANDLING
// ════════════════════════════════════════════════════════════════════════════

function showBackendError(message) {
  dialog.showErrorBox(
    'Tranqua — Startup Error',
    `The Python backend failed to start.\n\n` +
    `Error: ${message}\n\n` +
    `Make sure Python 3.10+ is installed and run:\n` +
    `pip install -r requirements.txt`
  );
}

function showFirstLaunchSetup() {
  dialog.showMessageBox(mainWindow, {
    type   : 'info',
    title  : 'Welcome to Tranqua! 🌱',
    message: 'Setting up your personal tracker...',
    detail : [
      'Tranqua will:',
      '• Track your app usage silently in the background',
      '• Analyse your diary entries for mental state',
      '• Keep ALL your data local — nothing leaves your laptop',
      '',
      'Click OK to get started.',
    ].join('\n'),
    buttons: ['Get Started'],
  });
  store.set('firstLaunch', false);
}


// ════════════════════════════════════════════════════════════════════════════
// IPC — Communication between Electron and React
// ════════════════════════════════════════════════════════════════════════════

// React can call these from the frontend
ipcMain.handle('get-app-version',  () => app.getVersion());
ipcMain.handle('get-backend-url',  () => BACKEND_URL);
ipcMain.handle('is-backend-ready', () => backendReady);
ipcMain.handle('open-logs',        () => shell.openPath(log.transports.file.getFile().path));
ipcMain.handle('open-data-folder', () => {
  const dataDir = path.join(ROOT, 'data');
  if (fs.existsSync(dataDir)) shell.openPath(dataDir);
});
ipcMain.handle('minimize-to-tray', () => mainWindow?.hide());
ipcMain.handle('quit-app', () => {
  app.isQuitting = true;
  app.quit();
});


// ════════════════════════════════════════════════════════════════════════════
// APP LIFECYCLE
// ════════════════════════════════════════════════════════════════════════════

app.whenReady().then(async () => {
  log.info('Electron app ready.');

  // Set app user model ID for Windows notifications
  if (process.platform === 'win32') {
    app.setAppUserModelId('com.shreeviveka.tranqua');
  }

  // 1. Create window (shows loading screen)
  createWindow();

  // 2. Create system tray
  createTray();

  // 3. Start Python backend
  startBackend();

  // 4. Start data collector
  startCollector();

  // 5. Poll until backend is ready, then load app
  waitForBackend();

  // 6. First launch welcome
  if (store.get('firstLaunch')) {
    setTimeout(() => showFirstLaunchSetup(), 2000);
  }
});

// Quit when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// Clean up before quitting
app.on('before-quit', () => {
  log.info('App quitting — stopping all processes...');
  app.isQuitting = true;
  stopAll();
});

app.on('quit', () => {
  log.info('Tranqua closed.');
});

// Handle second instance (bring window to front instead of opening new one)
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}
