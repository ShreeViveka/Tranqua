/**
 * electron/preload.js — Security Bridge
 * =======================================
 * This file runs in a sandboxed context between
 * Electron (Node.js) and the React frontend (browser).
 *
 * It exposes ONLY specific safe functions to React —
 * React cannot access Node.js directly (security).
 *
 * React uses these via: window.tranqua.getVersion() etc.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('tranqua', {
  // App info
  getVersion    : ()  => ipcRenderer.invoke('get-app-version'),
  getBackendUrl : ()  => ipcRenderer.invoke('get-backend-url'),
  isBackendReady: ()  => ipcRenderer.invoke('is-backend-ready'),

  // Actions
  openLogs      : ()  => ipcRenderer.invoke('open-logs'),
  openDataFolder: ()  => ipcRenderer.invoke('open-data-folder'),
  minimizeToTray: ()  => ipcRenderer.invoke('minimize-to-tray'),
  quitApp       : ()  => ipcRenderer.invoke('quit-app'),

  // Platform info
  platform      : process.platform,   // 'win32', 'darwin', 'linux'
  isElectron    : true,
});
