import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'node:path';
import fs from 'node:fs/promises';
import { spawn } from 'node:child_process';

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1600,
    height: 980,
    webPreferences: {
      preload: path.join(process.cwd(), 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });
  win.loadFile(path.join(process.cwd(), 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ------------ Helpers ------------
async function readConfig() {
  const cfgPath = path.join(process.cwd(), 'config.json');
  const raw = await fs.readFile(cfgPath, 'utf-8');
  const cfg = JSON.parse(raw);
  cfg.workspaceAbs = path.resolve(process.cwd(), cfg.workspaceRoot);
  cfg.composeAbs = path.resolve(process.cwd(), cfg.composeFile);
  cfg.orchestrator.scriptAbs = path.resolve(process.cwd(), cfg.orchestrator.script);
  cfg.orchestrator.planAbs = path.resolve(process.cwd(), cfg.orchestrator.plan);
  cfg.ai.apiKey = process.env[cfg.ai.apiKeyEnv] || null; // read env var for API key
  return cfg;
}

function spawnStream(cmd, args, cwd, onData, onExit) {
  const child = spawn(cmd, args, { cwd, shell: false });
  child.stdout.on('data', d => onData(d.toString()))
  child.stderr.on('data', d => onData(d.toString()))
  child.on('close', code => onExit(code ?? 0));
  return child;
}

async function detectPython() {
  return new Promise((resolve) => {
    const exe = process.platform === 'win32' ? 'where' : 'which';
    const test = spawn(exe, ['python']);
    test.on('close', (code) => {
      if (code === 0) return resolve('python');
      if (process.platform === 'win32') return resolve('py');
      resolve('python');
    });
  });
}

async function detectCompose() {
  return new Promise((resolve) => {
    const exe = process.platform === 'win32' ? 'where' : 'which';
    const test = spawn(exe, ['docker']);
    test.on('close', (code) => {
      if (code === 0) return resolve({ bin: 'docker', args: ['compose'] });
      resolve({ bin: 'docker-compose', args: [] });
    });
  });
}

// ------------ IPC: Config, FS, Proc, Compose, AI ------------
ipcMain.handle('cfg:get', async () => readConfig());

ipcMain.handle('fs:list', async (_e, rel) => {
  const cfg = await readConfig();
  const root = path.resolve(cfg.workspaceAbs, rel || '.');
  const ents = await fs.readdir(root, { withFileTypes: true });
  return ents.map(e => ({
    name: e.name,
    isDir: e.isDirectory(),
    path: path.relative(cfg.workspaceAbs, path.join(root, e.name))
  })).sort((a,b)=> (a.isDir===b.isDir ? a.name.localeCompare(b.name) : (a.isDir?-1:1)));
});

ipcMain.handle('fs:read', async (_e, rel) => {
  const cfg = await readConfig();
  const p = path.resolve(cfg.workspaceAbs, rel);
  return await fs.readFile(p, 'utf-8');
});

ipcMain.handle('fs:write', async (_e, rel, content) => {
  const cfg = await readConfig();
  const p = path.resolve(cfg.workspaceAbs, rel);
  await fs.writeFile(p, content, 'utf-8');
  return true;
});

ipcMain.handle('proc:run', async (_e, payload) => {
  const { cmd, args = [], cwdRel = '.', env = {} } = payload;
  const cfg = await readConfig();
  const cwd = path.resolve(cfg.workspaceAbs, cwdRel);
  return new Promise((resolve) => {
    const id = Math.random().toString(36).slice(2);
    win.webContents.send('proc:started', { id, cmd, args, cwd });
    spawnStream(cmd, args, cwd, (line) => {
      win.webContents.send('proc:data', { id, line });
    }, (code) => {
      win.webContents.send('proc:exit', { id, code });
      resolve({ id, code });
    });
  });
});

ipcMain.handle('orchestrator:run', async () => {
  const cfg = await readConfig();
  let py = cfg.orchestrator.python;
  if (py === 'auto') py = await detectPython();
  const args = [];
  if (py === 'py') args.push('-3');
  args.push(cfg.orchestrator.scriptAbs, '--plan', cfg.orchestrator.planAbs);
  const cwd = path.dirname(cfg.orchestrator.scriptAbs);
  return new Promise((resolve) => {
    const id = 'orch-' + Math.random().toString(36).slice(2);
    win.webContents.send('proc:started', { id, cmd: py, args, cwd });
    spawnStream(py, args, cwd, (line) => win.webContents.send('proc:data', { id, line }), (code) => {
      win.webContents.send('proc:exit', { id, code });
      resolve({ id, code });
    });
  });
});

ipcMain.handle('compose:up', async () => {
  const cfg = await readConfig();
  const { bin, args: base } = await detectCompose();
  const args = [...base, '-f', cfg.composeAbs, 'up', '-d', 'julia-backend'];
  const cwd = path.dirname(cfg.composeAbs);
  return new Promise((resolve) => {
    const id = 'dcup-' + Math.random().toString(36).slice(2);
    win.webContents.send('proc:started', { id, cmd: bin, args, cwd });
    spawnStream(bin, args, cwd, (line) => win.webContents.send('proc:data', { id, line }), (code) => {
      win.webContents.send('proc:exit', { id, code });
      resolve({ id, code });
    });
  });
});

ipcMain.handle('compose:down', async () => {
  const cfg = await readConfig();
  const { bin, args: base } = await detectCompose();
  const args = [...base, '-f', cfg.composeAbs, 'down'];
  const cwd = path.dirname(cfg.composeAbs);
  return new Promise((resolve) => {
    const id = 'dcdown-' + Math.random().toString(36).slice(2);
    win.webContents.send('proc:started', { id, cmd: bin, args, cwd });
    spawnStream(bin, args, cwd, (line) => win.webContents.send('proc:data', { id, line }), (code) => {
      win.webContents.send('proc:exit', { id, code });
      resolve({ id, code });
    });
  });
});

ipcMain.handle('ai:chat', async (_e, payload) => {
  const cfg = await readConfig();
  const { messages, stream } = payload;
  if (!cfg.ai.apiKey) throw new Error(`Missing API key; set ${cfg.ai.apiKeyEnv}`);
  const url = cfg.ai.baseUrl.replace(/\/$/, '') + '/chat/completions';
  const headers = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${cfg.ai.apiKey}` };
  const body = JSON.stringify({ model: cfg.ai.model, temperature: cfg.ai.temperature, messages, stream: !!stream });
  // Stream in main and relay to renderer as events
  const id = 'chat-' + Math.random().toString(36).slice(2);
  win.webContents.send('ai:chat:open', { id });
  const res = await fetch(url, { method: 'POST', headers, body });
  if (!stream) {
    const json = await res.json();
    win.webContents.send('ai:chat:data', { id, chunk: JSON.stringify(json) });
    win.webContents.send('ai:chat:close', { id });
    return { id };
  }
  const reader = res.body.getReader();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    win.webContents.send('ai:chat:data', { id, chunk: new TextDecoder().decode(value) });
  }
  win.webContents.send('ai:chat:close', { id });
  return { id };
});
