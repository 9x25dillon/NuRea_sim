import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('NuRea', {
  cfg: { get: () => ipcRenderer.invoke('cfg:get') },
  fs: {
    list: (rel) => ipcRenderer.invoke('fs:list', rel),
    read: (rel) => ipcRenderer.invoke('fs:read', rel),
    write: (rel, content) => ipcRenderer.invoke('fs:write', rel, content),
  },
  proc: {
    run: (payload) => ipcRenderer.invoke('proc:run', payload),
    onStarted: (cb) => ipcRenderer.on('proc:started', (_e, d) => cb(d)),
    onData: (cb) => ipcRenderer.on('proc:data', (_e, d) => cb(d)),
    onExit: (cb) => ipcRenderer.on('proc:exit', (_e, d) => cb(d)),
  },
  orchestrator: { run: () => ipcRenderer.invoke('orchestrator:run') },
  compose: { up: () => ipcRenderer.invoke('compose:up'), down: () => ipcRenderer.invoke('compose:down') },
  ai: {
    chat: (payload) => ipcRenderer.invoke('ai:chat', payload),
    onOpen: (cb) => ipcRenderer.on('ai:chat:open', (_e, d) => cb(d)),
    onData: (cb) => ipcRenderer.on('ai:chat:data', (_e, d) => cb(d)),
    onClose: (cb) => ipcRenderer.on('ai:chat:close', (_e, d) => cb(d)),
  }
});
