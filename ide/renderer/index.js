/* global monaco, Terminal, FitAddon */
const $ = (sel) => document.querySelector(sel);
const listEl = $('#file-list');
const resultsEl = $('#search-results');
const wsPathEl = $('#workspace-path');
const healthEl = $('#health');
const tabbar = $('#tabbar');
const saveBtn = $('#save');
const diffBtn = $('#diff-toggle');
const aiRefactorBtn = $('#ai-refactor');
const aiExplainBtn = $('#ai-explain');
const aiTestBtn = $('#ai-test');
const runPlanBtn = $('#btn-run-plan');
const backendUpBtn = $('#btn-backend-up');
const backendDownBtn = $('#btn-backend-down');
const searchInput = $('#search-input');
const searchBtn = $('#search-btn');
const chatLog = $('#chat-log');
const chatInput = $('#chat-input');
const chatSend = $('#chat-send');

let editor, diffEditor, fitAddon, term;
let openTabs = new Map(); // rel -> {model, viewState, dirty}
let activeRel = null;
let aiSession = [];

function logTerm(line){ if(!term) return; term.writeln((line||'').replace(/\r?\n$/, '')); }
function addChat(role, text){
  const div = document.createElement('div');
  div.className = 'msg '+role;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function boot(){
  const cfg = await window.NuRea.cfg.get();
  wsPathEl.textContent = cfg.workspaceAbs;
  await buildEditors();
  await refreshList();
  initTerminal();
  setInterval(checkHealth, 5000); checkHealth();
}

async function buildEditors(){
  editor = monaco.editor.create(document.getElementById('monaco'), {
    value: '', language: 'python', theme: 'vs-dark', automaticLayout: true,
    minimap: { enabled: false }, fontLigatures: true, fontSize: 13
  });
  diffEditor = monaco.editor.createDiffEditor(document.getElementById('diff'), {
    theme: 'vs-dark', readOnly: false, renderSideBySide: true, automaticLayout: true
  });
}

function extToLang(rel){
  const m = rel.match(/\.([^.]+)$/); const e = m?m[1].toLowerCase():'';
  return ({ js:'javascript', ts:'typescript', py:'python', jl:'julia', json:'json', md:'markdown', rb:'ruby', toml:'ini', yml:'yaml', yaml:'yaml' }[e])||'plaintext';
}

async function refreshList(){
  const entries = await window.NuRea.fs.list('.');
  const interesting = entries.filter(e => e.isDir || /\.(py|jl|json|md|txt|toml|rb|yml|yaml)$/i.test(e.name));
  listEl.innerHTML = '';
  for (const e of interesting){
    const li = document.createElement('li');
    li.innerHTML = e.isDir ? `<span class="dir">📁 ${e.path}</span>` : `📄 ${e.path}`;
    li.onclick = async()=>{ if (!e.isDir) openFile(e.path); else listDir(e.path); };
    listEl.appendChild(li);
  }
}

async function listDir(dir){
  const sub = await window.NuRea.fs.list(dir);
  listEl.innerHTML = '';
  for (const s of sub){
    const li = document.createElement('li');
    li.innerHTML = s.isDir ? `<span class="dir">📁 ${s.path}</span>` : `📄 ${s.path}`;
    li.onclick = async()=>{ if (!s.isDir) openFile(s.path); else listDir(s.path); };
    listEl.appendChild(li);
  }
}

async function openFile(rel){
  const txt = await window.NuRea.fs.read(rel);
  let entry = openTabs.get(rel);
  if (!entry){
    const model = monaco.editor.createModel(txt, extToLang(rel));
    entry = { model, dirty:false };
    openTabs.set(rel, entry);
    addTab(rel);
  }else{
    entry.model.setValue(txt);
    entry.dirty = false;
  }
  setActive(rel);
}

function addTab(rel){
  const span = document.createElement('span');
  span.className = 'tab'; span.textContent = rel.split('/').pop();
  span.onclick = () => setActive(rel);
  span.dataset.rel = rel;
  tabbar.appendChild(span);
}

function setActive(rel){
  const entry = openTabs.get(rel); if(!entry) return;
  activeRel = rel;
  editor.setModel(entry.model);
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  const activeTab = Array.from(document.querySelectorAll('.tab')).find(t=>t.dataset.rel===rel);
  if (activeTab) activeTab.classList.add('active');
}

saveBtn.onclick = async()=>{
  if (!activeRel) return;
  const entry = openTabs.get(activeRel);
  await window.NuRea.fs.write(activeRel, entry.model.getValue());
  entry.dirty = false;
  logTerm('Saved '+activeRel);
};

diffBtn.onclick = ()=>{
  const diffEl = document.getElementById('diff');
  const monacoEl = document.getElementById('monaco');
  const show = diffEl.style.display === 'none';
  diffEl.style.display = show ? 'block' : 'none';
  monacoEl.style.display = show ? 'none' : 'block';
};

function initTerminal(){
  term = new Terminal({ convertEol:true, fontSize:12 });
  fitAddon = new FitAddon.FitAddon();
  term.loadAddon(fitAddon);
  term.open(document.getElementById('terminal'));
  fitAddon.fit();
  window.addEventListener('resize', ()=>fitAddon.fit());
  window.NuRea.proc.onStarted(({id, cmd, args})=>logTerm(`[${id}] $ ${cmd} ${args.join(' ')}`));
  window.NuRea.proc.onData(({id, line})=>logTerm(line));
  window.NuRea.proc.onExit(({id, code})=>logTerm(`[${id}] exited ${code}`));
}

backendUpBtn.onclick = async()=>{ logTerm('Starting Julia backend…'); await window.NuRea.compose.up(); setTimeout(checkHealth,1500); };
backendDownBtn.onclick = async()=>{ logTerm('Stopping Julia backend…'); await window.NuRea.compose.down(); setTimeout(checkHealth,500); };
runPlanBtn.onclick = async()=>{ logTerm('Running orchestrator…'); await window.NuRea.orchestrator.run(); };

async function checkHealth(){
  const cfg = await window.NuRea.cfg.get();
  try{ const r = await fetch(cfg.julia.healthUrl); const ok = r.ok; healthEl.textContent = ok?'healthy':'unhealthy'; healthEl.style.color = ok?'#52c41a':'#fa541c'; }
  catch{ healthEl.textContent = 'down'; healthEl.style.color = '#fa541c'; }
}

// --- Simple repo search (text or /regex/)
searchBtn.onclick = async()=>{
  const q = searchInput.value.trim(); if (!q) return;
  const entries = await window.NuRea.fs.list('.');
  const files = entries.filter(e=>!e.isDir && /\.(py|jl|json|md|txt|toml|rb|yml|yaml)$/i.test(e.name));
  resultsEl.innerHTML='';
  const rx = q.startsWith('/') && q.endsWith('/') ? new RegExp(q.slice(1,-1)) : null;
  for (const f of files){
    const txt = await window.NuRea.fs.read(f.path);
    const lines = txt.split(/\r?\n/);
    lines.forEach((line, i)=>{
      const hit = rx ? rx.test(line) : line.includes(q);
      if (hit){
        const li = document.createElement('li');
        li.textContent = `${f.path}:${i+1}: ${line.slice(0,140)}`;
        li.onclick = ()=>{ openFile(f.path).then(()=>{ editor.revealLineInCenter(i+1); editor.setPosition({lineNumber:i+1, column:1}); }); };
        resultsEl.appendChild(li);
      }
    });
  }
};

// --- AI helpers
function buildAiMessages(kind){
  const rel = activeRel; const entry = rel?openTabs.get(rel):null;
  const code = entry? entry.model.getValue(): '';
  const selection = editor.getModel()? editor.getValueInRange(editor.getSelection()) : '';
  const sys = { role:'system', content:'You are an assistant embedded in the NuRea IDE. Be concise. Output patches as unified diffs when changing code.' };
  const base = [sys];
  base.push({ role:'user', content:`Task: ${kind}
File: ${rel||'(none)'}
Selection:
${selection}

Full file content follows. Provide clear explanation, and when editing, return a unified diff (patch) that applies cleanly.

<FILE>
${code}
</FILE>`});
  return base;
}

async function aiChat(messages, {stream=true}={}){
  addChat('user', messages.filter(m=>m.role==='user').map(m=>m.content).join('\n\n'));
  const { id } = await window.NuRea.ai.chat({ messages, stream });
  let buf = '';
  window.NuRea.ai.onData(({id:rid, chunk})=>{
    if (rid!==id) return;
    // Try to parse SSE lines if streaming from OpenAI-compatible servers
    const lines = chunk.split(/\r?\n/).filter(Boolean);
    for (const l of lines){
      if (l.startsWith('data: ')){
        const payload = l.slice(6);
        if (payload === '[DONE]') continue;
        try{
          const obj = JSON.parse(payload);
          const part = obj.choices?.[0]?.delta?.content || obj.choices?.[0]?.message?.content || '';
          buf += part; renderAi(buf);
        }catch{ /* ignore */ }
      } else {
        // Non-SSE body; append raw
        buf += l; renderAi(buf);
      }
    }
  });
  window.NuRea.ai.onClose(({id:rid})=>{ if (rid===id) renderAi(buf,true); });
}

function renderAi(text, final=false){
  const last = chatLog.lastElementChild;
  if (!last || !last.classList.contains('msg') || last.classList.contains('user')){
    const div = document.createElement('div');
    div.className = 'msg ai';
    chatLog.appendChild(div);
  }
  const node = chatLog.lastElementChild;
  node.textContent = text;
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatSend.onclick = async()=>{
  const q = chatInput.value.trim(); if (!q) return;
  chatInput.value='';
  await aiChat([{ role:'user', content:q }]);
};

aiExplainBtn.onclick = async()=> aiChat(buildAiMessages('Explain the code and selection.'));
aiRefactorBtn.onclick = async()=> aiChat(buildAiMessages('Refactor the selection (or file if no selection) for clarity and safety. Return a unified diff.'));
aiTestBtn.onclick = async()=> aiChat(buildAiMessages('Write unit tests for this file. Return tests and a unified diff of new files or additions.'));

// Apply a unified diff returned in the last AI message to the active file
async function applyUnifiedDiff(patch){
  if (!activeRel) return alert('Open a file first.');
  const original = openTabs.get(activeRel).model.getValue();
  // naive patch apply: only supports full-file replacements when diff contains @@ or start-end markers omitted
  if (!/^diff|^@@/m.test(patch)){
    // treat as whole-file replacement
    diffEditor.setModel({
      original: monaco.editor.createModel(original, extToLang(activeRel)),
      modified: monaco.editor.createModel(patch, extToLang(activeRel))
    });
    diffBtn.click();
    return;
  }
  // For real unified diff parsing, integrate a tiny diff lib later.
  alert('Received a unified diff. Visualizing raw is not yet implemented; paste the modified file content instead.');
}

// Allow pasting AI content into diff view quickly
chatLog.addEventListener('dblclick', (e)=>{
  const target = e.target; if (!target.classList.contains('msg')) return;
  const text = target.textContent || '';
  applyUnifiedDiff(text);
});

// boot
boot();
