"""
dashboard.py — FastAPI autoresearch dashboard on port 8081.

Endpoints
─────────
  GET  /                      → HTML dashboard
  GET  /api/history           → full experiment history JSON
  GET  /api/best              → best experiment so far
  GET  /api/status            → agent loop status (running/idle)
  POST /api/run               → start agent loop (background task)
  POST /api/stop              → stop running agent loop
  GET  /api/schema            → param schema with bounds
  GET  /health                → health check

Run:
  uvicorn ml.autoresearch.dashboard:app --host 0.0.0.0 --port 8081 --reload
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ml.autoresearch.agent import run_agent
from ml.autoresearch.experiment_runner import PARAM_SCHEMA

log = logging.getLogger("autoresearch.dashboard")

HISTORY_PATH = Path(os.environ.get("AUTORESEARCH_HISTORY", "ml/autoresearch/history.json"))
PROGRAM_PATH = Path(os.environ.get("AUTORESEARCH_PROGRAM", "ml/autoresearch/program.md"))
DB_URL       = os.environ.get("DATABASE_URL", "")

app = FastAPI(title="Autoresearch Dashboard", version="1.0.0")

# ── State ──────────────────────────────────────────────────────────────────────

_agent_task: asyncio.Task | None = None
_agent_status = {"running": False, "started_at": None, "experiments_run": 0}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            return []
    return []


def find_best(history: list[dict]) -> dict | None:
    if not history:
        return None
    return max(history, key=lambda e: e.get("metrics", {}).get("kelly_return", -999))


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/history")
async def get_history():
    return JSONResponse(load_history())


@app.get("/api/best")
async def get_best():
    best = find_best(load_history())
    if not best:
        raise HTTPException(status_code=404, detail="No experiments yet")
    return JSONResponse(best)


@app.get("/api/status")
async def get_status():
    return JSONResponse({**_agent_status, "history_count": len(load_history())})


@app.get("/api/schema")
async def get_schema():
    return JSONResponse(PARAM_SCHEMA)


class RunRequest(BaseModel):
    max_experiments: int = 50
    hours: float         = 8.0
    lookback_days: int   = 7
    dry_run: bool        = False


@app.post("/api/run")
async def start_agent(req: RunRequest, background_tasks: BackgroundTasks):
    global _agent_task
    if _agent_status["running"]:
        raise HTTPException(status_code=409, detail="Agent is already running")
    if not DB_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not set")

    _agent_status["running"]          = True
    _agent_status["started_at"]       = datetime.now(timezone.utc).isoformat()
    _agent_status["experiments_run"]  = 0

    async def _run():
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: run_agent(
                    db_url          = DB_URL,
                    history_path    = HISTORY_PATH,
                    program_path    = PROGRAM_PATH,
                    max_experiments = req.max_experiments,
                    hours_limit     = req.hours,
                    lookback_days   = req.lookback_days,
                    dry_run         = req.dry_run,
                ),
            )
        except asyncio.CancelledError:
            log.info("Agent task cancelled")
        finally:
            _agent_status["running"] = False

    _agent_task = asyncio.create_task(_run())
    return {"message": "Agent started", **req.dict()}


@app.post("/api/stop")
async def stop_agent():
    global _agent_task
    if not _agent_status["running"]:
        return {"message": "Agent not running"}
    if _agent_task:
        _agent_task.cancel()
    _agent_status["running"] = False
    return {"message": "Agent stop signal sent"}


# ── HTML Dashboard ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autoresearch Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --yellow: #d29922;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-running  { background: #1a4731; color: var(--green); }
  .badge-idle     { background: #2d2517; color: var(--yellow); }
  main { max-width: 1400px; margin: 24px auto; padding: 0 24px; display: grid; gap: 20px; }
  .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
  .card h2 { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px; }
  .metric { font-size: 28px; font-weight: 700; }
  .metric.green  { color: var(--green); }
  .metric.red    { color: var(--red); }
  .metric.blue   { color: var(--blue); }
  .metric.yellow { color: var(--yellow); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; }
  tr:hover td { background: rgba(255,255,255,.03); }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
  .controls label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .controls input { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 6px; width: 110px; }
  button { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; transition: opacity .15s; }
  button:hover { opacity: .85; }
  .btn-run  { background: var(--green);  color: #000; }
  .btn-stop { background: var(--red);    color: #fff; }
  .btn-ref  { background: var(--border); color: var(--text); }
  pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 14px; font-size: 12px; overflow-x: auto; color: var(--purple); }
  .section-title { font-size: 15px; font-weight: 600; margin-bottom: 12px; }
</style>
</head>
<body>
<header>
  <h1>🔬 Autoresearch Dashboard</h1>
  <span id="statusBadge" class="badge badge-idle">IDLE</span>
</header>
<main>
  <!-- Summary metrics -->
  <div class="row">
    <div class="card"><h2>Total Experiments</h2><div id="expCount" class="metric blue">–</div></div>
    <div class="card"><h2>Best Kelly Return</h2><div id="bestKelly" class="metric green">–</div></div>
    <div class="card"><h2>Best Win Rate</h2><div id="bestWR" class="metric yellow">–</div></div>
    <div class="card"><h2>Best Pass Rate</h2><div id="bestFPR" class="metric blue">–</div></div>
    <div class="card"><h2>Best N Trades</h2><div id="bestN" class="metric">–</div></div>
  </div>

  <!-- Controls -->
  <div class="card">
    <div class="section-title">Launch Agent</div>
    <div class="controls">
      <div><label>Max experiments</label><input id="maxExp" type="number" value="50" min="1" max="200"></div>
      <div><label>Hours limit</label><input id="hours" type="number" value="8" min="1" max="48" step="0.5"></div>
      <div><label>Lookback days</label><input id="lookback" type="number" value="7" min="1" max="30"></div>
      <div style="display:flex;align-items:center;gap:6px;padding-top:18px">
        <input type="checkbox" id="dryRun" style="width:auto"> <label for="dryRun" style="margin:0;color:var(--text)">Dry run</label>
      </div>
      <div style="padding-top:18px;display:flex;gap:8px">
        <button class="btn-run"  onclick="startAgent()">▶ Start</button>
        <button class="btn-stop" onclick="stopAgent()">■ Stop</button>
        <button class="btn-ref"  onclick="refresh()">⟳ Refresh</button>
      </div>
    </div>
  </div>

  <!-- Best params -->
  <div class="card">
    <div class="section-title">Best Parameter Set</div>
    <pre id="bestParams">No experiments yet.</pre>
  </div>

  <!-- History table -->
  <div class="card">
    <div class="section-title">Experiment History</div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>#</th><th>Timestamp</th>
          <th>Kelly ↑</th><th>Win Rate</th><th>Pass Rate</th><th>N Trades</th>
          <th>ML Score</th><th>TP1</th><th>SL</th><th>Time Stop</th>
        </tr></thead>
        <tbody id="historyBody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
async function refresh() {
  const [statusRes, histRes, bestRes] = await Promise.all([
    fetch('/api/status').then(r=>r.json()).catch(()=>{}),
    fetch('/api/history').then(r=>r.json()).catch(()=>[]),
    fetch('/api/best').then(r=>r.ok?r.json():null).catch(()=>null),
  ]);
  if (statusRes) {
    const running = statusRes.running;
    const badge = document.getElementById('statusBadge');
    badge.textContent = running ? 'RUNNING' : 'IDLE';
    badge.className = 'badge ' + (running ? 'badge-running' : 'badge-idle');
  }
  const h = Array.isArray(histRes) ? histRes : [];
  document.getElementById('expCount').textContent = h.length;
  if (bestRes) {
    const m = bestRes.metrics || {};
    document.getElementById('bestKelly').textContent = (m.kelly_return*100).toFixed(2)+'%';
    document.getElementById('bestWR').textContent    = (m.win_rate*100).toFixed(1)+'%';
    document.getElementById('bestFPR').textContent   = (m.filter_pass_rate*100).toFixed(1)+'%';
    document.getElementById('bestN').textContent     = m.n_trades ?? '–';
    document.getElementById('bestParams').textContent = JSON.stringify(bestRes.params, null, 2);
  }
  const tbody = document.getElementById('historyBody');
  tbody.innerHTML = '';
  h.slice().reverse().forEach((e, i) => {
    const m = e.metrics || {};
    const p = e.params || {};
    const kr = m.kelly_return ?? 0;
    const cls = kr >= 0 ? 'positive' : 'negative';
    const ts = e.timestamp ? e.timestamp.slice(0,19).replace('T',' ') : '?';
    const row = `<tr>
      <td>${h.length - i}</td>
      <td>${ts}</td>
      <td class="${cls}">${(kr*100).toFixed(2)}%</td>
      <td>${((m.win_rate??0)*100).toFixed(1)}%</td>
      <td>${((m.filter_pass_rate??0)*100).toFixed(1)}%</td>
      <td>${m.n_trades??'?'}</td>
      <td>${p.min_ensemble_score??'?'}</td>
      <td>${p.tp_1_multiplier??'?'}×</td>
      <td>${((p.hard_stop_loss_pct??0)*100).toFixed(0)}%</td>
      <td>${p.time_stop_minutes??'?'}m</td>
    </tr>`;
    tbody.insertAdjacentHTML('beforeend', row);
  });
}

async function startAgent() {
  const body = {
    max_experiments: parseInt(document.getElementById('maxExp').value),
    hours:           parseFloat(document.getElementById('hours').value),
    lookback_days:   parseInt(document.getElementById('lookback').value),
    dry_run:         document.getElementById('dryRun').checked,
  };
  const res = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  alert(res.ok ? '✅ Agent started' : '❌ ' + (data.detail || 'Error'));
  refresh();
}

async function stopAgent() {
  const res = await fetch('/api/stop', {method:'POST'});
  const data = await res.json();
  alert(data.message);
  refresh();
}

refresh();
setInterval(refresh, 15_000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)
