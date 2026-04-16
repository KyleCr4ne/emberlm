"""Training-monitor HTTP server (port 8001 by default).

Read-only dashboard over `runs/*/loss.jsonl`. Runs in parallel with the
inference server on a separate port.

Usage:
    python scripts/serve_monitor.py
    PORT=9001 python scripts/serve_monitor.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
RUNS_DIR = REPO_ROOT / "runs"

app = FastAPI()


@app.get("/")
@app.get("/train")
def train_page():
    return FileResponse(str(WEB_DIR / "train.html"))


@app.get("/api/runs")
def list_runs():
    if not RUNS_DIR.exists():
        return {"runs": []}
    names = sorted(
        p.name for p in RUNS_DIR.iterdir()
        if p.is_dir() and (p / "loss.jsonl").exists()
    )
    return {"runs": names}


@app.get("/api/runs/{name}/loss")
def run_loss(name: str):
    """Return the run's loss.jsonl parsed, plus config.json if present."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())) or not run_dir.is_dir():
        return {"events": [], "config": None}
    events = []
    log_path = run_dir / "loss.jsonl"
    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    config = None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            config = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            config = None
    return {"events": events, "config": config}


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port, log_level="info")
