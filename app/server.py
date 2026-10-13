"""Vercausa live demo dashboard.

    python -m app.server            # then open http://127.0.0.1:8000

Loads the trained model from models/causal_layer.joblib (training it first
if it is missing), replays AI4I 2020 as a live machine feed, and streams
every update to the browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import warnings
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from app.engine import ReplayEngine, json_safe
from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC = pathlib.Path(__file__).resolve().parent / "static"
MODEL = ROOT / "models" / "causal_layer.joblib"
SPEEDS = (1, 2, 5, 10, 25)


class Control(BaseModel):
    action: str
    value: Optional[float] = None


class WhatIf(BaseModel):
    event_id: str
    interventions: dict


def load_model() -> CausalLayer:
    if MODEL.exists():
        return CausalLayer.load(MODEL)
    print("No trained model found -- training one now (about 6 seconds)...")
    store = open_store()
    store.apply_schema()
    layer = CausalLayer.fit_from_store(store, training_frame(DEFAULT_CSV))
    MODEL.parent.mkdir(exist_ok=True)
    layer.save(MODEL)
    return layer


def create_app(engine: ReplayEngine | None = None) -> FastAPI:
    state = {"engine": engine}

    async def player() -> None:
        while True:
            eng = state["engine"]
            if eng.playing and not eng.finished:
                new = eng.step()
                if new and eng.pause_on_failure:
                    eng.playing = False
                await asyncio.sleep(1 / eng.speed)
            else:
                await asyncio.sleep(0.05)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if state["engine"] is None:
            state["engine"] = ReplayEngine(open_store, load_model())
        task = asyncio.create_task(player())
        yield
        task.cancel()

    app = FastAPI(title="Vercausa Live Monitor", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def always_revalidate(request: Request, call_next):
        # Browsers otherwise keep serving a stale copy of the page after it is
        # updated; "no-cache" makes them check for a newer version each load.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    async def meta():
        return state["engine"].meta()

    @app.get("/api/state")
    async def get_state():
        return json_safe(state["engine"].state())

    @app.get("/api/stream")
    async def stream(request: Request):
        async def events():
            last = -1
            while not await request.is_disconnected():
                eng = state["engine"]
                if eng.version != last:
                    last = eng.version
                    yield f"data: {json.dumps(json_safe(eng.state()))}\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/api/control")
    async def control(c: Control):
        eng = state["engine"]
        if c.action == "play":
            eng.playing = not eng.finished
        elif c.action == "pause":
            eng.playing = False
        elif c.action == "step":
            eng.playing = False
            eng.step()
        elif c.action == "next_failure":
            eng.playing = False
            eng.next_failure()
        elif c.action == "reset":
            eng.reset()
        elif c.action == "speed" and c.value in SPEEDS:
            eng.speed = float(c.value)
        elif c.action == "pause_on_failure":
            eng.pause_on_failure = bool(c.value)
        else:
            raise HTTPException(400, f"Unknown control '{c.action}' or value {c.value}")
        eng.version += 1
        return {"ok": True}

    @app.post("/api/simulate")
    async def simulate(w: WhatIf):
        eng = state["engine"]
        if not any(e["id"] == w.event_id for e in eng.events):
            raise HTTPException(404, "That failure event is not in the current replay. Press Reset?")
        try:
            return json_safe(eng.simulate(w.event_id, w.interventions))
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


app = create_app()


def main() -> None:
    url = "http://127.0.0.1:8000"
    print(f"Vercausa Live Monitor starting at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
