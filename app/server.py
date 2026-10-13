"""Vercausa live demo dashboard.

    python -m app.server                  # replay AI4I inside the server
    python -m app.server --source mqtt    # live: readings arrive over MQTT
    python scripts/publish_machine.py     # (second terminal) the machine

Loads the trained model from models/causal_layer.joblib (training it first
if it is missing) and streams every update to the browser. In MQTT mode the
server runs the bundled broker on 127.0.0.1:1883 unless --no-broker is given,
or a broker such as Mosquitto is already running, in which case it uses that.
"""

from __future__ import annotations

import argparse
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
from vercausa.reasoning import HELP, Answer, GraphRAG, GraphRetriever, OllamaLLM
from vercausa.streaming import DEFAULT_HOST, DEFAULT_PORT, Subscriber, broker_running, serve_broker

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


class Question(BaseModel):
    question: str


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


def create_app(engine: ReplayEngine | None = None, source: str = "replay",
               broker: tuple = (DEFAULT_HOST, DEFAULT_PORT), run_broker: bool = True, qos: int = 0,
               llm=None) -> FastAPI:
    state = {"engine": engine, "broker": None, "llm": llm, "llm_ok": False}

    async def llm_ready() -> bool:
        """Re-checks until the local model answers, so starting Ollama after the dashboard works."""
        if state["llm"] is None:
            state["llm"] = OllamaLLM()
        if not state["llm_ok"]:
            state["llm_ok"] = await asyncio.to_thread(state["llm"].available)
        return state["llm_ok"]

    async def player() -> None:
        last_link = None
        while True:
            eng = state["engine"]
            if eng.mode == "mqtt":
                link = eng.link_state()
                if link != last_link:
                    last_link = link
                    eng.version += 1
                await asyncio.sleep(0.25)
            elif eng.playing and not eng.finished:
                new = eng.step()
                if new and eng.pause_on_failure:
                    eng.playing = False
                await asyncio.sleep(1 / eng.speed)
            else:
                await asyncio.sleep(0.05)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if state["engine"] is None:
            state["engine"] = ReplayEngine(open_store, load_model(), mode=source)
        eng = state["engine"]
        await llm_ready()
        print(f"Reasoning: {'local LLM ' + state['llm'].model if state['llm_ok'] else 'no LLM found -- answers built from graph facts only'}")
        tasks = [asyncio.create_task(player())]
        subscriber = None
        if eng.mode == "mqtt":
            if run_broker and not broker_running(*broker):
                tasks.append(asyncio.create_task(serve_broker(*broker)))
                state["broker"] = "bundled"
                await asyncio.sleep(0.3)
            else:
                state["broker"] = "external"
            print(f"MQTT: using the {state['broker']} broker at {broker[0]}:{broker[1]} (QoS {qos})")
            loop = asyncio.get_running_loop()
            subscriber = Subscriber(
                "vercausa-server",
                on_reading=lambda r: loop.call_soon_threadsafe(eng.ingest_external, r),
                on_reject=lambda reason, topic: loop.call_soon_threadsafe(eng.reject, reason, topic),
                on_link=lambda ok: loop.call_soon_threadsafe(eng.set_broker_connected, ok),
                host=broker[0], port=broker[1], qos=qos,
            )
        yield
        if subscriber:
            subscriber.close()
        for t in tasks:
            t.cancel()

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
        ok = await llm_ready()
        return dict(state["engine"].meta(), llm={"available": ok, "model": getattr(state["llm"], "model", None)})

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
        if eng.mode == "mqtt" and c.action != "reset":
            raise HTTPException(409, "In MQTT mode the machine publisher controls the pace; only Reset is available.")
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

    @app.post("/api/ask")
    async def ask(q: Question):
        question = q.question.strip()
        if not question:
            raise HTTPException(400, "Type a question first.")
        eng = state["engine"]
        await llm_ready()
        rag = GraphRAG(GraphRetriever(eng.store, eng.causal, eng.sim, eng.typical, eng.machine_id),
                       state["llm"] if state["llm_ok"] else None)
        intent = await asyncio.to_thread(rag.understand, question)
        if intent["intent"] == "unsupported":
            return Answer(question, intent, HELP, [], "help").to_dict()
        facts = rag.retrieve(intent)
        answer = await asyncio.to_thread(rag.generate, question, intent, facts)
        return json_safe(answer.to_dict())

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
    parser = argparse.ArgumentParser(description="Vercausa live monitor")
    parser.add_argument("--source", choices=("replay", "mqtt"), default="replay",
                        help="replay the dataset in-process, or receive readings over MQTT")
    parser.add_argument("--broker", default=f"{DEFAULT_HOST}:{DEFAULT_PORT}", help="MQTT broker host:port")
    parser.add_argument("--no-broker", action="store_true",
                        help="never start the bundled broker (a running Mosquitto is used automatically anyway)")
    parser.add_argument("--qos", type=int, choices=(0, 1), default=0, help="1 needs Mosquitto or another full broker")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port")
    args = parser.parse_args()
    host, _, port = args.broker.rpartition(":")

    url = f"http://127.0.0.1:{args.port}"
    print(f"Vercausa Live Monitor starting at {url}  (Ctrl+C to stop)")
    if args.source == "mqtt":
        print(f"Listening for machine readings on mqtt://{host}:{port}, topic vercausa/+/readings")
        print("Start the machine in another terminal:  python scripts/publish_machine.py")
    webbrowser.open(url)
    uvicorn.run(create_app(source=args.source, broker=(host, int(port)), run_broker=not args.no_broker, qos=args.qos),
                host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
