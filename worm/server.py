"""Live viewer: runs the simulation in a thread and streams frames to a browser.

Stdlib only -- ThreadingHTTPServer plus server-sent events. No web framework,
no build step, nothing to install.

  GET  /         the viewer page
  GET  /stream   server-sent event stream of frames
  GET  /genes    catalogue of genes with a modelled phenotype
  POST /control  {"cmd": ..., ...} to poke, place food, knock out genes, reset
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .environment import Environment
from .genome import GENE_EFFECTS
from .simulation import SimConfig, WormSimulation

from .paths import viewer_html


def _viewer_ok() -> bool:
    try:
        viewer_html()
        return True
    except FileNotFoundError:
        return False


class SimRunner:
    """Owns the simulation and steps it on a clock in a background thread."""

    def __init__(self, sim: WormSimulation, fps: int = 30) -> None:
        self.sim = sim
        self.fps = fps
        self.speed = 1
        self.paused = False
        self.lock = threading.Lock()
        self.subscribers: list[queue.Queue] = []
        self._stop = threading.Event()
        self.last_frame: dict = {}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, frame: dict) -> None:
        self.last_frame = frame
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass  # slow client: drop the frame rather than stall the sim

    def run(self) -> None:
        interval = 1.0 / self.fps
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if not self.paused:
                with self.lock:
                    for _ in range(max(1, self.speed)):
                        tel = self.sim.step()
                    frame = self.sim.snapshot()
                frame["telemetry"] = tel
                frame["env"] = self.env_state()
                self.publish(frame)
            next_t += interval
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    def env_state(self) -> dict:
        e = self.sim.env
        return {
            "width": e.width, "height": e.height,
            "sources": [{"x": s.x, "y": s.y, "kind": s.kind,
                         "strength": s.strength, "sigma": s.sigma}
                        for s in e.sources],
            "oxygen": e.oxygen,
            "drag_ratio": e.drag_ratio,
            "temp_low": e.temp_low, "temp_high": e.temp_high,
            "cultivation_temp": e.cultivation_temp,
            "background_pheromone": e.background_pheromone,
            "life_speedup": self.sim.cfg.life_speedup,
        }

    def stop(self) -> None:
        self._stop.set()

    # -- commands -------------------------------------------------------
    def control(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        with self.lock:
            sim, env = self.sim, self.sim.env
            try:
                if cmd == "poke":
                    where = msg.get("region", msg.get("u", "anterior"))
                    if not isinstance(where, str):
                        where = float(where)
                    env.poke(where, float(msg.get("strength", 1.0)),
                             float(msg.get("duration", 0.3)),
                             harsh=msg.get("harsh"))
                    return {"ok": True, "msg": f"poked {where}"}
                if cmd == "poke_at":
                    hit = sim.poke_at(float(msg["x"]), float(msg["y"]),
                                      strength=float(msg.get("strength", 1.0)),
                                      harsh=msg.get("harsh"))
                    if hit is None:
                        return {"ok": False, "msg": "missed the worm"}
                    kind = "harsh prod" if hit["harsh"] else "gentle touch"
                    pct = round(hit["u"] * 100)
                    return {"ok": True, "hit": hit,
                            "msg": f"{kind} at {pct}% along the body"}
                if cmd == "add_source":
                    env.add_source(float(msg["x"]), float(msg["y"]),
                                   kind=msg.get("kind", "salt"),
                                   strength=float(msg.get("strength", 1.0)),
                                   sigma=float(msg.get("sigma", 8.0)))
                    return {"ok": True, "msg": f"added {msg.get('kind')} source"}
                if cmd == "clear_sources":
                    env.clear_sources(msg.get("kind"))
                    return {"ok": True, "msg": "cleared sources"}
                if cmd == "ablate":
                    rec = sim.ablate(msg["cell"])
                    return {"ok": True, "record": rec,
                            "msg": f"ablated {rec['cell']}"}
                if cmd == "restore_cell":
                    sim.restore_cell(msg["cell"])
                    return {"ok": True, "msg": f"{msg['cell']} restored"}
                if cmd == "clear_ablations":
                    sim.clear_ablations()
                    return {"ok": True, "msg": "all cells restored"}
                if cmd == "knockout":
                    rec = sim.knock_out(msg["gene"])
                    return {"ok": True, "msg": f"{rec['gene']} knocked out",
                            "record": rec}
                if cmd == "restore":
                    sim.restore(msg["gene"])
                    return {"ok": True, "msg": f"{msg['gene']} restored"}
                if cmd == "reset_genome":
                    sim.reset_genome()
                    return {"ok": True, "msg": "genome restored to wild type"}
                if cmd == "reset":
                    sim.reset()
                    return {"ok": True, "msg": "simulation reset"}
                if cmd == "set":
                    for k in ("oxygen", "drag_ratio", "cultivation_temp",
                              "temp_low", "temp_high", "background_pheromone",
                              "depletion_rate"):
                        if k in msg:
                            setattr(env, k, float(msg[k]))
                    if "life_speedup" in msg:
                        sim.cfg.life_speedup = max(0.0, float(msg["life_speedup"]))
                    if "restart_as_egg" in msg:
                        from .lifecycle import Lifecycle
                        sim.life = Lifecycle()          # a fertilised egg
                        sim.body.set_length(sim.life.body_length_mm)
                        sim.events.clear()
                        return {"ok": True,
                                "msg": "restarted as a fertilised egg"}
                    if "restart_as_L1" in msg:
                        from .lifecycle import Lifecycle
                        sim.life = Lifecycle()
                        sim.life.stage = "L1"
                        sim.body.set_length(sim.life.body_length_mm)
                        sim.events.clear()
                        return {"ok": True, "msg": "restarted as a freshly hatched L1"}
                    if "speed" in msg:
                        self.speed = max(1, min(20, int(msg["speed"])))
                    if "paused" in msg:
                        self.paused = bool(msg["paused"])
                    return {"ok": True, "msg": "updated"}
            except KeyError as exc:
                return {"ok": False, "msg": f"missing field {exc}"}
            except Exception as exc:  # surface the reason to the UI
                return {"ok": False, "msg": str(exc)}
        return {"ok": False, "msg": f"unknown command {cmd!r}"}


def make_handler(runner: SimRunner):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console clean
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                vp = viewer_html() if _viewer_ok() else None
                if vp is None:
                    self._send(500, b"viewer/index.html missing", "text/plain")
                    return
                self._send(200, vp.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/network":
                # Static graph, fetched once. Only the activity vector is
                # streamed per frame.
                conn = runner.sim.conn
                payload = dict(conn.layout())
                payload["edges"] = conn.edge_list(min_weight=3.0)
                payload["index"] = conn.names
                self._send(200, json.dumps(payload, separators=(",", ":")).encode(),
                           "application/json")
            elif self.path == "/genes":
                cat = [{"gene": g, "product": e.product, "phenotype": e.phenotype}
                       for g, e in sorted(GENE_EFFECTS.items())]
                self._send(200, json.dumps(cat).encode(), "application/json")
            elif self.path == "/stream":
                self.stream()
            else:
                self._send(404, b"not found", "text/plain")

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = runner.subscribe()
            try:
                while True:
                    try:
                        frame = q.get(timeout=5.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(frame, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                runner.unsubscribe(q)

        def do_POST(self):
            if self.path != "/control":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                msg = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b'{"ok":false,"msg":"bad json"}', "application/json")
                return
            result = runner.control(msg)
            self._send(200, json.dumps(result).encode(), "application/json")

    return Handler


def default_environment() -> Environment:
    env = Environment(width=44.0, height=32.0)
    env.add_source(13.0, 7.0, kind="salt", strength=1.0, sigma=7.0)
    env.add_source(13.0, 7.0, kind="food", strength=1.0, sigma=4.0)
    return env


def serve(host: str = "127.0.0.1", port: int = 8080, fps: int = 30,
          seed: int = 0) -> None:
    sim = WormSimulation(env=default_environment(), config=SimConfig(seed=seed))
    runner = SimRunner(sim, fps=fps)
    threading.Thread(target=runner.run, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(runner))
    print(f"  worm viewer -> http://{host}:{port}")
    print("  ctrl-c to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        runner.stop()
        httpd.server_close()
