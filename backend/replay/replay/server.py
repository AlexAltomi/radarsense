"""The replay server: one WebSocket endpoint, /ws/stream, with one asyncio task per client.

App -> server (JSON text):
  {"type": "start", "user": "user3" | "any", "gesture": "any" | "background" | <class name>, "loop": true}
  {"type": "stop"}
Server -> app:
  {"type": "session", "fps": 33, "shape": [3, 32, 32], "encoding": "uint8", "classes": [...]}  before the first frame
  one binary frame (see replay.codec) every 1/33 s. Bit 0 of the header flags marks the first frame of every clip, which
  includes the first frame after a start. It is a discontinuity, so the app clears its frame buffer.
  {"type": "stopped"} after "stop", {"type": "ended"} when a stream with loop=false has played one recording,
  {"type": "error", "message": ...} for a bad request. The connection stays open after an error.

Pacing: frame n is due at start + n / 33 seconds on time.monotonic(). The loop sleeps until that time and never for a
fixed interval, so a late frame does not delay the following ones.
"""
import asyncio
import contextlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from radarsense_dsp.windows import CLASS_NAMES, load_splits
from replay import codec
from replay.index import ANY, ReplayError, ReplayIndex
from replay.sources import next_clip

FPS = 33
INTERVAL_S = 1.0 / FPS  # 30.303 ms
MAX_LAG_S = 0.25  # A client that falls this far behind gets a fresh schedule instead of a burst of frames.

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_SPLITS_PATH = REPO_ROOT / "training" / "splits.json"


def session_message() -> dict:
    return {"type": "session", "fps": FPS, "shape": list(codec.FRAME_SHAPE), "encoding": "uint8", "classes": list(CLASS_NAMES)}


class Client:
    """The state of one connected app."""

    def __init__(self, ws: WebSocket, index: ReplayIndex):
        self.ws = ws
        self.index = index
        self.rng = np.random.default_rng()
        self.task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def send_json(self, message: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(message)

    async def error(self, message: str) -> None:
        await self.send_json({"type": "error", "message": message})

    async def stop_stream(self) -> None:
        task, self.task = self.task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def start(self, user: str, gesture: str, loop: bool) -> None:
        try:  # A rejected start leaves a running stream alone.
            first = await asyncio.to_thread(next_clip, self.index, self.rng, user, gesture)
        except ReplayError as e:
            await self.error(str(e))
            return
        await self.stop_stream()
        await self.send_json(session_message())
        self.task = asyncio.create_task(self._stream(first, user, gesture, loop))

    async def _stream(self, clip, user: str, gesture: str, loop: bool) -> None:
        loop_ = asyncio.get_running_loop()
        try:
            seq, start = 0, None
            while True:
                prefetch = loop_.run_in_executor(None, next_clip, self.index, self.rng, user, gesture) if loop else None
                for i in range(len(clip.frames)):
                    now = time.monotonic()
                    if start is None:
                        start = now
                    delay = start + seq * INTERVAL_S - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    elif delay < -MAX_LAG_S:
                        start = now - seq * INTERVAL_S
                    flags = codec.FLAG_CLIP_START if i == 0 else 0  # The first frame of every clip is a discontinuity.
                    message = codec.pack_frame(
                        seq, time.time_ns() // 1_000_000, int(clip.labels[i]), clip.frames[i], flags
                    )
                    async with self._send_lock:
                        await self.ws.send_bytes(message)
                    seq += 1
                if prefetch is None:
                    await self.send_json({"type": "ended"})
                    return
                clip = await prefetch
        except (WebSocketDisconnect, RuntimeError):
            return  # The client went away.


def _parse_start(msg: dict) -> tuple[str, str, bool]:
    user, gesture, loop = msg.get("user", ANY), msg.get("gesture", ANY), msg.get("loop", True)
    if not isinstance(user, str) or not isinstance(gesture, str) or not isinstance(loop, bool):
        raise ReplayError("'user' and 'gesture' must be strings and 'loop' must be true or false.")
    return user, gesture, loop


def create_app(cache_dir: str | Path | None = None, splits_path: str | Path | None = None) -> FastAPI:
    """Create the app. The index is built at startup (in the lifespan), not at import time."""
    cache = Path(cache_dir or os.environ.get("RADARSENSE_CACHE_DIR") or DEFAULT_CACHE_DIR)
    splits_file = Path(splits_path or os.environ.get("RADARSENSE_SPLITS") or DEFAULT_SPLITS_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.index = await asyncio.to_thread(ReplayIndex, cache, load_splits(splits_file))
        yield

    app = FastAPI(title="RadarSense replay server", lifespan=lifespan)

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        client = Client(ws, ws.app.state.index)
        try:
            while True:
                event = await ws.receive()
                if event["type"] == "websocket.disconnect":
                    return
                text = event.get("text")
                if text is None:
                    await client.error("Send JSON text messages, not binary data.")
                    continue
                try:
                    msg = json.loads(text)
                except ValueError:
                    msg = None
                if not isinstance(msg, dict):
                    await client.error("A message must be a JSON object with a 'type'.")
                elif msg.get("type") == "start":
                    try:
                        await client.start(*_parse_start(msg))
                    except ReplayError as e:
                        await client.error(str(e))
                elif msg.get("type") == "stop":
                    await client.stop_stream()
                    await client.send_json({"type": "stopped"})
                else:
                    await client.error(f"Unknown message type {msg.get('type')!r}. Use 'start' or 'stop'.")
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            await client.stop_stream()

    return app
