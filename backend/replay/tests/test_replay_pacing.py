"""Pacing is measured against a real uvicorn server in a separate thread and the websockets client. Starlette's
TestClient adds its own threading overhead, which would make these timings meaningless."""
import json
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from websockets.sync.client import connect

from replay import codec
from replay.server import create_app

DURATION_S = 10.0


@pytest.fixture(scope="module")
def server_url(cache_dir, splits_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))  # an ephemeral port
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(cache_dir, splits_path), log_level="warning", ws="websockets-sansio"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "The uvicorn server did not start."
        time.sleep(0.01)
    yield f"ws://127.0.0.1:{port}/ws/stream"
    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


def test_stream_runs_at_33_fps_with_a_30_3_ms_mean_interval(server_url):
    arrivals, seqs = [], []
    with connect(server_url, max_size=None) as ws:
        ws.send(json.dumps({"type": "start", "user": "any", "gesture": "any", "loop": True}))
        assert json.loads(ws.recv())["fps"] == 33
        t0 = None
        while True:
            message = ws.recv()
            now = time.perf_counter()
            assert isinstance(message, bytes) and len(message) == codec.FRAME_SIZE
            if t0 is None:
                t0 = now
            if now - t0 >= DURATION_S:
                break
            arrivals.append(now)
            seqs.append(codec.unpack_frame(message).seq)
        ws.send(json.dumps({"type": "stop"}))
    assert seqs == list(range(len(seqs)))  # no frame is dropped or reordered, also across clip boundaries
    fps = len(arrivals) / DURATION_S
    mean_interval_ms = np.diff(arrivals).mean() * 1000
    print(f"fps={fps:.2f} mean={mean_interval_ms:.2f}ms max={np.diff(arrivals).max() * 1000:.1f}ms")
    assert 32 <= fps <= 34, f"{fps:.2f} fps"
    assert 28.3 <= mean_interval_ms <= 32.3, f"mean interval {mean_interval_ms:.2f} ms"
