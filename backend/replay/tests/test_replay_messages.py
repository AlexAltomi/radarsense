import numpy as np
import pytest
from fastapi.testclient import TestClient

from radarsense_dsp.windows import CLASS_NAMES
from replay import codec
from replay.server import create_app


@pytest.fixture(scope="module")
def client(cache_dir, splits_path):
    with TestClient(create_app(cache_dir, splits_path)) as c:
        yield c


def recv_frames(ws, n):
    out = []
    for _ in range(n):
        event = ws.receive()
        assert event.get("bytes") is not None, event
        out.append(codec.unpack_frame(event["bytes"]))
    return out


def test_start_sends_session_then_binary_frames(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "any", "gesture": "any", "loop": True})
        assert ws.receive_json() == {
            "type": "session",
            "fps": 33,
            "shape": [3, 32, 32],
            "encoding": "uint8",
            "classes": list(CLASS_NAMES),
        }
        frames = recv_frames(ws, 5)
        assert [f[0] for f in frames] == [0, 1, 2, 3, 4]
        assert all(f[3].shape == (3, 32, 32) for f in frames)
        assert all(f[2] in range(6) for f in frames)


def test_each_frame_message_is_3088_bytes(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        assert len(ws.receive_bytes()) == 3088


def test_stop_ends_the_stream_and_the_connection_stays_usable(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": "any", "loop": True})
        ws.receive_json()
        recv_frames(ws, 2)
        ws.send_json({"type": "stop"})
        while True:  # frames that were already sent may arrive before the confirmation
            event = ws.receive()
            if event.get("text") is not None:
                break
        assert event["text"] == '{"type":"stopped"}'
        ws.send_json({"type": "start", "user": "user5"})
        assert ws.receive_json()["type"] == "session"
        assert recv_frames(ws, 1)[0][0] == 0  # a new stream starts at seq 0


def test_without_loop_one_recording_is_played_then_ended(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": CLASS_NAMES[2], "loop": False})
        assert ws.receive_json()["type"] == "session"
        frames = recv_frames(ws, 100)
        assert [f[0] for f in frames] == list(range(100))
        assert ws.receive_json() == {"type": "ended"}


def test_gesture_frames_carry_the_true_label(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": CLASS_NAMES[3], "loop": False})
        ws.receive_json()
        labels = [f[2] for f in recv_frames(ws, 100)]
        assert set(labels) == {0, 3}
        assert labels.count(3) == 8


def test_background_frames_have_label_zero_and_skip_the_gesture(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": "background", "loop": True})
        ws.receive_json()
        frames = recv_frames(ws, 60)
        assert {f[2] for f in frames} == {0}
        numbers = np.rint(codec.decode_frames(np.stack([f[3] for f in frames]))[:, 0, 0, 0] * 99)
        # The zones [s-20, s+L+9] of the four user3 recordings all contain frames 30 to 57.
        assert not ((numbers >= 30) & (numbers <= 57)).any()


@pytest.mark.parametrize(
    "start",
    [
        {"type": "start", "user": "user99"},
        {"type": "start", "user": "user10"},  # a test user: the cache has none, so it is unknown
        {"type": "start", "user": "user12", "gesture": "any"},
        {"type": "start", "gesture": "wave"},
        {"type": "start", "user": "user5", "gesture": CLASS_NAMES[2]},  # user5 has no class-2 recording
        {"type": "start", "user": 3},
        {"type": "start", "loop": "yes"},
    ],
)
def test_bad_start_gives_an_error_message_and_the_connection_survives(client, start):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(start)
        reply = ws.receive_json()
        assert reply["type"] == "error" and isinstance(reply["message"], str)
        ws.send_json({"type": "start", "user": "user3", "loop": False})
        assert ws.receive_json()["type"] == "session"


def test_a_test_user_is_never_served(client):
    with client.websocket_connect("/ws/stream") as ws:
        for user in ("user10", "user12"):
            ws.send_json({"type": "start", "user": user})
            assert ws.receive_json()["type"] == "error"


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"no": "type"}', '{"type": "dance"}', "null"])
def test_garbage_gives_an_error_message(client, raw):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(raw)
        assert ws.receive_json()["type"] == "error"


def test_binary_input_gives_an_error_message(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(b"\x00\x01")
        assert ws.receive_json()["type"] == "error"


def test_stop_without_a_stream_is_fine(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "stopped"}


def test_disconnect_while_streaming_is_clean(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        recv_frames(ws, 3)
    # The server is still alive for the next client.
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "stopped"}
