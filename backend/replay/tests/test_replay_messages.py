import json

import numpy as np
import pytest
from conftest import SPLITS, write_cache
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
        assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
        assert all(f.frame.shape == (3, 32, 32) for f in frames)
        assert all(f.label in range(6) for f in frames)


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
        assert recv_frames(ws, 1)[0].seq == 0  # a new stream starts at seq 0


def test_without_loop_one_recording_is_played_then_ended(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": CLASS_NAMES[2], "loop": False})
        assert ws.receive_json()["type"] == "session"
        frames = recv_frames(ws, 100)
        assert [f.seq for f in frames] == list(range(100))
        assert ws.receive_json() == {"type": "ended"}


def test_gesture_frames_carry_the_true_label(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": CLASS_NAMES[3], "loop": False})
        ws.receive_json()
        labels = [f.label for f in recv_frames(ws, 100)]
        assert set(labels) == {0, 3}
        assert labels.count(3) == 8


def test_background_frames_have_label_zero_and_skip_the_gesture(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": "background", "loop": True})
        ws.receive_json()
        frames = recv_frames(ws, 60)
        assert {f.label for f in frames} == {0}
        numbers = np.rint(codec.decode_frames(np.stack([f.frame for f in frames]))[:, 0, 0, 0] * 99)
        # The zones [s-20, s+L+9] of the four user3 recordings all contain frames 30 to 52 (the zones are 20-57, 25-62, 15-52 and 30-67).
        assert not ((numbers >= 30) & (numbers <= 52)).any()


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


def test_the_first_frame_after_a_start_has_the_clip_start_flag(client):
    with client.websocket_connect("/ws/stream") as ws:
        for _ in range(2):  # a start, then a restart without a stop
            ws.send_json({"type": "start", "user": "user3", "loop": True})
            while True:  # frames of the previous stream can arrive before the new session message
                event = ws.receive()
                if event.get("text") is not None:
                    assert json.loads(event["text"])["type"] == "session"
                    break
            first, second = recv_frames(ws, 2)
            assert first.seq == 0 and first.flags == codec.FLAG_CLIP_START and first.clip_start
            assert second.flags == 0 and not second.clip_start


def test_only_the_first_frame_of_each_clip_is_flagged(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": "any", "loop": True})
        ws.receive_json()
        frames = recv_frames(ws, 250)
        # Every clip of a gesture or user source is a whole recording of 100 frames. The sequence continues across
        # clips, so the flag is the only sign of the jump.
        assert [f.seq for f in frames if f.clip_start] == [0, 100, 200]
        assert all(f.flags in (0, codec.FLAG_CLIP_START) for f in frames)


def test_background_clips_are_flagged_at_every_clip_start(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": "background", "loop": True})
        ws.receive_json()
        frames = recv_frames(ws, 200)
        starts = [f.seq for f in frames if f.clip_start]
        assert starts[0] == 0 and len(starts) >= 3  # the background stretches of user3 are 20 to 50 frames long


def test_a_rejected_start_does_not_stop_a_running_stream(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "loop": True})
        ws.receive_json()
        before = recv_frames(ws, 2)[-1].seq
        ws.send_json({"type": "start", "user": "user99"})
        ws.send_json({"type": "start", "loop": "yes"})
        errors = 0
        seqs = []
        while errors < 2:
            event = ws.receive()
            if event.get("text") is not None:
                errors += '"error"' in event["text"]
            else:
                seqs.append(codec.unpack_frame(event["bytes"]).seq)
        assert seqs == list(range(before + 1, before + 1 + len(seqs)))  # no gap and no restart at seq 0
        assert recv_frames(ws, 1)[0].seq == before + len(seqs) + 1


def test_background_without_a_usable_run_gives_an_error_message(tmp_path):
    cache = write_cache(tmp_path / "cache", {"user3_e1": (3, 1, None, "complete", [(2, 30, 30)])})
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(SPLITS), encoding="utf-8")
    with TestClient(create_app(cache, splits)) as c, c.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "user": "user3", "gesture": "background"})
        reply = ws.receive_json()
        assert reply["type"] == "error" and "background run" in reply["message"]
        ws.send_json({"type": "start", "user": "user3", "gesture": "any", "loop": False})  # the connection survives
        assert ws.receive_json()["type"] == "session"
