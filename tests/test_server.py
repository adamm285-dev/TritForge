"""
Unit Tests for TritForge OpenAI-Compatible Local Server (tritforge.server)
"""

import json
from pathlib import Path
import socket
import threading
import time
import urllib.request
import pytest

from tritforge.server import ThreadedHTTPServer, TritForgeAPIHandler


def get_free_port() -> int:
    """Finds an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def local_server():
    """Starts a local TritForge server on an ephemeral port for testing."""
    port = get_free_port()
    server = ThreadedHTTPServer(("127.0.0.1", port), TritForgeAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Allow server to initialize

    base_url = f"http://127.0.0.1:{port}"
    yield base_url

    server.shutdown()
    server.server_close()


def test_server_web_ui(local_server):
    """Verify that root GET / serves the interactive Web Chat HTML dashboard."""
    req = urllib.request.Request(f"{local_server}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        html = resp.read().decode("utf-8")
        assert "TritForge 1.58b" in html
        assert "Active Experts" in html
        assert "expert-grid" in html


def test_server_v1_models(local_server):
    """Verify that GET /v1/models returns OpenAI-compliant model listing."""
    req = urllib.request.Request(f"{local_server}/v1/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "qwen3.8-2.4t-tritforge"


def test_server_v1_chat_completions_sync(local_server):
    """Verify that POST /v1/chat/completions handles non-streaming requests."""
    payload = {
        "model": "qwen3.8-2.4t-tritforge",
        "messages": [{"role": "user", "content": "What is TritForge?"}],
        "stream": False,
        "max_tokens": 10,
    }
    req = urllib.request.Request(
        f"{local_server}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "content" in data["choices"][0]["message"]
        assert len(data["active_experts"]) == 10


def test_server_v1_chat_completions_stream(local_server):
    """Verify that POST /v1/chat/completions handles streaming SSE chunks."""
    payload = {
        "model": "qwen3.8-2.4t-tritforge",
        "messages": [{"role": "user", "content": "Stream response please"}],
        "stream": True,
        "max_tokens": 5,
    }
    req = urllib.request.Request(
        f"{local_server}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

        valid_chunks = 0
        has_done = False
        while True:
            line = resp.readline().decode("utf-8")
            if not line:
                break
            line = line.strip()
            if line == "data: [DONE]":
                has_done = True
                break
            if line.startswith("data: "):
                chunk_data = json.loads(line[6:])
                assert "choices" in chunk_data
                assert "delta" in chunk_data["choices"][0]
                valid_chunks += 1

        assert valid_chunks > 0
        assert has_done


def test_server_payload_too_large(local_server):
    """Adversarial test: verify that payloads > 10MB are rejected with HTTP 413."""
    import urllib.error
    # Construct a request declaring Content-Length > 10 MB
    req = urllib.request.Request(
        f"{local_server}/v1/chat/completions",
        data=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": str(15 * 1024 * 1024)},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 413


def test_server_conversion_status(local_server):
    """Verify that GET /api/conversion/status returns valid JSON telemetry."""
    req = urllib.request.Request(f"{local_server}/api/conversion/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "total_layers" in data
        assert data["total_layers"] == 92
        assert "completed_count" in data
        assert "overall_pct" in data
        assert "stats" in data
        assert "eta_hours" in data["stats"]
