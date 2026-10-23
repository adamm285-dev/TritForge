"""
Unit Tests for TritForge OpenAI-Compatible Local Server (tritforge.server)
"""

import json
from pathlib import Path
import socket
import threading
import time
import urllib.error
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
        assert "Inference Studio (Not Connected)" in html
        assert "✓ SHA-256 Block Match" not in html
        # D4: Assert HTML contains no "Zero-Multiply" and no "92.8"
        assert "Zero-Multiply" not in html, "HTML dashboard contains leftover 'Zero-Multiply' claim"
        assert "92.8" not in html, "HTML dashboard contains leftover '92.8' claim"



def test_server_v1_models(local_server):
    """Verify that GET /v1/models returns OpenAI-compliant model listing."""
    req = urllib.request.Request(f"{local_server}/v1/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "qwen3.8-2.4t-tritforge"


def test_server_status(local_server):
    """Verify that GET /api/status reports model_loaded: false (I-1)."""
    req = urllib.request.Request(f"{local_server}/api/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "idle"
        assert data["model_loaded"] is False


def test_server_v1_chat_completions_returns_501(local_server):
    """I-1: Verify that POST /v1/chat/completions returns HTTP 501 Not Implemented."""
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
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 501
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "Inference not implemented for qwen3_5_moe" in body["error"]


def test_server_security_invalid_host(local_server):
    """S-1: Verify that requests with non-localhost/127.0.0.1 Host headers return 403."""
    req = urllib.request.Request(
        f"{local_server}/api/status",
        headers={"Host": "evil.attacker.com"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 403


def test_server_security_schema_validation(local_server):
    """S-3: Verify that invalid request bodies or types return HTTP 400."""
    # 1. Invalid JSON body
    req = urllib.request.Request(
        f"{local_server}/v1/chat/completions",
        data=b"not a json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400

    # 2. messages is not a list
    req = urllib.request.Request(
        f"{local_server}/v1/chat/completions",
        data=json.dumps({"messages": "hello"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400

    # 3. max_tokens is not an int
    req = urllib.request.Request(
        f"{local_server}/v1/chat/completions",
        data=json.dumps({"messages": [], "max_tokens": "infinite"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_server_payload_too_large(local_server):
    """Adversarial test: verify that payloads > 10MB are rejected with HTTP 413."""
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
