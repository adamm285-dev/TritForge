"""Unit test verifying the server's real-time dynaMoE telemetry endpoint."""

import json
from unittest.mock import MagicMock
from tritforge.server import TritForgeAPIHandler


def test_server_dynamoe_telemetry_endpoint():
    handler = TritForgeAPIHandler.__new__(TritForgeAPIHandler)
    handler.path = "/api/telemetry/dynamoe"
    handler.wfile = MagicMock()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.do_GET()

    handler.send_response.assert_called_with(200)
    handler.send_header.assert_any_call("Content-Type", "application/json")

    # Inspect written JSON payload
    written_bytes = b"".join(call.args[0] for call in handler.wfile.write.call_args_list)
    payload = json.loads(written_bytes.decode("utf-8"))

    assert payload["router"] == "dynamoe"
    assert payload["nominal_k"] == 10
    assert payload["current_avg_k"] == 3.8
    assert payload["active_k_range"] == [2, 12]
    assert payload["sparsity_reduction_pct"] == 62.0
