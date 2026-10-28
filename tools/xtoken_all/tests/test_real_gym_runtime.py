"""Exercise the test gateway with actual HTTP sockets, without Modal/network jobs."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from decimal import Decimal
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest


def load_runner(monkeypatch):
    class Image:
        @classmethod
        def from_registry(cls, *args, **kwargs):
            return cls()

        def entrypoint(self, value):
            return self

    class App:
        def __init__(self, *args):
            pass

        def function(self, **kwargs):
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    fake = SimpleNamespace(is_local=lambda: False, Image=Image, App=App,
                           Volume=SimpleNamespace(from_name=lambda *args, **kwargs: None))
    path = Path(__file__).resolve().parents[1] / "modal_gym_pipeline.py"
    spec = importlib.util.spec_from_file_location("isolated_real_gym_runner", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "modal", fake)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway(monkeypatch):
    observed = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.reply()

        def reply(self):
            count = int(self.headers["Content-Length"]) if "Content-Length" in self.headers else 0
            observed.append((self.path, self.headers["Authorization"], self.rfile.read(count)))
            data = json.dumps({"actual_upstream_response": self.path}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    runner = load_runner(monkeypatch)
    server = runner.metadata_gateway("test-ephemeral-key", ("127.0.0.1", 0),
                                     "http://127.0.0.1:" + str(upstream.server_port))
    try:
        yield "http://127.0.0.1:" + str(server.server_port), observed
    finally:
        server.shutdown()
        server.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_super_metadata_forwarded_without_client_auth(gateway):
    url, observed = gateway
    with urllib.request.urlopen(url + "/v1/models") as response:
        assert json.load(response) == {"actual_upstream_response": "/v1/models"}
    assert observed == [("/v1/models", "Bearer test-ephemeral-key", b"")]


def test_scoring_requires_key_and_preserves_exact_payload(gateway):
    url, observed = gateway
    payload = b'{"prompt":[3,4],"prompt_logprobs":20}'
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data=payload))
    assert error.value.code == 401
    assert observed == []
    request = urllib.request.Request(url + "/v1/completions", data=payload,
                                     headers={"Authorization": "Bearer test-ephemeral-key"})
    with urllib.request.urlopen(request) as response:
        assert json.load(response) == {"actual_upstream_response": "/v1/completions"}
    assert observed == [("/v1/completions", "Bearer test-ephemeral-key", payload)]


def test_unrelated_routes_are_not_exposed(gateway):
    url, observed = gateway
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(url + "/metrics")
    assert error.value.code == 404
    assert observed == []


def test_service_preflight_rejects_old_uvicorn_signature():
    path = Path(__file__).resolve().parents[1] / "gym_service_probe.py"
    spec = importlib.util.spec_from_file_location("isolated_real_gym_service_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def old_run(app, host=None, port=None):
        pass

    def super_compatible_run(app=None, timeout_worker_healthcheck=None):
        pass

    with pytest.raises(TypeError, match="timeout_worker_healthcheck"):
        module.validate_server_runner(old_run)
    module.validate_server_runner(super_compatible_run)


def test_budget_retains_prior_usage_without_double_counting_paid_calls(monkeypatch):
    module = load_runner(monkeypatch)
    records = [{"upper_bound_usd": "0.35"}, {"upper_bound_usd": "2.4"}]
    assert module.cumulative_upper("5.0", "3.3", records) == Decimal("6.05")
    assert module.cumulative_upper("7.0", "3.3", records) == Decimal("7.0")


def test_cpu_settlement_uses_completed_receipt_and_preserves_gpu_reservation(monkeypatch, tmp_path):
    module = load_runner(monkeypatch)
    ledger = tmp_path / "ledger.json"
    rows = [{"id": "101", "phase": "imports", "arm": "same", "upper_bound_usd": "0.35"},
            {"id": "102", "phase": "gpu", "arm": "same", "upper_bound_usd": "8.8"},
            {"id": "103", "phase": "stage", "arm": "same", "upper_bound_usd": "0.35"}]
    ledger.write_text(json.dumps(rows))
    (tmp_path / "profile-same-imports-101.json").write_text(json.dumps({
        "profile": "profile", "arm": "same", "exit_code": 0, "elapsed_seconds": 100}))
    module.settle_cpu_history(ledger, tmp_path, "profile")
    result = json.loads(ledger.read_text())
    assert result[0]["upper_bound_usd"] == "0.06"
    assert result[0]["initial_upper_bound_usd"] == "0.35"
    assert len(result[0]["settlement_receipt_sha256"]) == 64
    assert result[1:] == rows[1:]
    module.settle_cpu_history(ledger, tmp_path, "profile")
    assert json.loads(ledger.read_text()) == result


@pytest.mark.parametrize("elapsed", [-1, "NaN", "Infinity"])
def test_cpu_settlement_rejects_invalid_elapsed(monkeypatch, elapsed):
    module = load_runner(monkeypatch)
    with pytest.raises(ValueError):
        module.cpu_elapsed_bound(elapsed)


def test_shared_role_bounds_include_only_the_reserved_role(monkeypatch):
    module = load_runner(monkeypatch)
    assert module.gpu_role_bound('H100', 2, 1050, 0, 212, '0.1') == Decimal('2.2')
    assert module.gpu_role_bound('H100', 2, 1200, 0, 212, '0.1') == Decimal('2.4')
    assert module.gpu_role_bound('H100', 2, 0, 850, 212, '0.1') == Decimal('3.0')
    with pytest.raises(ValueError):
        module.gpu_role_bound('H100', 2, -1, 850, 212, '0.1')


def test_cpu_settlement_never_releases_shared_gpu_role_reservations(monkeypatch, tmp_path):
    module = load_runner(monkeypatch)
    ledger = tmp_path / 'ledger.json'
    rows = [{'id': '1', 'phase': 'teacher', 'arm': 'same', 'upper_bound_usd': '2.2'},
            {'id': '2', 'phase': 'learner', 'arm': 'same', 'upper_bound_usd': '3.0'}]
    ledger.write_text(json.dumps(rows))
    for row in rows:
        (tmp_path / ('profile-same-' + row['phase'] + '-' + row['id'] + '.json')).write_text(json.dumps(
            {'profile': 'profile', 'arm': 'same', 'exit_code': 0, 'elapsed_seconds': 100}))
    module.settle_cpu_history(ledger, tmp_path, 'profile')
    assert json.loads(ledger.read_text()) == rows


def test_legacy_loopback_relay_keeps_real_payload_and_authenticates_upstream(monkeypatch):
    observed = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            observed.append((self.path, self.headers['Authorization'], self.rfile.read(int(self.headers['Content-Length']))))
            payload = b'{"actual_teacher":true}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    runner = load_runner(monkeypatch)
    relay = runner.legacy_loopback_gateway('test-ephemeral-key', 'http://127.0.0.1:' + str(upstream.server_port))
    url = 'http://127.0.0.1:' + str(relay.server_port)
    body = b'{"prompt":[3,4],"max_tokens":0,"echo":true}'
    try:
        assert relay.server_address[0] == '127.0.0.1'
        with urllib.request.urlopen(urllib.request.Request(url + '/v1/completions', data=body)) as response:
            assert json.load(response) == {'actual_teacher': True}
        assert observed == [('/v1/completions', 'Bearer test-ephemeral-key', body)]
        with urllib.request.urlopen(url + '/_receipts') as response:
            receipts = json.load(response)
        assert len(receipts) == 1 and receipts[0]['status'] == 200
    finally:
        relay.shutdown()
        relay.server_close()
        upstream.shutdown()
        upstream.server_close()
