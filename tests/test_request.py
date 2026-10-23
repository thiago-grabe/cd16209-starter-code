"""Tests for request.py.

Prediction correctness is covered in test_api.py with the real artifacts.
The fakes here only simulate server behaviour to exercise the exit codes;
the last test performs a real HTTP round trip against uvicorn.
"""

import socket
import threading
import time

import requests
import uvicorn

import main
import request as request_script
from starter import examples

BASE = "http://api.test:1"


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeHttp:
    """Records calls and replays scripted responses or exceptions."""

    def __init__(self, get_response, post_responses):
        self.get_response = get_response
        self.post_responses = list(post_responses)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, timeout))
        if isinstance(self.get_response, Exception):
            raise self.get_response
        return self.get_response

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, timeout))
        return self.post_responses.pop(0)


WELCOME = FakeResponse(200, {"message": "Welcome"})
HIGH = FakeResponse(200, {"prediction": ">50K", "probability_over_50k": 0.9})
LOW = FakeResponse(200, {"prediction": "<=50K", "probability_over_50k": 0.1})


def test_parse_args_defaults():
    args = request_script.parse_args([])
    assert args.base_url == "http://127.0.0.1:8000"
    assert args.timeout == 10.0


def test_main_success_prints_requests_statuses_and_bodies(capsys):
    fake = FakeHttp(WELCOME, [HIGH, LOW])
    assert request_script.main(["--base-url", BASE], http=fake) == 0
    out = capsys.readouterr().out
    assert f"GET {BASE}/" in out
    assert f"POST {BASE}/predict" in out
    assert "payload:" in out
    assert out.count("status: 200") == 3
    assert '">50K"' in out
    assert '"<=50K"' in out
    assert [call[0] for call in fake.calls] == ["GET", "POST", "POST"]
    assert all(call[2] == 10.0 for call in fake.calls)


def test_main_returns_1_on_bad_get_status():
    fake = FakeHttp(FakeResponse(500, {"detail": "boom"}), [HIGH, LOW])
    assert request_script.main(["--base-url", BASE], http=fake) == 1


def test_main_returns_1_on_bad_post_status():
    broken = FakeResponse(500, {"detail": "boom"})
    fake = FakeHttp(WELCOME, [broken, LOW])
    assert request_script.main(["--base-url", BASE], http=fake) == 1


def test_main_returns_1_on_wrong_prediction(capsys):
    fake = FakeHttp(WELCOME, [LOW, LOW])
    assert request_script.main(["--base-url", BASE], http=fake) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_main_handles_non_json_body(capsys):
    html = FakeResponse(200, None, "<html>oops</html>")
    fake = FakeHttp(html, [HIGH, LOW])
    assert request_script.main(["--base-url", BASE], http=fake) == 1
    assert "<html>oops</html>" in capsys.readouterr().out


def test_main_returns_1_on_non_json_prediction_body(capsys):
    html = FakeResponse(200, None, "<html>oops</html>")
    fake = FakeHttp(WELCOME, [html, LOW])
    assert request_script.main(["--base-url", BASE], http=fake) == 1
    out = capsys.readouterr().out
    assert "<html>oops</html>" in out
    assert "MISMATCH" in out


def test_main_returns_2_on_connection_error(capsys):
    fake = FakeHttp(requests.ConnectionError("refused"), [])
    assert request_script.main(["--base-url", BASE], http=fake) == 2
    assert "Could not reach" in capsys.readouterr().err


def test_main_strips_trailing_slash_and_uses_timeout_option():
    fake = FakeHttp(WELCOME, [HIGH, LOW])
    argv = ["--base-url", BASE + "/", "--timeout", "3"]
    assert request_script.main(argv, http=fake) == 0
    assert fake.calls[0][1] == f"{BASE}/"
    assert all(call[2] == 3.0 for call in fake.calls)


def test_build_session_ignores_proxy_env():
    session = request_script.build_session()
    assert isinstance(session, requests.Session)
    assert session.trust_env is False


def test_payloads_match_shared_examples():
    assert request_script.HIGH_INCOME_EXAMPLE == examples.HIGH_INCOME_EXAMPLE
    assert request_script.LOW_INCOME_EXAMPLE == examples.LOW_INCOME_EXAMPLE


def test_request_script_against_live_uvicorn(artifacts):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        main.app, host="127.0.0.1", port=port, log_level="warning",
        log_config=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    try:
        argv = ["--base-url", f"http://127.0.0.1:{port}"]
        assert request_script.main(argv) == 0
    finally:
        server.should_exit = True
        thread.join(10)
