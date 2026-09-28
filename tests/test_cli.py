"""CLI smoke tests using subprocess."""
import json
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args, check=True):
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = REPO
    # Force-disable SOCKS proxy at subprocess level too
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        full_env.pop(k, None)
    return subprocess.run(
        [sys.executable, "main.py", *args],
        capture_output=True, text=True,
        cwd=REPO,
        env=full_env,
        check=check,
    )


class TestCLISmoke:
    def test_help_exits_zero(self):
        result = _run("--help")
        assert result.returncode == 0
        assert "laya-scam-detector" in result.stdout

    def test_route_chinese_routes_to_multilingual(self):
        result = _run("--route", "您好世界")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["routed_to"] == "multilingual"
        assert "cjk_han" in data["reason"]

    def test_route_english_routes_to_english(self):
        result = _run("--route", "Hello world")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["routed_to"] == "english"

    def test_no_args_exits_one(self):
        result = _run(check=False)
        assert result.returncode == 1

    def test_predict_without_questions_exits_two(self):
        result = _run("--predict", "test", check=False)
        assert result.returncode == 2
        assert "--questions" in result.stderr

    def test_eval_without_input_exits_two(self):
        result = _run("--eval", check=False)
        assert result.returncode == 2
        assert "--input" in result.stderr