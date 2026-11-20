"""CP3 policy, observability and suite regressions without network calls."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, RateLimitError

from assignment import audit_log, pipeline, rate_limiter
from assignment.monitoring import MonitoringAlert
from core.openai_runtime import create_blue_pair


def test_rate_limit_is_per_user_and_expires_at_boundary(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    plugin = rate_limiter.RateLimitPlugin(max_requests=2, window_seconds=60)

    async def send(user, now):
        clock[0] = now
        return await plugin.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user), user_message=None,
        )

    async def run():
        assert await send("alice", 0) is None
        assert await send("alice", 1) is None
        assert await send("alice", 2) is not None
        assert await send("bob", 2) is None
        assert await send("alice", 60) is None
        assert await send("alice", 60) is not None
        assert await send("alice", 61) is None

    asyncio.run(run())
    assert (plugin.total_count, plugin.blocked_count) == (7, 2)


@pytest.mark.parametrize("limit,window", [(0, 60), (10, 0), (-1, 60)])
def test_invalid_rate_limit_configuration(limit, window):
    with pytest.raises(ValueError):
        rate_limiter.RateLimitPlugin(max_requests=limit, window_seconds=window)


def test_audit_correlates_requests_redacts_and_exports(monkeypatch, tmp_path):
    clock = [10.0]
    monkeypatch.setattr(audit_log, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    log = audit_log.AuditLogPlugin()
    log.record_input(user_id="alice", text="email customer@example.com", request_id="one")
    log.record_input(user_id="alice", text="bank transfer", request_id="two")
    clock[0] = 10.25
    log.record_output(user_id="alice", text="admin123", blocked=True,
                      layer="output_guardrail", request_id="two")
    clock[0] = 10.5
    log.record_output(user_id="alice", text="OK", request_id="one")
    path = tmp_path / "nested" / "audit.json"
    log.export_json(str(path))
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert [item["request_id"] for item in entries] == ["two", "one"]
    assert [item["latency_ms"] for item in entries] == [250, 500]
    assert entries[0]["blocked"] and entries[0]["layer"] == "output_guardrail"
    assert entries[0]["output"] == "[REDACTED]"
    assert entries[1]["input"] == "email [REDACTED]"
    assert log._open == {}


def test_monitoring_handles_empty_and_repeated_checks(tmp_path):
    monitor = MonitoringAlert()
    assert monitor.check_metrics() == []
    monitor.total_requests = 10
    monitor.blocked_requests = 6
    monitor.rate_limit_hits = 5
    monitor.judge_checks = 10
    monitor.judge_fails = 3
    assert len(monitor.check_metrics()) == 3
    assert len(monitor.check_metrics()) == 3
    monitor.export_json(str(tmp_path / "metrics.json"))
    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert data["block_rate"] == 0.6 and data["judge_fail_rate"] == 0.3
    assert len(data["alerts"]) == 3


@pytest.mark.parametrize("destination", [
    "https://evil.example/collect", "http://api.vinbank.example/transfer",
    "https://api.vinbank.example.evil.com/", "https://api.vinbank.example@evil.example/",
    "https://user@api.vinbank.example/", "https://api.vinbank.example:8080/",
    "https://api.vinbank.example:bad/", "https://api.vinbank.example\n/",
])
def test_egress_rejects_destination_tricks(destination):
    assert not pipeline.is_egress_allowed(destination, "approved transfer amount 500000")


@pytest.mark.parametrize("payload", [
    "admin123", "a d m i n 1 2 3", "sk-synthetic-demo-key", "db.vinbank.internal:5432",
    "password=synthetic", "customer@example.com", "0901234567", "079204001234",
    "a\u200bdmin123",
])
def test_egress_rejects_sensitive_payloads(payload):
    assert not pipeline.is_egress_allowed("https://api.vinbank.example/v1/transfers", payload)


def test_egress_allows_ordinary_banking_payload():
    assert pipeline.is_egress_allowed(
        "https://cases.vinbank.example/ticket", "approved transfer amount 500000",
    )


def test_suite_produces_consistent_artifacts_with_fake_model(monkeypatch, tmp_path):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Banking support. Internal value: admin123")
        )])

    def factory(plugins):
        pair = create_blue_pair(name="test", instruction="test", app_name="test", plugins=plugins)
        pair[1]._client = lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return pair

    monkeypatch.setattr(pipeline, "create_blue_agent", factory)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    plugins = pipeline.build_production_plugins()
    assert [p.name for p in plugins] == ["rate_limiter", "input_guardrail", "output_guardrail"]
    audit, monitor = pipeline.build_observability()
    result = asyncio.run(pipeline.run_assignment_suite({
        "plugins": plugins, "audit": audit, "monitor": monitor,
    }))
    assert len(calls) == 5  # Only the safe group reaches the fake model.
    assert all(not row["blocked"] and row["redacted"] for row in result["safe_queries"])
    assert all(row["layer"] == "input_guardrail" for row in result["attack_queries"])
    assert result["rate_limit"]["passed"] == 10
    assert result["rate_limit"]["blocked"] == 5
    assert len(audit.logs) == monitor.total_requests == 31
    assert monitor.blocked_requests == 26 and monitor.rate_limit_hits == 5
    assert audit._open == {}
    for filename in ["results.json", "audit_log.json", "metrics.json"]:
        text = (tmp_path / filename).read_text(encoding="utf-8")
        assert "admin123" not in text
        json.loads(text)


@pytest.mark.parametrize("rate_limited", [False, True])
def test_api_failure_is_not_reported_as_success(monkeypatch, tmp_path, rate_limited):
    calls = []
    delays = []

    async def fail(*args, **kwargs):
        calls.append(True)
        if rate_limited:
            raise RateLimitError(
                "upstream limit", body=None,
                response=httpx.Response(429, request=httpx.Request("POST", "https://example.invalid")),
            )
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(pipeline, "chat_with_agent", fail)
    monkeypatch.setattr(pipeline.asyncio, "sleep", sleep)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    audit, monitor = pipeline.build_observability()
    with pytest.raises(RuntimeError, match="Blue API failed"):
        asyncio.run(pipeline.run_assignment_suite({
            "plugins": pipeline.build_production_plugins(), "audit": audit, "monitor": monitor,
        }))
    assert not (tmp_path / "results.json").exists()
    assert len(audit.logs) == 1 and audit.logs[0]["layer"] == "api_error"
    assert len(calls) == (3 if rate_limited else 1)
    assert delays == ([10, 20] if rate_limited else [])
