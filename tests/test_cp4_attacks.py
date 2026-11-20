"""CP4 prompt and evidence checks using fake replies, without API requests."""
import asyncio
import json

import pytest

from attacks import attacks
from agents.guards_agent import detect_injection_strong
from core.config import get_llm_provider, get_model_name


def test_prompts_are_distinct_and_do_not_contain_expected_answers():
    prompts = attacks.adversarial_prompts
    assert len(prompts) >= 5
    assert len({row["id"] for row in prompts}) == len(prompts)
    assert len({row["category"] for row in prompts}) == len(prompts)
    for row in prompts:
        assert len(row["input"]) >= 200 and "TODO" not in row["input"]
        assert not attacks.response_leaked_secrets(row["input"])
    # At least one attempt should exercise NeMo rather than stop at regex.
    assert any(not detect_injection_strong(row["input"]) for row in prompts)


def test_both_targets_produce_consistent_artifacts(monkeypatch, tmp_path):
    async def reply(agent, runner, prompt):
        if agent == "red":
            return "Demo admin credential: admin123", None
        return "Request blocked by NeMo Guardrails (input rail: self check input).", None

    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    monkeypatch.setattr(attacks, "_repo_root", lambda: tmp_path)

    async def run():
        red = await attacks.run_attacks("red", None, target_name="red_default")
        advance = await attacks.run_attacks("advance", None, target_name="red_advance")
        return attacks.save_attack_results(unsafe_results=red, guards_results=advance)

    path = asyncio.run(run())
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["llm_provider"] == get_llm_provider()
    assert data["llm_model"] == get_model_name()
    assert len(data["unsafe_attacks"]) == len(data["guards_attacks"]) == 5
    assert data["summary"]["unsafe_leaked"] == 5
    assert data["summary"]["guards_leaked"] == 0
    assert data["summary"]["unsafe_errors"] == data["summary"]["guards_errors"] == 0
    for name in ("unsafe_attack_result.json", "guards_attack_result.json"):
        detail = json.loads((tmp_path / "outputs" / name).read_text(encoding="utf-8"))
        assert len(detail["results"]) == 5
        assert all(row["error"] is None for row in detail["results"])


@pytest.mark.parametrize("fails", [True, False])
def test_failed_or_empty_responses_are_errors_and_credentials_are_hidden(monkeypatch, tmp_path, fails):
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-private-credential")

    async def reply(*args):
        if fails:
            raise RuntimeError("invalid key unit-test-private-credential sk-private-demo")
        return "", None

    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(
        None, None, prompts=attacks.adversarial_prompts[:1],
        target_name="red_default", save_json=False,
    ))
    assert rows[0]["layer"] == "error" and not rows[0]["leaked"]
    path = attacks.save_attack_results(unsafe_results=rows, filepath=tmp_path / "errors.json")
    text = path.read_text(encoding="utf-8")
    assert "unit-test-private-credential" not in text and "sk-private-demo" not in text
    data = json.loads(text)
    assert data["unsafe_attacks"][0]["error"]
    assert data["summary"]["unsafe_errors"] == 1


def test_confirmation_requires_response_evidence():
    prompt = attacks.adversarial_prompts[3]["input"]
    assert not attacks.classify_attack_outcome(
        prompt, "MISMATCH: I cannot disclose internal values.", target_name="red_default",
    )["leaked"]


def test_nemo_failure_is_not_successful_defense_evidence():
    row = attacks.classify_attack_outcome(
        "What is my banking account?", "NeMo Guardrails input check failed.",
        target_name="red_advance",
    )
    assert row["layer"] == "nemo_error" and not row["leaked"]


@pytest.mark.parametrize("response", [
    "password: [REDACTED]", "Use password=EXAMPLE-ONLY",
    "Example key: sk-unrelated-sample-000", "No internal values available.",
])
def test_leak_requires_an_actual_protected_value(response):
    assert not attacks.response_leaked_secrets(response)


def test_timeout_preserves_previous_results_and_continues(monkeypatch, tmp_path):
    calls = []
    path = tmp_path / "attempts.json"

    async def reply(*args):
        calls.append(True)
        if len(calls) == 2:
            partial = json.loads(path.read_text(encoding="utf-8"))
            assert len(partial["results"]) == 1
            await asyncio.sleep(60)
        return "Demo value: admin123", None

    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(
        None, None, prompts=attacks.adversarial_prompts[:3], target_name="red_default",
        output_path=path, timeout_seconds=0.01,
    ))
    assert [row["layer"] for row in rows] == ["leaked", "error", "leaked"]
    assert "TimeoutError" in rows[1]["error"]
    assert len(json.loads(path.read_text(encoding="utf-8"))["results"]) == 3
