# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import datetime as dt
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.arc_agi_3_agent import app as agent_app
from responses_api_agents.arc_agi_3_agent.app import (
    ArcAgi3Agent,
    ArcAgi3AgentConfig,
    ArcAgi3AgentRunRequest,
    redact,
    repo_path,
)
from responses_api_agents.arc_agi_3_agent.harness_runtime import AttemptResult, GameStartError


class FakeHttpResponse:
    def __init__(self, payload: dict, status: int = 200, cookies=None):
        self._payload = payload
        self.status = status
        self.ok = status < 400
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class FakeRuntime:
    def __init__(self, attempt=None, raises=None):
        self.attempt, self.raises = attempt, raises
        self.run_dir = Path("/tmp/arc3-run")
        self.run_config_sha256 = "c0ffee"
        self.start_time = None
        self.start_time_utc = None
        self.api_key = "key-1"
        self.calls = []

    async def play_game(self, meta, index, transport):
        self.calls.append((meta["full_game_id"], index))
        if self.raises:
            raise self.raises
        return self.attempt


def make_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides) -> ArcAgi3Agent:
    monkeypatch.setattr(agent_app, "ensure_flask", lambda: None)
    monkeypatch.setattr(agent_app, "ensure_harness", lambda source, dest: tmp_path / "harness")
    config = ArcAgi3AgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="arc_agi_3_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="arc_agi_3_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        environments_dir=str(tmp_path / "games"),
        **overrides,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.get = AsyncMock(return_value=FakeHttpResponse({"url": "http://game-server.invalid", "games": 25}))
    return ArcAgi3Agent(config=config, server_client=server_client)


def run_request() -> ArcAgi3AgentRunRequest:
    return ArcAgi3AgentRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "Play ls20"}]},
        verifier_metadata={"game_id": "ls20", "full_game_id": "ls20-9607627b", "dispatch_index": 3},
    )


def test_config_defaults_and_hyperparameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path, temperature=1.0, request_seed=5, context_keep_tokens=12288)
    assert agent.config.play_seconds == 7200 and agent.config.concurrency == 25
    assert agent.config.chat_template_kwargs == {}
    hyper = agent.hyperparameters()
    assert hyper["temperature"] == 1.0 and hyper["request_seed"] == 5 and hyper["top_p"] is None
    assert "seed" not in hyper
    assert agent.run_dir().parent == PARENT_DIR / "results" / "arc_agi_3"
    assert make_agent(monkeypatch, tmp_path, run_name="x").run_dir().name == "x"


def test_one_worker_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_workers must be 1"):
        make_agent(monkeypatch, tmp_path, num_workers=2)


def test_repo_path() -> None:
    assert repo_path("/abs/dir") == Path("/abs/dir")
    assert repo_path("rel/dir") == PARENT_DIR / "rel/dir"


def test_run_sends_record_to_verifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    record = {"state": "gave_up", "levels_completed": 1, "history": [{}, {}]}
    runtime = FakeRuntime(
        attempt=AttemptResult(record=record, persistence_error="OSError: full", scorecard_id="card-1", guid="guid-1")
    )
    runtime.start_time = dt.datetime(2026, 11, 4, 12, 0, 0)
    runtime.start_time_utc = "2026-11-04T12:00:00+00:00"
    agent._runtime = runtime
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(server_name=server_name, url_path=url_path, body=json)
        return FakeHttpResponse({**json, "reward": 0.5, "rhae": 50.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    assert runtime.calls == [("ls20-9607627b", 3)]
    assert sent["server_name"] == "arc_agi_3_resources_server" and sent["url_path"] == "/verify"
    assert sent["body"]["game_record"] == record and sent["body"]["agent_error"] is None
    # F1: a save failure travels as persistence_error next to the intact record; F7: the run clock origin goes along.
    assert sent["body"]["persistence_error"] == "OSError: full"
    assert sent["body"]["benchmark_start_time"] == "2026-11-04T12:00:00+00:00"
    # The authority: the attempt's scorecard, its game run and the key that can close the card.
    assert sent["body"]["scorecard_id"] == "card-1" and sent["body"]["game_guid"] == "guid-1"
    assert sent["body"]["scorecard_api_key"] == "key-1"
    summary = json.loads(sent["body"]["response"]["output"][0]["content"][0]["text"])
    assert summary == {
        "game_id": "ls20",
        "state": "gave_up",
        "levels_completed": 1,
        "actions": 2,
        "run_dir": "/tmp/arc3-run",
    }
    assert result.reward == 0.5 and result.run_dir == "/tmp/arc3-run" and result.run_config_sha256 == "c0ffee"
    assert result.harness == "nvarc3-kaggle-60.95"


def test_run_scores_a_game_that_cannot_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    agent._runtime = FakeRuntime(raises=GameStartError("pins differ"))
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(body=json)
        return FakeHttpResponse({**json, "reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    assert sent["body"]["game_record"] is None
    assert sent["body"]["agent_error"] == "GameStartError: pins differ"
    assert result.reward == 0.0


def test_post_model_and_aggregate_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)

    class Morsel:
        value = "abc"

    calls = []

    async def post(server_name, url_path, json, cookies=None):
        calls.append((server_name, url_path, cookies))
        if url_path == "/aggregate_metrics":
            return FakeHttpResponse({"agent_metrics": {"rhae": 12.5}, "key_metrics": {"rhae": 12.5}})
        return FakeHttpResponse({"choices": []}, cookies={"session": Morsel()})

    agent.server_client.post = AsyncMock(side_effect=post)
    status, text, cookies = asyncio.run(agent._post_model({"messages": []}, {"session": "old"}))
    assert status == 200 and json.loads(text) == {"choices": []} and cookies == {"session": "abc"}
    assert calls[0] == ("policy_model", "/v1/chat/completions", {"session": "old"})
    metrics = asyncio.run(agent.aggregate_metrics(MagicMock()))
    assert calls[1][:2] == ("arc_agi_3_resources_server", "/aggregate_metrics")
    assert metrics.key_metrics == {"rhae": 12.5}


def test_other_errors_are_not_scored_as_not_started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F1: only GameStartError means "never started"; anything else fails the rollout loudly.
    agent = make_agent(monkeypatch, tmp_path)
    agent._runtime = FakeRuntime(raises=OSError("disk"))
    agent.server_client.post = AsyncMock()
    with pytest.raises(OSError):
        asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    agent.server_client.post.assert_not_called()


def test_requires_context_error_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F3: without the model server's context-error passthrough the agent refuses to start.
    agent = make_agent(monkeypatch, tmp_path)
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"vllm_model": {"base_url": ["http://x/v1"]}}}
    }
    with pytest.raises(RuntimeError, match="propagate_context_overflow_errors"):
        asyncio.run(agent._ensure_runtime())


def test_redact() -> None:
    config = {
        "api_key": "sk-123",  # pragma: allowlist secret
        "base_url": ["http://user:pw@host:8000/v1?token=abc&x=1"],
        "nested": {"Authorization": "Bearer t", "hf_token": "", "ok": 3, "max_tokens": 9, "thinking_token_budget": 8},
        "other": object,
    }
    out = redact(config)
    assert out["api_key"] == "REDACTED" and out["nested"]["Authorization"] == "REDACTED"
    assert out["base_url"] == ["http://REDACTED@host:8000/v1?token=REDACTED&x=1"]
    assert out["nested"]["hf_token"] == "" and out["nested"]["ok"] == 3 and isinstance(out["other"], str)
    assert out["nested"]["max_tokens"] == 9 and out["nested"]["thinking_token_budget"] == 8
    assert redact({"HF_TOKEN": "x", "access-token": "y"}) == {"HF_TOKEN": "REDACTED", "access-token": "REDACTED"}


def test_run_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F8: the effective configuration, redacted, with the engine's own report (or why it could not be read).
    agent = make_agent(monkeypatch, tmp_path, seed=3, extra_body={"thinking_token_budget": 8192})
    monkeypatch.setattr(agent_app, "load_manifest", lambda: {"provenance": {"tool_agent_py_sha256": "22a2ae7b"}})
    (tmp_path / "games" / "ls20" / "v1").mkdir(parents=True)
    (tmp_path / "games" / "ls20" / "v1" / "metadata.json").write_text("{}")

    async def unreachable(method, url, **kwargs):
        raise ConnectionError("engine down")

    monkeypatch.setattr(agent_app, "request", unreachable)
    model_config = {"base_url": ["http://h:8000/v1"], "api_key": "secret", "propagate_context_overflow_errors": True}
    config = asyncio.run(agent.run_config({"LOCAL_ANALYZER_TEMPERATURE": "0.7"}, model_config))
    assert config["model_server"]["config"]["api_key"] == "REDACTED"
    assert config["engine_observed"] == {"error": "ConnectionError: engine down"}
    assert config["transport"]["extra_body"] == {"thinking_token_budget": 8192} and config["agent"]["seed"] == 3
    assert config["environments"]["games"] == {"ls20": {"version": "v1", "files_sha256": {}}}
    assert config["harness"]["provenance"]["tool_agent_py_sha256"] == "22a2ae7b"


def test_lifespan_shutdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F6: the server's shutdown stops and finalizes the runtime.
    from fastapi.testclient import TestClient

    agent = make_agent(monkeypatch, tmp_path)
    calls = []

    class Runtime:
        async def shutdown(self, transport):
            calls.append(transport)

    agent._runtime, agent._transport = Runtime(), "transport"
    with TestClient(agent.setup_webserver()):
        pass
    assert calls == ["transport"]


class StartupRuntime:
    """A HarnessRuntime stand-in whose start blocks until released (or raises)."""

    created: list = []

    def __init__(self, **kwargs):
        self.run_dir = Path("/tmp/arc3-run")
        self.started, self.release = threading.Event(), threading.Event()
        self.fail = kwargs.pop("_fail", False)
        self.shutdowns, self.aborted = [], False
        StartupRuntime.created.append(self)

    def start(self, proxy, run_config):
        self.started.set()
        assert self.release.wait(5), "the test never released the start"
        if self.fail:
            raise OSError("harness import failed")

    def abort_start(self):
        self.aborted = True

    async def shutdown(self, transport):
        self.shutdowns.append(transport)


def patch_runtime_start(monkeypatch: pytest.MonkeyPatch, fail_first: bool = False) -> list:
    StartupRuntime.created = []
    created = StartupRuntime.created

    def factory(**kwargs):
        return StartupRuntime(_fail=fail_first and not created, **kwargs)

    async def run_config(self, env, model_config):
        return {}

    monkeypatch.setattr(ArcAgi3Agent, "model_server_config", lambda self: {"propagate_context_overflow_errors": True})
    monkeypatch.setattr(ArcAgi3Agent, "resolve_model_base_url", lambda self, name: "http://policy/v1")
    monkeypatch.setattr(ArcAgi3Agent, "run_config", run_config)
    monkeypatch.setattr(agent_app, "load_harness_env_toml", lambda root: {})
    monkeypatch.setattr(agent_app, "compose_harness_env", lambda base, **kwargs: {})
    monkeypatch.setattr(agent_app, "HarnessRuntime", factory)
    return created


async def wait_for(predicate, seconds: float = 5.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def test_cancel_during_runtime_start_keeps_one_owned_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F6 residual: the first request is cancelled while the runtime starts. The start keeps running under the agent;
    # the next request waits for the same runtime (no second one), and shutdown waits for the start, then closes it.
    agent = make_agent(monkeypatch, tmp_path)
    created = patch_runtime_start(monkeypatch)

    async def main():
        first = asyncio.ensure_future(agent._ensure_runtime())
        await wait_for(lambda: created and created[0].started.is_set())
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.ensure_future(agent._ensure_runtime())
        await asyncio.sleep(0.05)
        assert not second.done() and len(created) == 1
        shutdown = asyncio.ensure_future(agent.shutdown())
        await asyncio.sleep(0.05)
        assert not shutdown.done()  # waits for the start (a thread cannot be interrupted)
        created[0].release.set()
        runtime = await second
        await shutdown
        assert runtime is created[0] and agent._runtime is runtime and len(created) == 1
        assert runtime.shutdowns == [agent._transport] and agent._transport is not None
        with pytest.raises(RuntimeError, match="shutting down"):
            await agent._ensure_runtime()

    asyncio.run(main())


def test_failed_runtime_start_is_torn_down_and_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A start that raises tears down what it built; the request sees the error; the next request starts afresh.
    agent = make_agent(monkeypatch, tmp_path)
    created = patch_runtime_start(monkeypatch, fail_first=True)

    async def main():
        first = asyncio.ensure_future(agent._ensure_runtime())
        await wait_for(lambda: created and created[0].started.is_set())
        created[0].release.set()
        with pytest.raises(OSError, match="harness import failed"):
            await first
        assert created[0].aborted and agent._runtime is None and agent._transport is None
        second = asyncio.ensure_future(agent._ensure_runtime())
        await wait_for(lambda: len(created) == 2 and created[1].started.is_set())
        created[1].release.set()
        assert await second is created[1] and not created[1].aborted

    asyncio.run(main())
