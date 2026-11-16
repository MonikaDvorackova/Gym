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
"""The server against a stand-in for AF3's env_proxy (the real one needs the pool, the games and the harness)."""

import dataclasses
import json
import sys
import types
from http.cookiejar import CookieJar
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core.failure_kinds import ENVIRONMENT_PREPARE_FAILED, MODEL_CONTEXT_LIMIT, MODEL_REQUEST_ERROR
from resources_servers.arc_agi_3_pivot import create_dataset
from resources_servers.arc_agi_3_pivot.app import (
    ArcAgi3PivotConfig,
    ArcAgi3PivotResourcesServer,
    assistant_reply_text,
)


@dataclasses.dataclass
class CallInfo:
    actions: List[str] = dataclasses.field(default_factory=list)
    level_completed: bool = False
    game_over: bool = False
    tool_error: Any = None
    discard: bool = False
    d_before: Any = None
    d_after: Any = None


class FakeEnv:
    """d0 = 5; each acting reply lowers d by 1; a reply whose code says "win" completes the level; "huge" overflows."""

    instances: List["FakeEnv"] = []
    fail_restore = False

    def __init__(self, row, games, **kwargs):
        self.row, self.games, self.kwargs = row, games, kwargs
        self.d0, self.d, self.closed, self.texts = 5, 5, False, []
        FakeEnv.instances.append(self)

    def reset(self):
        if FakeEnv.fail_restore:
            raise RuntimeError("RestoreError: frame_sha256 mismatch")
        return {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "board"}],
            "tools": [{"type": "function", "function": {"name": "python"}}],
            "chat_template_kwargs": {"enable_thinking": True},
        }

    def step(self, text, gen_tokens=None, finish_reason="stop", prompt_tokens=None):
        self.texts.append(text)
        reply = json.loads(text)
        info = CallInfo(d_before=self.d)
        new = []
        for i, call in enumerate(reply["tool_calls"]):
            code = call["function"]["arguments"]["code"]
            new.append({"role": "tool", "tool_call_id": f"env_{len(self.texts)}_{i}", "content": f"ran {code}"})
            info.actions.append("ACTION1")
            if "win" in code:
                info.level_completed = True
            elif "huge" in code:
                info.discard = True
            else:
                self.d -= 1
        if info.level_completed:
            self.d = 0
        info.d_after = self.d
        done = info.level_completed or info.discard or len(self.texts) >= 3
        if not done:
            new.append({"role": "user", "content": "next opener"})
        return new, done, info

    def close(self):
        self.closed = True


@pytest.fixture
def af3(monkeypatch):
    """Stand-in modules for agent_factory_3.envs.arc3.{env_proxy, env}."""
    FakeEnv.instances, FakeEnv.fail_restore = [], False
    prefetched = []
    env_proxy = types.SimpleNamespace(
        Arc3WindowEnvProxy=FakeEnv,
        prefetch=lambda row, games, **kw: prefetched.append(json.loads(row["pivot_json"])["id"]) or "started",
    )
    env = types.SimpleNamespace(reply_text=lambda message: json.dumps(message))
    names = ["agent_factory_3", "agent_factory_3.envs", "agent_factory_3.envs.arc3"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "agent_factory_3.envs.arc3.env_proxy", env_proxy)
    monkeypatch.setitem(sys.modules, "agent_factory_3.envs.arc3.env", env)
    sys.modules["agent_factory_3.envs.arc3"].env_proxy = env_proxy
    sys.modules["agent_factory_3.envs.arc3"].env = env
    return prefetched


def meta(pivot_id: str) -> dict:
    return {
        "pivot_id": pivot_id,
        "game_id": "ls20-9607627b",
        "pivot": {"pivot_json": json.dumps({"id": pivot_id}), "harness": "30d2cc8b", "harness_env": "env/x.toml"},
    }


def make_client(tmp_path, **overrides):
    (tmp_path / "pool").mkdir(exist_ok=True)
    (tmp_path / "pool" / "games.parquet").write_bytes(b"")
    config = ArcAgi3PivotConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        pool_dir=str(tmp_path / "pool"),
        af3_src=str(tmp_path),
        harness_base=str(tmp_path),
        **overrides,
    )
    server = ArcAgi3PivotResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    class StatelessCookieJar(CookieJar):
        def extract_cookies(self, response: object, request: object) -> None:
            pass

    return server, TestClient(server.setup_webserver(), cookies=StatelessCookieJar())


def reply(code: str, call_id: str = "model_call_1") -> dict:
    return {
        "role": "assistant",
        "reasoning_content": "think",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }
        ],
    }


def verify_body(m: dict, **extra) -> dict:
    response = {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [],
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": False,
    }
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "x"}]},
        "verifier_metadata": m,
        "response": response,
        **extra,
    }


def seed(client, m):
    response = client.post("/seed_session", json={"verifier_metadata": m})
    assert response.status_code == 200, response.text
    return response.json(), response.cookies


def test_window_progress_reward_and_tool_ids(af3, tmp_path):
    server, client = make_client(tmp_path, max_calls=24)
    prompt, cookies = seed(client, meta("p1"))
    assert [m["role"] for m in prompt["messages"]] == ["system", "user"]
    assert prompt["chat_template_kwargs"] == {"enable_thinking": True} and prompt["restore_error"] is None
    env = FakeEnv.instances[0]
    assert env.kwargs == {"harness_base": str(tmp_path), "max_calls": 24}
    assert env.row == {k: meta("p1")["pivot"][k] for k in ("pivot_json", "harness", "harness_env")}

    step = client.post("/step", json={"message": reply("a = 1"), "gen_tokens": 7}, cookies=cookies).json()
    assert step["done"] is False
    # The env's own call id is replaced by the model's.
    assert step["messages"][0] == {"role": "tool", "tool_call_id": "model_call_1", "content": "ran a = 1"}
    assert step["messages"][1]["role"] == "user"
    client.post("/step", json={"message": reply("b = 2"), "gen_tokens": 3}, cookies=cookies)
    last = client.post("/step", json={"message": reply("c = 3")}, cookies=cookies).json()
    assert last["done"] is True and [m["role"] for m in last["messages"]] == ["tool"]

    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["reward"] == 3.0 and result["progress"] == 3.0  # d 5 -> 2
    assert result["d0"] == 5 and result["d_end"] == 2 and result["level_completed"] == 0.0
    assert result["actions"] == 3.0 and result["calls"] == 3.0 and result["gen_tokens"] == 10.0
    assert result["mask_sample"] is False
    assert env.closed  # released after verify
    # The env got the reply as AF3's reply text (here the stand-in's JSON), tool arguments decoded.
    assert json.loads(env.texts[0])["tool_calls"][0]["function"]["arguments"] == {"code": "a = 1"}


def test_level_completion_scores_d0(af3, tmp_path):
    _, client = make_client(tmp_path)
    _, cookies = seed(client, meta("p1"))
    step = client.post("/step", json={"message": reply("win")}, cookies=cookies).json()
    assert step["done"] is True
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["reward"] == 5.0 and result["level_completed"] == 1.0 and result["d_end"] == 0


def test_restore_error_is_masked_not_scored(af3, tmp_path):
    FakeEnv.fail_restore = True
    _, client = make_client(tmp_path)
    prompt, cookies = seed(client, meta("p1"))
    assert "frame_sha256 mismatch" in prompt["restore_error"] and prompt["messages"] == []
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == ENVIRONMENT_PREPARE_FAILED
    assert FakeEnv.instances[0].closed


def test_context_cap_discard_is_masked(af3, tmp_path):
    _, client = make_client(tmp_path)
    _, cookies = seed(client, meta("p1"))
    assert client.post("/step", json={"message": reply("huge")}, cookies=cookies).json()["done"] is True
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == MODEL_CONTEXT_LIMIT


def test_unfinished_window_is_masked(af3, tmp_path):
    _, client = make_client(tmp_path)
    _, cookies = seed(client, meta("p1"))
    client.post("/step", json={"message": reply("a = 1")}, cookies=cookies)
    result = client.post("/verify", json=verify_body(meta("p1"), agent_error="TimeoutError"), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == MODEL_REQUEST_ERROR


def test_prefetch_endpoint(af3, tmp_path):
    _, client = make_client(tmp_path)
    response = client.post(
        "/prefetch", json={"rows": [{"verifier_metadata": meta("p7")}, meta("p8"), {"pivot_id": "x"}]}
    )
    assert response.json()["status"][:2] == ["started", "started"]
    assert response.json()["status"][2].startswith("invalid")
    assert sorted(af3) == ["p7", "p8"]


def test_read_ahead_prefetches_the_next_rows_once(af3, tmp_path):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("".join(json.dumps({"verifier_metadata": meta(f"p{i}")}) + "\n" for i in range(6)))
    _, client = make_client(tmp_path, prefetch_dataset=str(dataset), prefetch_lookahead=2)
    seed(client, meta("p1"))
    seed(client, meta("p1"))  # the group's second rollout: nothing new to prefetch
    seed(client, meta("p2"))
    assert sorted(af3) == ["p2", "p3", "p4"]


def test_reply_text_drops_unparseable_tool_calls(af3):
    message = reply("x = 1")
    message["tool_calls"].append(
        {"id": "bad", "type": "function", "function": {"name": "python", "arguments": "{oops"}}
    )
    message["content"] = [{"type": "text", "text": "hi"}]
    parsed = json.loads(assistant_reply_text(message))
    assert [c["function"]["arguments"] for c in parsed["tool_calls"]] == [{"code": "x = 1"}]
    assert parsed["content"] == "hi" and parsed["reasoning_content"] == "think"


def test_create_dataset_keeps_exact_host_rows(tmp_path):
    rows = [
        {
            "pivot_id": "a",
            "game_id": "g1",
            "restore_mode": "snapshot",
            "restore_ok": True,
            "pivot_json": "{}",
            "harness": "30d2cc8b",
            "level": 1,
            "po_d0": 4,
        },
        {
            "pivot_id": "b",
            "game_id": "g1",
            "restore_mode": "kit_ckpt",
            "restore_ok": True,
            "pivot_json": "{}",
            "harness": "38d0f201",
        },
        {
            "pivot_id": "c",
            "game_id": "g1",
            "restore_mode": "snapshot",
            "restore_ok": True,
            "pivot_json": "{}",
            "harness": "30d2cc8b",
        },
        {
            "pivot_id": "d",
            "game_id": "g2",
            "restore_mode": "snapshot",
            "restore_ok": False,
            "pivot_json": "{}",
            "harness": "30d2cc8b",
        },
    ]
    kept = create_dataset.select(rows, limit=None, per_game=1, ids=None)
    assert [r["pivot_id"] for r in kept] == ["a"]
    row = create_dataset.gym_row(kept[0])
    assert row["verifier_metadata"]["pivot"] == {"pivot_json": "{}", "harness": "30d2cc8b"}
    assert row["verifier_metadata"]["po_d0"] == 4
