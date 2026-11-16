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
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from responses_api_agents.arc_agi_3_pivot_agent.app import (
    ArcAgi3PivotAgent,
    ArcAgi3PivotAgentConfig,
    ArcAgi3PivotRunRequest,
    from_gym_reply,
    to_gym_message,
)


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status: int = 200):
        self.payload, self.status, self.cookies = payload, status, {}
        self.ok = status < 400
        self.content = self

    async def read(self):
        return json.dumps(self.payload).encode()

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class Servers:
    """The pivot resources server and the policy model server: a window that ends on the second step."""

    def __init__(self, restore_error=None, model_fails=False):
        self.calls: List[tuple] = []
        self.restore_error, self.model_fails = restore_error, model_fails
        self.steps = 0

    async def post(self, server_name: str, url_path: str, json: Any = None, cookies=None, **_):
        self.calls.append((server_name, url_path, json))
        if url_path == "/seed_session":
            if self.restore_error:
                return FakeResponse({"restore_error": self.restore_error})
            return FakeResponse(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "old thought",
                            "tool_calls": [
                                {
                                    "id": "r1",
                                    "type": "function",
                                    "function": {"name": "python", "arguments": "{}", "index": 0},
                                }
                            ],
                        },
                        {"role": "user", "content": [{"type": "text", "text": "board"}]},
                    ],
                    "tools": [{"type": "function", "function": {"name": "python"}}],
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
                }
            )
        if url_path.endswith("/v1/chat/completions"):
            if self.model_fails:
                return FakeResponse({"error": "boom"}, status=500)
            n = sum(1 for c in self.calls if c[1].endswith("/chat/completions"))
            message = {
                "role": "assistant",
                "content": f"<think>plan {n}</think>",
                "tool_calls": [
                    {"id": f"c{n}", "type": "function", "function": {"name": "python", "arguments": '{"code": "x"}'}}
                ],
            }
            return FakeResponse(
                {
                    "choices": [{"message": message, "finish_reason": "tool_calls"}],
                    "usage": {"completion_tokens": 11, "prompt_tokens": 900},
                }
            )
        if url_path == "/step":
            self.steps += 1
            done = self.steps >= 2
            messages = [{"role": "tool", "tool_call_id": json["message"]["tool_calls"][0]["id"], "content": "ok"}]
            if not done:
                messages.append({"role": "user", "content": "opener"})
            return FakeResponse({"messages": messages, "done": done, "call": {}})
        if url_path == "/verify":
            return FakeResponse(json | {"reward": 2.0})
        raise AssertionError(url_path)


def make_agent(servers: Servers) -> ArcAgi3PivotAgent:
    config = ArcAgi3PivotAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="arc_agi_3_pivot_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="pivot"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    agent = ArcAgi3PivotAgent(config=config, server_client=MagicMock(spec=ServerClient))
    agent.server_client.post = servers.post
    return agent


def run_body() -> ArcAgi3PivotRunRequest:
    return ArcAgi3PivotRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "x"}]},
            "verifier_metadata": {"pivot_id": "p1", "pivot": {"pivot_json": "{}", "harness": "30d2cc8b"}},
        }
    )


def request() -> MagicMock:
    r = MagicMock()
    r.cookies = {}
    return r


async def test_window_loop_until_done():
    servers = Servers()
    result = await make_agent(servers).run(request(), run_body())
    paths = [c[1] for c in servers.calls]
    assert paths == ["/seed_session", "/v1/chat/completions", "/step", "/v1/chat/completions", "/step", "/verify"]

    first = servers.calls[1][2]
    # History reasoning goes as a <think> prefix; top_k and the template kwargs through metadata.
    assert first["messages"][1]["content"] == "<think>old thought</think>"
    # Recorded engine extras (a streaming index) are dropped; the Gym model server refuses them.
    assert first["messages"][1]["tool_calls"] == [
        {"id": "r1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
    ]
    assert json.loads(first["metadata"]["extra_body"]) == {"top_k": 20}
    assert json.loads(first["metadata"]["chat_template_kwargs"]) == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert first["temperature"] == 1.0 and first["max_tokens"] == 12288 and first["tool_choice"] == "auto"

    step = servers.calls[2][2]
    assert step["message"]["reasoning_content"] == "plan 1" and step["message"]["content"] == ""
    assert step["gen_tokens"] == 11 and step["prompt_tokens"] == 900 and step["finish_reason"] == "tool_calls"

    # The second call extends the first: reply, tool result, opener appended in order.
    second = servers.calls[3][2]["messages"]
    assert [m["role"] for m in second[3:]] == ["assistant", "tool", "user"]
    assert second[3]["content"] == "<think>plan 1</think>"

    verify = servers.calls[-1][2]
    assert verify["agent_error"] is None
    window = json.loads(verify["response"]["output"][0]["content"][0]["text"])
    assert [m["role"] for m in window] == ["assistant", "tool", "user", "assistant", "tool"]
    assert result.reward == 2.0


async def test_restore_error_goes_straight_to_verify():
    servers = Servers(restore_error="RestoreError: x")
    await make_agent(servers).run(request(), run_body())
    assert [c[1] for c in servers.calls] == ["/seed_session", "/verify"]


async def test_model_failure_is_reported_not_stepped():
    servers = Servers(model_fails=True)
    await make_agent(servers).run(request(), run_body())
    assert [c[1] for c in servers.calls] == ["/seed_session", "/v1/chat/completions", "/verify"]
    assert "HTTP 500" in servers.calls[-1][2]["agent_error"]


@pytest.mark.parametrize(
    "message,expected",
    [
        ({"content": "<think>r</think>answer"}, ("r", "answer")),
        ({"content": "answer", "reasoning_content": "r"}, ("r", "answer")),
        ({"content": "answer"}, (None, "answer")),
    ],
)
def test_reasoning_split(message, expected):
    out = from_gym_reply(message)
    assert (out.get("reasoning_content"), out["content"]) == expected
    if expected[0]:
        assert to_gym_message(out)["content"] == f"<think>{expected[0]}</think>{expected[1]}"
