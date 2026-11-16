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
from nemo_gym.config import ModelServerRef
from responses_api_agents.arc_agi_3_agent.call_log import DRAIN_SUMMARY_PREFIX, GameCallLog, observations


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")
SYSTEM = {"role": "system", "content": "play"}


def turn(i):
    return [
        {"role": "user", "content": f"state {i}"},
        {"role": "assistant", "content": f"act {i}", "tool_calls": [{"id": f"c{i}"}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"},
    ]


def add(log, messages, response_id, *, role="gameplay", history_id="main", status=200, prompt_tokens=None):
    response = {"id": response_id, "usage": {"prompt_tokens": prompt_tokens}} if response_id else None
    log.add(role=role, history_id=history_id, messages=messages, status=status, response=response)


def bundle(log):
    return observations(log, invocation_id="g1", model_ref=MODEL, status="completed", stop_reason="not_won")


def records(result, kind):
    return [r for r in result.records if r.kind == kind]


def test_one_chain_has_no_compaction_and_lists_gameplay_calls_in_order():
    log = GameCallLog()
    history = [SYSTEM]
    for i in range(3):
        history = history + turn(i)
        add(log, history, f"r{i}")
    add(
        log,
        history + [{"role": "user", "content": "Summarize the level"}],
        "h0",
        role="helper",
        history_id="level-summary",
    )
    result = bundle(log)
    (invocation,) = records(result, "agent_invocation")
    assert [r.response_id for r in invocation.model_calls] == ["r0", "r1", "r2"]
    assert invocation.status == "completed" and invocation.stop_reason == "not_won"
    assert records(result, "context_compaction") == [] and result.gaps == []


def test_drain_with_summary_is_one_compaction_with_its_helper_call():
    log = GameCallLog()
    full = [SYSTEM] + turn(0) + turn(1) + turn(2)
    add(log, full, "r0", prompt_tokens=60000)
    add(
        log,
        full + [{"role": "user", "content": "summarize the dropped span"}],
        "s0",
        role="helper",
        history_id="drain-summary",
    )
    summary = {"role": "user", "content": f"{DRAIN_SUMMARY_PREFIX} (steps 0-1)]\nkeys open doors"}
    drained = [SYSTEM] + turn(2) + [summary] + turn(3)
    add(log, drained, "r1", prompt_tokens=20000)
    add(log, drained + turn(4), "r2", prompt_tokens=21000)
    result = bundle(log)
    (compaction,) = records(result, "context_compaction")
    assert compaction.before_model_call.response_id == "r0"
    assert compaction.after_model_call.response_id == "r1"
    assert [r.response_id for r in compaction.model_calls] == ["s0"]
    assert compaction.summary.startswith(DRAIN_SUMMARY_PREFIX) and "keys open doors" in compaction.summary
    assert (compaction.tokens_before, compaction.tokens_after) == (60000, 20000)
    assert compaction.outcome == "completed" and compaction.invocation_id == "g1"
    (invocation,) = records(result, "agent_invocation")
    assert [r.response_id for r in invocation.model_calls] == ["r0", "r1", "r2"]


def test_plain_drain_without_summary_is_a_compaction():
    log = GameCallLog()
    add(log, [SYSTEM] + turn(0) + turn(1), "r0")
    add(log, [SYSTEM] + turn(1), "r1")
    (compaction,) = records(bundle(log), "context_compaction")
    assert compaction.summary is None and compaction.model_calls == []


def test_rewrite_without_drop_is_a_gap_not_a_compaction():
    log = GameCallLog()
    add(log, [SYSTEM] + turn(0), "r0")
    add(log, [{"role": "system", "content": "play; lessons: keys open doors"}] + turn(0) + turn(1), "r1")
    result = bundle(log)
    assert records(result, "context_compaction") == []
    assert [g.code for g in result.gaps] == ["history_rewritten"]


def test_failed_call_is_a_gap_and_not_a_reference():
    log = GameCallLog()
    add(log, [SYSTEM] + turn(0), "r0")
    add(log, [SYSTEM] + turn(0) + turn(1), None, status=400)
    add(log, [SYSTEM] + turn(0) + turn(1), "r1")
    result = bundle(log)
    (invocation,) = records(result, "agent_invocation")
    assert [r.response_id for r in invocation.model_calls] == ["r0", "r1"]
    assert [g.code for g in result.gaps] == ["model_call_failed"]
    assert records(result, "context_compaction") == []
