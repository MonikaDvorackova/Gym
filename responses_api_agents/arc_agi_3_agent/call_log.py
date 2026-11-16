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
"""The model calls of one game and what they say about its context: chains, drains and helper calls.

The transport records every call of a game here (``GameCallLog.add``): its role, its ``gdn_history_id``, the
response id the model server returned, and a digest of every request message. The harness marks its own calls:
``gameplay`` calls play the game, ``helper`` calls write text the harness keeps (``drain-summary``,
``level-summary``, ``rolling-summary``, ``turn-compaction``).

:func:`observations` turns the log into the rollout's ``ng_agent_observations``:

- one ``AgentInvocation`` whose ``model_calls`` are the gameplay calls in order (the accepted calls, the references
  a training framework joins to its captured calls to rebuild every context chain);
- one ``ContextCompactionObservation`` per drain: a gameplay request that no longer starts with the previous
  gameplay request (history was dropped), with the summary text the harness put in its place, the call before and
  after it, and the helper calls that wrote the summary.

A gameplay request that rewrites its history without dropping any of it (for example, new level lessons in an
earlier message) also starts a new chain; it is reported as a ``history_rewritten`` gap, not as a compaction.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nemo_gym.config import ModelServerRef
from nemo_gym.trajectory.trajectory_record import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
)


GAMEPLAY_ROLE = "gameplay"
# Helper calls that write the text a drain keeps in place of the dropped history.
COMPACTION_HISTORY_IDS = frozenset({"drain-summary", "turn-compaction", "rolling-summary"})
# The harness's drain summary message (tool_agent.py, turn-compaction patch v5).
DRAIN_SUMMARY_PREFIX = "[Summary of earlier history"
SUMMARY_MAX_CHARS = 20000


def message_digest(message: Any) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _drain_summary(messages: List[Dict[str, Any]]) -> Optional[str]:
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") == "user" and isinstance(content, str) and content.startswith(DRAIN_SUMMARY_PREFIX):
            return content[:SUMMARY_MAX_CHARS]
    return None


@dataclass
class ModelCall:
    index: int
    role: str
    history_id: str
    status: int
    response_id: Optional[str]
    request_digests: List[str]
    prompt_tokens: Optional[int] = None
    drain_summary: Optional[str] = None
    observed_at: float = field(default_factory=time.time)


class GameCallLog:
    """Thread-safe log of one game's model calls, in the order they were sent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: List[ModelCall] = []

    def add(
        self,
        *,
        role: str,
        history_id: str,
        messages: List[Dict[str, Any]],
        status: int,
        response: Optional[Dict[str, Any]],
    ) -> None:
        usage = (response or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        with self._lock:
            self.calls.append(
                ModelCall(
                    index=len(self.calls),
                    role=role,
                    history_id=history_id,
                    status=status,
                    response_id=(response or {}).get("id") if status < 400 else None,
                    request_digests=[message_digest(m) for m in messages],
                    prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else None,
                    drain_summary=_drain_summary(messages) if role == GAMEPLAY_ROLE else None,
                )
            )


def observations(
    log: GameCallLog,
    *,
    invocation_id: str,
    model_ref: ModelServerRef,
    status: str,
    stop_reason: Optional[str],
) -> AgentObservationBundle:
    """The rollout's agent observations (see the module docstring)."""

    def ref(call: ModelCall) -> Optional[ModelCallRef]:
        return ModelCallRef(model_ref=model_ref, response_id=call.response_id) if call.response_id else None

    calls = list(log.calls)
    records: List[Any] = []
    gaps: List[ObservationGap] = []
    gameplay = [c for c in calls if c.role == GAMEPLAY_ROLE and c.response_id]
    for call in calls:
        if call.status >= 400:
            gaps.append(
                ObservationGap(
                    code="model_call_failed",
                    invocation_id=invocation_id,
                    detail=f"call {call.index} ({call.role}/{call.history_id}): HTTP {call.status}",
                )
            )
    for previous, current in zip(gameplay, gameplay[1:]):
        prefix = previous.request_digests
        if current.request_digests[: len(prefix)] == prefix:
            continue  # the same chain: the request extends the previous one
        helpers = [
            c
            for c in calls
            if previous.index < c.index < current.index
            and c.role != GAMEPLAY_ROLE
            and c.history_id in COMPACTION_HISTORY_IDS
            and c.response_id
        ]
        dropped = len(current.request_digests) <= len(prefix) or current.drain_summary is not None or helpers
        if not dropped:
            gaps.append(
                ObservationGap(
                    code="history_rewritten",
                    invocation_id=invocation_id,
                    detail=f"gameplay call {current.index} rewrote earlier history without dropping it",
                )
            )
            continue
        records.append(
            ContextCompactionObservation(
                invocation_id=invocation_id,
                source_model_ref=model_ref,
                observed_at=current.observed_at,
                trigger="automatic",
                tokens_before=previous.prompt_tokens,
                tokens_after=current.prompt_tokens,
                outcome="completed",
                summary=current.drain_summary,
                before_model_call=ref(previous),
                model_calls=[r for r in (ref(c) for c in helpers) if r is not None],
                after_model_call=ref(current),
            )
        )
    records.insert(
        0,
        AgentInvocation(
            invocation_id=invocation_id,
            status=status,
            stop_reason=stop_reason,
            model_calls=[r for r in (ref(c) for c in gameplay) if r is not None],
        ),
    )
    return AgentObservationBundle(source="arc_agi_3_agent", records=records, gaps=gaps)
