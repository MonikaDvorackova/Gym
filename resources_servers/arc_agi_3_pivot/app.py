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
"""ARC-AGI-3 pivot windows: RL from recorded game states (PivotRL), on AF3's window environment.

A pivot is a recorded model call of an ARC-AGI-3 game played with the duck harness. A window restores the pivot exactly
(the game, the harness's host state and the live Python sandbox, verified against the recording) and lets the policy
continue from there for a few calls. The environment is AF3's ``Arc3WindowEnv`` (agent-factory-3,
``agent_factory_3.envs.arc3``), run through its ``env_proxy`` worker pool; this server only maps it to Gym:

- ``/seed_session``: restore the row's pivot; returns the pivot request (messages, tools, chat template kwargs).
- ``/step``: one model reply (an assistant chat message) -> the harness's tool results and the next opener, ``done``.
- ``/verify``: reward = oracle-distance progress over the window (``d0 - d_end``; ``d0`` when the level completes).
- ``/prefetch``: start the restores of upcoming rows in the background.

Restores (AF3 exact host, ``AF3_EXACT_FORK_GROUP``): the first window of a pivot on a node starts a template process that
restores the pivot once; every window of that pivot gets a forked copy (game and sandbox), so a group of G rollouts
costs one restore. Restores start ahead of dispatch through ``/prefetch`` (rows named by the caller) and through
read-ahead (``prefetch_dataset`` + ``prefetch_lookahead``: the next K rows of the dataset after each seeded row). Both
are optimizations under AF3's memory guards; a window whose template is not ready restores itself.
"""

import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym import PARENT_DIR
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core.failure_kinds import ENVIRONMENT_PREPARE_FAILED, MODEL_CONTEXT_LIMIT, MODEL_REQUEST_ERROR
from nemo_gym.server_runtime import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.arc_agi_3.setup_arc_agi import ensure_flask
from resources_servers.arc_agi_3_pivot.setup_af3 import DEFAULT_DEST, ensure_af3


LOG = logging.getLogger(__name__)
# Pool row columns the env reads (pivot_json; the row's harness revision and recorded harness env file).
ROW_KEYS = ("pivot_json", "harness", "harness_env", "harness_env_sha256")
# AF3 exact host with one restore per pivot, forked per window (AF3 docs/arc3-port/FLAGS.md).
DEFAULT_AF3_ENV = {"AF3_EXACT_HOST": "1", "AF3_EXACT_FORK_GROUP": "1"}


def repo_path(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else PARENT_DIR / path


def pool_row(verifier_metadata: Dict[str, Any]) -> Dict[str, Any]:
    row = verifier_metadata.get("pivot") or {}
    missing = [k for k in ("pivot_json", "harness") if not row.get(k)]
    if missing:
        raise ValueError(f"verifier_metadata.pivot lacks {missing}")
    return {k: row[k] for k in ROW_KEYS if row.get(k) is not None}


def assistant_reply_text(message: Dict[str, Any]) -> str:
    """The model's reply as the generation text AF3's env parses (qwen3 reasoning + qwen3_coder tool-call markup).

    A tool call whose arguments are not a JSON object is left out: the env then answers the reply as one without a
    tool call (the harness's nudge), as the harness does with a call it cannot parse.
    """
    from agent_factory_3.envs.arc3.env import reply_text

    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            calls.append({"type": "function", "function": {"name": fn.get("name"), "arguments": parsed}})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return reply_text(
        {"reasoning_content": message.get("reasoning_content") or "", "content": content or "", "tool_calls": calls}
    )


class ArcAgi3PivotConfig(BaseResourcesServerConfig):
    # The AF3 pool directory: games.parquet, env/ (recorded harness env files), snap/, snap-raw/. Rows name their
    # snapshot and env files relative to it.
    pool_dir: str
    # agent-factory-3 src dir and the directory holding harness-<revision>/ (default: the pinned trees setup_af3.py
    # installs under .af3/ next to this file).
    af3_src: Optional[str] = None
    harness_base: Optional[str] = None
    # Where setup_af3.py fetches the pinned trees from on the first start (when af3_src / harness_base are unset).
    af3_git_url: Optional[str] = None
    harness_git_url: Optional[str] = None
    # Window: at most this many model calls (the exact-host recipe's AF3_WINDOW_MAX_CALLS). Actions per window:
    # AF3_WINDOW_ACTION_BUDGET in af3_env (AF3's default under the exact host is 12).
    max_calls: int = 24
    # AF3 environment flags, applied before AF3 is imported (memory guards, fork template TTL, action budget, ...).
    af3_env: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_AF3_ENV))
    # Read-ahead: on each seed, prefetch the next prefetch_lookahead rows of this JSONL (by pivot_id order).
    prefetch_dataset: Optional[str] = None
    prefetch_lookahead: int = 0


class PivotSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Dict[str, Any]


class PivotSeedResponse(BaseSeedSessionResponse):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # A pivot that could not be restored (infrastructure): no window, the rollout is masked.
    restore_error: Optional[str] = None


class PivotStepRequest(BaseModel):
    # The model's reply: an assistant chat message (content, reasoning_content, tool_calls).
    message: Dict[str, Any]
    finish_reason: str = "stop"
    gen_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None


class PivotStepResponse(BaseModel):
    # Messages to append after the reply: the harness's tool results, then the next opener or nudge (none when done).
    messages: List[Dict[str, Any]]
    done: bool
    call: Dict[str, Any]


class PrefetchRequest(BaseModel):
    # Rows (Gym rows or their verifier_metadata) whose restores should start now.
    rows: List[Dict[str, Any]]


class PrefetchResponse(BaseModel):
    status: List[str]


class PivotVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Dict[str, Any] = Field(default_factory=dict)
    # Set by the agent when the window could not be played to its end (a failed model call).
    agent_error: Optional[str] = None


class PivotVerifyResponse(BaseMultiRewardVerifyResponse):
    model_config = ConfigDict(extra="allow")
    progress: float = 0.0
    level_completed: float = 0.0
    actions: float = 0.0
    calls: float = 0.0
    tool_errors: float = 0.0
    gen_tokens: float = 0.0
    d0: Optional[int] = None
    d_end: Optional[int] = None


@dataclasses.dataclass
class _Window:
    env: Any = None
    prompt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    d0: Optional[int] = None
    restore_error: Optional[str] = None
    done: bool = False
    calls: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    gen_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class ArcAgi3PivotResourcesServer(SimpleResourcesServer):
    config: ArcAgi3PivotConfig
    _windows: Dict[str, _Window] = PrivateAttr(default_factory=dict)
    _read_ahead: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _read_ahead_index: Dict[str, int] = PrivateAttr(default_factory=dict)
    _prefetched: set = PrivateAttr(default_factory=set)
    _prefetch_tasks: set = PrivateAttr(default_factory=set)

    def model_post_init(self, context: Any) -> None:
        ensure_flask()  # arc_agi (AF3's game catalog) imports flask
        if self.config.num_workers not in (None, 1):
            raise ValueError("windows live in this process (and its env workers); num_workers must be 1")
        if self.config.af3_src and self.config.harness_base:
            src, base = repo_path(self.config.af3_src), repo_path(self.config.harness_base)
        else:
            src, base = ensure_af3(
                DEFAULT_DEST, af3_url=self.config.af3_git_url, harness_url=self.config.harness_git_url
            )
        self._harness_base = str(base)
        self._games = str(repo_path(self.config.pool_dir) / "games.parquet")
        if not Path(self._games).is_file():
            raise FileNotFoundError(f"{self._games} missing (pool_dir must hold the AF3 pool's games.parquet)")
        os.environ.update({k: str(v) for k, v in self.config.af3_env.items()})
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from agent_factory_3.envs.arc3 import env_proxy

        # AF3's restore / fork lines (FORK_ATTACH hit|warm|miss|fallback, FORK_PREFETCH) in this server's log
        fork_log = logging.getLogger("agent_factory_3.envs.arc3.exact_host.fork_service")
        if not fork_log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
            fork_log.addHandler(handler)
            fork_log.setLevel(logging.INFO)

        self._proxy = env_proxy
        if self.config.prefetch_dataset and self.config.prefetch_lookahead > 0:
            with open(repo_path(self.config.prefetch_dataset)) as fh:
                for line in fh:
                    meta = json.loads(line).get("verifier_metadata") or {}
                    if meta.get("pivot_id") and meta["pivot_id"] not in self._read_ahead_index:
                        self._read_ahead_index[meta["pivot_id"]] = len(self._read_ahead)
                        self._read_ahead.append(meta)
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/prefetch")(self.prefetch)
        return app

    # ------------------------------------------------------------------ restores
    def _env_kwargs(self) -> Dict[str, Any]:
        return {"harness_base": self._harness_base, "max_calls": self.config.max_calls}

    def _prefetch_one(self, meta: Dict[str, Any]) -> str:
        try:
            row = pool_row(meta)
        except ValueError as exc:
            return f"invalid: {exc}"
        return self._proxy.prefetch(row, self._games, **self._env_kwargs())

    async def _prefetch(self, metas: List[Dict[str, Any]]) -> List[str]:
        return list(await asyncio.gather(*(asyncio.to_thread(self._prefetch_one, m) for m in metas)))

    def _read_ahead_after(self, pivot_id: Optional[str]) -> List[Dict[str, Any]]:
        i = self._read_ahead_index.get(pivot_id) if pivot_id else None
        if i is None:
            return []
        out = []
        for meta in self._read_ahead[i + 1 : i + 1 + self.config.prefetch_lookahead]:
            if meta["pivot_id"] not in self._prefetched:
                self._prefetched.add(meta["pivot_id"])
                out.append(meta)
        return out

    async def prefetch(self, body: PrefetchRequest) -> PrefetchResponse:
        metas = [r.get("verifier_metadata", r) for r in body.rows]
        return PrefetchResponse(status=await self._prefetch(metas))

    def _restore(self, row: Dict[str, Any]) -> tuple:
        env = self._proxy.Arc3WindowEnvProxy(row, self._games, **self._env_kwargs())
        try:
            prompt = env.reset()
            return env, prompt, env.d0
        except BaseException:
            env.close()
            raise

    # Session state hooks (the base seeds through them and releases on close / retirement; verify releases too).
    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._windows

    async def initialize_session_state(self, session_id: str, body: PivotSeedRequest) -> None:
        window = self._windows[session_id] = _Window()
        meta = body.verifier_metadata
        ahead = self._read_ahead_after(meta.get("pivot_id"))
        if ahead:  # fire and forget: the next rows' templates restore while this window plays
            task = asyncio.ensure_future(self._prefetch(ahead))
            self._prefetch_tasks.add(task)
            task.add_done_callback(self._prefetch_tasks.discard)
        try:
            window.env, window.prompt, window.d0 = await asyncio.to_thread(self._restore, pool_row(meta))
        except Exception as exc:  # RestoreError, EnvWorkerError, a bad row: infrastructure, never a policy outcome
            window.restore_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("pivot %s not restored: %s", meta.get("pivot_id"), window.restore_error)

    async def release_session_state(self, session_id: str) -> None:
        window = self._windows.get(session_id)
        if window is not None and window.env is not None:
            await asyncio.to_thread(window.env.close)
        self._windows.pop(session_id, None)

    async def seed_session(self, request: Request, body: PivotSeedRequest) -> PivotSeedResponse:
        await super().seed_session(body, request=request)
        window = self._windows[request.session[SESSION_ID_KEY]]
        if window.restore_error is not None:
            return PivotSeedResponse(restore_error=window.restore_error)
        prompt = window.prompt
        return PivotSeedResponse(
            messages=prompt["messages"],
            tools=prompt.get("tools") or [],
            chat_template_kwargs=prompt.get("chat_template_kwargs") or {},
        )

    # ------------------------------------------------------------------ play
    async def step(self, request: Request, body: PivotStepRequest) -> PivotStepResponse:
        window = self._windows.get(request.session[SESSION_ID_KEY])
        if window is None or window.env is None:
            raise ValueError("step on a session without a restored window")
        async with window.lock:
            if window.done:
                raise ValueError("step after the window ended")
            text = assistant_reply_text(body.message)
            new, done, info = await asyncio.to_thread(
                window.env.step,
                text,
                gen_tokens=body.gen_tokens,
                finish_reason=body.finish_reason,
                prompt_tokens=body.prompt_tokens,
            )
            window.done = bool(done)
            window.gen_tokens += int(body.gen_tokens or 0)
            call = dataclasses.asdict(info) if dataclasses.is_dataclass(info) else dict(vars(info))
            window.calls.append(call)
        # The env names the reply's tool calls itself; give its tool results the ids of the model's own calls (same order).
        ids = [c.get("id") for c in body.message.get("tool_calls") or []]
        env_ids: List[str] = []
        for m in new:
            if m.get("role") == "tool" and m.get("tool_call_id") not in env_ids:
                env_ids.append(m.get("tool_call_id"))
        remap = {e: i for e, i in zip(env_ids, ids) if i}
        messages = [
            dict(m, tool_call_id=remap.get(m["tool_call_id"], m["tool_call_id"])) if m.get("role") == "tool" else m
            for m in new
        ]
        return PivotStepResponse(messages=messages, done=window.done, call=json.loads(json.dumps(call, default=str)))

    # ------------------------------------------------------------------ score
    async def verify(self, request: Request, body: PivotVerifyRequest) -> PivotVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        try:
            return await self._score(
                self._windows.get(session_id) or _Window(restore_error="no window was seeded"), body
            )
        finally:  # the window's env worker (a forked copy of its pivot's template) exits now, not at session close
            await self.release_session_state(session_id)

    async def _score(self, window: _Window, body: PivotVerifyRequest) -> PivotVerifyResponse:
        payload = body.model_dump()
        masked = None
        if window.restore_error is not None:
            masked = (ENVIRONMENT_PREPARE_FAILED, window.restore_error)
        elif any(c.get("discard") for c in window.calls):
            masked = (MODEL_CONTEXT_LIMIT, "the window reached the harness's context budget (discarded, not scored)")
        elif body.agent_error or not window.done:
            masked = (MODEL_REQUEST_ERROR, body.agent_error or "the window ended before its termination rule")
        if masked is not None:
            return PivotVerifyResponse(
                **payload,
                reward=0.0,
                reward_components={"progress": 0.0},
                mask_sample=True,
                failure_kind=masked[0],
                failure_reason=masked[1],
                d0=window.d0,
                calls=float(len(window.calls)),
            )
        level_completed = any(c.get("level_completed") for c in window.calls)
        d_end = 0 if level_completed else await asyncio.to_thread(getattr, window.env, "d")
        progress = float((window.d0 or 0) - (d_end or 0))
        actions = float(sum(len(c.get("actions") or []) for c in window.calls))
        tool_errors = float(sum(1 for c in window.calls if c.get("tool_error")))
        return PivotVerifyResponse(
            **payload,
            reward=progress,
            reward_components={"progress": progress, "level_completed": float(level_completed), "actions": actions},
            progress=progress,
            level_completed=float(level_completed),
            actions=actions,
            calls=float(len(window.calls)),
            tool_errors=tool_errors,
            gen_tokens=float(window.gen_tokens),
            d0=window.d0,
            d_end=d_end,
        )


if __name__ == "__main__":
    ArcAgi3PivotResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ArcAgi3PivotResourcesServer.run_webserver()  # noqa: F401
