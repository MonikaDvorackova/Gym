# ARC-AGI-3 pivot windows (PivotRL)

RL from recorded decision points of ARC-AGI-3 games. A row is a **pivot**: one recorded model call of a game played
with the duck harness. A rollout restores the game at that call, continues it with the policy for a short window, and
is rewarded by how much closer to completing the level it got. One window is a plain multi-call chain: the model calls
the `python` tool, the harness runs it (game actions included) and answers, and every call extends the previous one.
There is no compaction and no subagent, so training-token capture sees one chain per rollout.

The environment is AF3's ARC-AGI-3 window environment (`agent_factory_3.envs.arc3`, what AF3 RL trains on) under its
**exact host** (`AF3_EXACT_HOST=1`): every model-visible byte of the host side is the harness's own code. This server
maps it to Gym; the agent is [`responses_api_agents/arc_agi_3_pivot_agent`](../../responses_api_agents/arc_agi_3_pivot_agent/README.md).

## Endpoints

| Endpoint | What |
|---|---|
| `/seed_session` | Restore the row's pivot and return its request: `messages`, `tools`, `chat_template_kwargs`. A pivot that cannot be restored returns `restore_error` (the rollout is masked). |
| `/step` | One model reply (an assistant chat message) -> the harness's tool results and the next opener (or nudge), `done`, and the call's record (actions, level completed, game over, tool error, oracle distance before and after). |
| `/verify` | Reward and components, then the window is released. |
| `/prefetch` | `{"rows": [...]}`: start the restores of these rows now (Gym rows or their `verifier_metadata`). |

## Restore

The restore is exact and verified, as in AF3:

1. The engine prefix runs through the harness's own step path, so the action numbers, history entries, timer mask and
   animation records are the harness's.
2. The Python sandbox state is restored from the recording (the pivot's snapshot capture, replayed cell by cell and
   digest-checked).
3. The board's `frame_sha256`, the request's `request_sha256` and the sandbox's state views must equal the recorded ones.
   A mismatch is a restore error: the rollout is masked (`environment_prepare_failed`), never scored as a failure.

**One restore per pivot, forked per rollout** (`AF3_EXACT_FORK_GROUP=1`): the first window of a pivot on the node starts
a template process that restores the pivot once; every window of that pivot (the G rollouts of a group) connects to it
and gets its own forked copy of the game and the sandbox.

**Ahead of dispatch**, the templates of upcoming rows start restoring in the background:

- `/prefetch`: the caller names the rows (a training loop that knows its next batch).
- read-ahead: with `prefetch_dataset` (the JSONL the rollouts come from) and `prefetch_lookahead: K`, each seed of a row
  starts the next K rows of that file (each row once).

Both stay under AF3's memory guards (`AF3_FORK_MEM_MARGIN_GIB`, `AF3_FORK_MEMGUARD_GIB`, `AF3_FORK_PREFETCH_MAX_GIB`,
`AF3_FORK_PREFETCH_PARALLEL`, set through `af3_env`). A window whose template is not ready waits for it, or restores
itself when there is none; the server log shows each window's `FORK_ATTACH kind=hit|warm|miss|fallback`.

## Window and reward

The window ends at the first of: the level completed, `max_calls` model calls (default 24, the AF3 exact-host recipe's
`AF3_WINDOW_MAX_CALLS`), the window's action budget (`AF3_WINDOW_ACTION_BUDGET`, default 12 under the exact host), or
the harness's context budget. Game over does not end it: the harness resets, as in a full run.

| Field | Meaning |
|---|---|
| `reward` = `progress` | `d0 - d_end`: the oracle distance to completing the level at the pivot minus at the window's end (`d0` when the level completes) |
| `level_completed`, `actions`, `calls`, `tool_errors`, `gen_tokens` | window components (`reward_components` holds progress, level_completed, actions) |
| `d0`, `d_end` | the oracle distances |

A training recipe that weights the components itself (AF3's GDPO z-scores progress, actions and tokens within the
group) reads them from the rollout. Masked: a restore error (`environment_prepare_failed`), a window that reached the
context budget (`model_context_limit`, discarded as in AF3), and a window cut short by a failed model call
(`model_request_error`).

## Setup

**External code** (NVIDIA-internal, not vendored; fetched at pinned commits by `setup_af3.py` on the first start, into
`.af3/` here):

| Tree | Pin | Used |
|---|---|---|
| `efficient-reasoning/agent-factory-3` | `1925d672` (af3-stable-20261112u.4.43.29) | `src/agent_factory_3/envs/arc3` only, put on `sys.path` (numpy, pyarrow, arcengine; not AF3's training stack) |
| `isorokin/duck-harness` | `30d2cc8b` | `kit2/harness`, `kit2/config` (sparse checkout), as `harness-30d2cc8b/` |

To use your own checkouts (a git submodule, a shared copy), set `af3_src` (the agent-factory-3 `src` dir) and
`harness_base` (the directory holding `harness-<revision>/`).

**Pool**: an AF3 pivot pool directory (`pool_dir`) with `games.parquet`, `env/` (the harness env files the rows were
recorded under), `snap/` and `snap-raw/` (the snapshot captures), e.g. `<af3 root>/data/exact30d2-20261111c/pub25`
(the 24 public games, harness 30d2cc8b, snapshot restore). Rows refer to these files relative to the pool directory.

**Dataset**: one Gym row per pivot from a pool split; only rows that restore under the exact host (`restore_mode`
snapshot, `restore_ok`) are kept:

```bash
python resources_servers/arc_agi_3_pivot/create_dataset.py <pool_dir>/eval.parquet \
    --output resources_servers/arc_agi_3_pivot/data/eval.jsonl --per-game 4
```

The Python sandbox runs under `/usr/bin/python3` (3.12, AF3's exact-host default `AF3_SANDBOX_PYTHON`), so the node
needs it. Linux only.

## Run

```bash
R=arc_agi_3_pivot_resources_server.resources_servers.arc_agi_3_pivot
COMMON=(--config resources_servers/arc_agi_3_pivot/configs/arc_agi_3_pivot.yaml
        --config responses_api_models/vllm_model/configs/vllm_model.yaml
        ++$R.pool_dir=<pool_dir>
        ++$R.prefetch_dataset=resources_servers/arc_agi_3_pivot/data/eval.jsonl ++$R.prefetch_lookahead=8)
gym env start "${COMMON[@]}" &
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_pivot_agent \
    --input resources_servers/arc_agi_3_pivot/data/eval.jsonl --num-repeats 4 --concurrency 32 \
    --output results/arc_agi_3_pivot_rollouts.jsonl
```

For training, enable training-token capture as for any multi-call agent (the agent's model calls already use the
rollout's capture route).

## Tests

```bash
pytest resources_servers/arc_agi_3_pivot/tests
```

The tests run the server against a stand-in for AF3's `env_proxy`; a live check needs the pool, the pinned trees and a
model.
