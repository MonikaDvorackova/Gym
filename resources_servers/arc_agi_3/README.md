# ARC-AGI-3 (public25)

[ARC-AGI-3](https://arcprize.org/arc-agi/3/) is a set of interactive, turn-based games. The agent sees a 64×64 board, sends actions (keys, clicks, RESET) and must discover the rules and the goal of each level by itself. A game has several levels. The score of a game is RHAE (relative human action efficiency):

- a cleared level `i` (1-based) scores `min(115, (human_actions_i / agent_actions_i)^2 * 100)`;
- the game score is the mean of the level scores weighted by `i`, capped at the weighted fraction of the levels that scored, times 100;
- RHAE of a run is the mean game score (0-100) over the expected games. `rhae_<h>h` counts only the levels cleared in the first `h` hours of the run (one run clock: the first game's start, as the kit's benchmark start).

This environment plays the **25 public games** (public25) with the **NVARC3 Kaggle harness** (submission 57908411, public leaderboard 60.95). The agent is [`responses_api_agents/arc_agi_3_agent`](../../responses_api_agents/arc_agi_3_agent/README.md).

This server is the authority and the verifier for both, following the authority model of Ivan Sorokin's arc3-gym:
- **It runs the official game server.** This is `arc_agi`'s own server on the pinned game files (`game_server.py`), and the harness plays every game through it, one scorecard per rollout.
- **The reward is the card's score.** At `/verify` it closes the rollout's scorecard (if the harness has not), and sets the reward to the card's score / 100.
- **It fails loudly on a mismatch.** The agent's own report (game run guid, levels, won or not, action count) must equal the card, or verify raises.
- **Timed scores use the server's clock.** RHAE@1h / @2h come from the server's own per-action timestamps.
- **The pins are checked.** It compares the row's pins (full game id, version, sha256 of the game files) with the games it loaded, and checks that the agent's report is for that exact game.

## Data

The games are not committed. `create_dataset.py` builds the rows from the competition's `environment_files` (`<game>/<version>/{<game>.py, metadata.json}`, arc_agi 0.9.8, 3.9 MB):

```bash
# Option A: download the files from the Kaggle competition (needs a Kaggle token and the competition rules accepted)
python resources_servers/arc_agi_3/create_dataset.py \
    --download-kaggle-competition resources_servers/arc_agi_3/data \
    --output resources_servers/arc_agi_3/data/public25.jsonl
# Option B: use a copy you already have
python resources_servers/arc_agi_3/create_dataset.py \
    --environments-dir /path/to/environment_files --output resources_servers/arc_agi_3/data/public25.jsonl
# Check a copy against a dataset (sha256 of every game file)
python resources_servers/arc_agi_3/create_dataset.py \
    --environments-dir resources_servers/arc_agi_3/data/environment_files \
    --check resources_servers/arc_agi_3/data/public25.jsonl
```

The server and the agent read the games from `data/environment_files` (config `environments_dir`). The 25 game versions are pinned in `create_dataset.py`.

- `data/example.jsonl` (committed): 5 games (ft09, ls20, vc33, lp85, sp80).
- `data/public25.jsonl` (generated): all 25 games, one row per game.

Each row's `verifier_metadata` has the game id, version, baselines, number of levels, the sha256 of the game files and the harness dispatch index (rows are in sorted game-id order).

## The game server (the authority)

At start, this server launches `python -m resources_servers.arc_agi_3.game_server`, a child process, and stops it at shutdown. That process runs `arc_agi`'s own Flask server (`Arcade(OFFLINE).listen_and_serve`, arc_agi 0.9.8 / arcengine 0.9.3) on `environments_dir`. RESET keeps the level, as in the kit (`ONLY_RESET_LEVELS=true`).
- **Where agents find it:** `GET /game_server` returns its URL. The 60.95 agent plays through it with taaf's own ONLINE `ArcadeSpec`; the official agent plays upstream's own ONLINE `Arcade`.
- **What it writes** to `<game_server_dir>/<start time>/`:
  - `cards/<card_id>.json`: every scorecard as it was closed. A closed card is gone from the server, and the kit's taaf closes its card at the end of its game.
  - `recordings/<card_id>/<game>-<guid>.jsonl`: one line per response, with the server's UTC `timestamp`, the action, `levels_completed` and `state` (no frames).
  - `game_server.log`.
- **Human baselines:** the server's game-info routes (`/api/games`, `/api/games/<id>`) also give each game's `baseline_actions`. arc_agi's own server leaves them out (the competition setting), but both harnesses need them:
  - the kit's taaf records them, and the kit's `rhae.py` scores `benchmark.json` with them;
  - the official harness sets its per-level action budgets from them; without them it falls back to 10 actions per game.

  Scoring and game logic stay arc_agi's.
- **Idle cards:** a card idle for `scorecard_timeout_min` (default 240) is closed by the server. The server's own default, 15 min, can be shorter than one model turn.
- **Cost per action** (measured on one client): 2-7 ms through the server against 0.3 ms in process, and 4-10 ms with the recordings. Most of it is serializing the frame JSON.

## Verify

Request: the standard verify request plus:
- `game_record`: the agent's own report of the attempt, or null when the game could not start. The 60.95 agent sends the taaf `GameRun.to_json_dict()`; the official agent sends the same fields with an `actions` count instead of a `history`.
- `scorecard_id`, `game_guid`, `scorecard_api_key`: the rollout's card, its game run and the key that opened the card. The key lets the verifier close a card that is still open.
- `benchmark_start_time`: the run clock's origin as a UTC time, taken before the first game started; the agents always send it. Without it there is no timed score: `rhae_<h>h` are null and `clock_error` says why.
- `agent_error` / `persistence_error`: the agent's own problems.

The scorecard is the source of truth. Verify raises (HTTP error, so the rollout fails visibly) when any of these holds:
- the card does not hold exactly this game;
- a run's guid differs from the agent's;
- the last run's levels or won state differ from the agent's;
- the card's action count differs from the agent's;
- the server's recording disagrees with its own card;
- the request has no card.

Measured: the card counts exactly the actions the harness sent, and the RESET of `make()` is not counted.

A row without pins, or with pins that differ from the loaded game, is rejected (HTTP error, so the rollout fails visibly).

Response fields:

| Field | Meaning |
|---|---|
| `reward` | the scorecard's score / 100 |
| `rhae` | the scorecard's game score (0-100; the SDK's best run) |
| `rhae_1h`, `rhae_2h` | game score at 1 h and 2 h of the run clock (`score_hours`), from the server's action times; null when the timing is invalid or the request has no `benchmark_start_time` |
| `levels_completed`, `number_of_levels`, `actions` | the attempt, from the scorecard |
| `state` | terminal state from the scorecard: `won` / `not_won` (`not_started` without a game) |
| `agent_state` | the agent's own end state: `gave_up` (window over or budget spent), `won`, `crashed`, `cancelled`, `not_started` |
| `scorecard_id`, `scorecard_runs` | the rollout's card and its runs of the game (a full reset starts another run) |
| `timing_valid`, `timing_error`, `clock_offset_seconds`, `clock_error` | the timing check on the server's recording; the game's first server event on the run clock (null without the origin, `clock_error` says so) |
| `result_valid` | the attempt counts as a complete, verified result: not cancelled, timing valid, no `agent_error` / `persistence_error` |

`aggregate_metrics` reports the counts (`received_games`, `valid_results`, `not_started`, `crashed`, `cancelled`, `timing_invalid`, `persistence_errors`). Rollouts that failed in Gym never reach it, so the received games can be fewer than the evaluation's:
- with `expected_games` set in the server config (e.g. `++arc_agi_3_resources_server.resources_servers.arc_agi_3.expected_games=25` for public25), `rhae` / `rhae_<h>h` are reported only when all expected games are there with valid results (`complete: true`);
- without it, they are reported when every received row is valid, and they are a statistic of the received rows only (`score_over: received_rows`, `complete: null`).

The benchmark table comes from `summarize_rhae.py --expected`.

## Run

See the [agent README](../../responses_api_agents/arc_agi_3_agent/README.md) for the full launch (model server, agent, GB200/GB300). Summary of one or more runs over the expected game set:

```bash
python resources_servers/arc_agi_3/summarize_rhae.py --expected resources_servers/arc_agi_3/data/public25.jsonl \
    results/arc_agi_3/<run_name> [<run dir> | <benchmark.json> | <rollouts.jsonl> ...] --hours 1 2 [--json out.json]
```

What it reads and reports:
- **Run directory:** scores come from its `benchmark.json` (the run clock, as the kit's `rhae.py`; readable during play), with the verifier's flags from `rollouts.jsonl` and Gym's `rollouts_failures.jsonl`.
- **Completeness:** a column gets a score only when every expected game has a valid value; otherwise it reads `INCOMPLETE n/N`. Nothing is zero-filled.
- **Final or not:** a value is final when the game's attempt has ended, or (for `@T`) when the snapshot was saved at or after run start + T. A run directory with a `rollouts.jsonl` also needs the verifier's row for each game (`not_verified` otherwise). A column with all values present but not all final reads `<value> not final`: the score so far of a live or aborted run. `complete` is true only when every column is final.
- **Rollouts JSONL:** a row needs a real `clock_offset_seconds` (finite, non-negative) for its timed scores; otherwise it is flagged `no run clock`.
- **Counts:** missing, not started, cancelled, crashed, invalid timing, agent and persistence errors, and failed rollouts (a game whose scorecard disagrees with its agent fails its rollout).

## Results

Measured on 2026-11-04 and 2026-11-05 with this environment at b7d91de4 and the 60.95 harness (duck-harness 30d2cc8b, see the agent README).
- **Setup:** public25, all 25 games at once, 7200 s of play, 2 seeds per config (engine seed = run seed). One GB300 node (4 GPUs) per run, interactive Slurm; vLLM v0.32.0, TP4, MTP 3.
- **Score:** RHAE (0-100) from the kit's `rhae.py` on each run's `benchmark.json` (one run clock). Every cell has 25/25 games and 120 min of play. Δ is against the B0 mean.
- **B0** = the 60.95 harness settings: temperature 0.7, top_p 0.95, top_k 20, 24576 tokens of kept history, thinking on, 12288 output tokens.

### Nemotron 3.5 Super VL (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B, BF16 unless noted)

| config | change vs B0 | s1 @1h | s2 @1h | mean @1h | Δ @1h | s1 @2h | s2 @2h | mean @2h | Δ @2h | levels in 2 h (s1 / s2) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | — | 2.01 | 1.80 | 1.90 | | 2.03 | 1.98 | 2.00 | | 22 / 20 |
| S1 | temperature 1.0, top_p 1.0, no top_k | 1.77 | 2.07 | 1.92 | +0.02 | 1.80 | 2.24 | 2.02 | +0.02 | 32 / 30 |
| S2 | temperature 0.6, top_p 0.95, no top_k | 1.60 | 2.11 | 1.86 | −0.05 | 1.64 | 2.11 | 1.87 | −0.13 | 22 / 23 |
| S3 | temperature 1.0 | 2.96 | 1.89 | 2.42 | +0.52 | 3.12 | 1.89 | 2.50 | +0.50 | 34 / 21 |
| C1 | keep 12288 | 3.03 | 2.17 | 2.60 | +0.69 | 3.94 | 3.60 | **3.77** | **+1.77** | 37 / 34 |
| C2 | keep 49152 | 1.51 | 1.36 | 1.44 | −0.47 | 1.69 | 1.37 | 1.53 | −0.47 | 19 / 20 |
| C3 | keep 98304 | 1.45 | 1.72 | 1.58 | −0.32 | 1.45 | 1.75 | 1.60 | −0.40 | 20 / 22 |
| R1 | thinking off | 0.79 | 0.24 | 0.51 | −1.39 | 0.79 | 0.24 | 0.51 | −1.49 | 18 / 20 |
| R2 | thinking token budget 8192 | 2.29 | 3.06 | 2.68 | +0.78 | 2.32 | 3.07 | 2.69 | +0.69 | 25 / 23 |
| P1 | NVFP4 weights | 1.90 | 2.17 | 2.04 | +0.14 | 1.90 | 3.28 | 2.59 | +0.59 | 19 / 27 |
| TC0 | no turn compaction (`harness_env.ARC3_TURN_COMPACTION=0`) | 1.74 | 3.20 | 2.47 | +0.57 | 1.88 | 3.22 | 2.55 | +0.55 | 21 / 23 |
| TH | drop past reasoning (`truncate_history_thinking`) | 1.19 | 1.07 | 1.13 | −0.77 | 1.22 | 1.07 | 1.14 | −0.86 | 19 / 23 |

- C1 is the best config: both seeds are above both B0 seeds, and it still gains from 1 h to 2 h. R2 is the only other config with both seeds above both B0 seeds.
- Thinking off (R1) and dropping past reasoning (TH) are below B0. Sampling (S1-S3), NVFP4 (P1) and TC0 are within seed noise (the seed spread inside a config is up to 1.4).
- The absolute level is low: most games clear 0-1 levels. The harness is tuned for ARC-trained Qwen checkpoints; Nemotron's transcripts show many turns without an action and looping reasoning, with no integration errors.

### ARC-trained Qwen-type checkpoint (hr100, the model of the 60.95 submission)

Same environment and B0 settings, with only the model swapped: the NVFP4 export with its own MTP drafter and top-10 routing, served with the `qwen3` / `qwen3_coder` parsers and the model's chat template (vLLM v0.32.0, TP4 + expert parallel, `--max-model-len 262144`). Gym model config: `nemotron_3_5_super_vl.yaml` at b7d91de4 with the model name swapped; its model settings are the same as in `qwen_vl_arc3_kit.yaml`, which was added after these runs.

| model | s1 @1h | s2 @1h | mean @1h | s1 @2h | s2 @2h | mean @2h | levels in 2 h (s1 / s2) |
|---|---|---|---|---|---|---|---|
| hr100 | 62.33 | 64.12 | 63.22 | 69.66 | 74.62 | **72.14** | 158 / 165 |
| Nemotron B0 | 2.01 | 1.80 | 1.90 | 2.03 | 1.98 | 2.00 | 22 / 20 |

- All 25 games score above 0 in both seeds.
- Workload: here one node plays 25 games; the kit's runs play 110 per node. Each game gets about 4x the engine time, so these hours are not comparable with the kit's 110-game hours.
- Training data: the training-stage manifests show none of the 25 public games in hr100's training data (three distillation stages on synthetic games: 0 public game ids in every row; the curation code excludes the public games). The first SFT stage has no readable row list; its data policy excludes public games.

### Drain summaries

When the harness drops old history (a drain), it asks the model for a summary of the dropped turns. Usable share (written summaries in the request logs against `drain summary unusable` warnings; 17 of the 24 Nemotron runs counted):
- Nemotron with thinking on (B0, S1-S3, C1-C3, R2, P1): 61-82 %;
- Nemotron R1 (thinking off): 99 %; TH: 100 %; TC0: no summary requests;
- hr100: 100 % (241 / 235 summaries per seed, written in all 25 games of each seed, 0 unusable).

### Validity

- All 26 runs: 25/25 games, 120 min of play, one run clock, 0 replay mismatches. These runs used the earlier replay verifier, which replayed every recorded action in a fresh engine; the scorecard authority replaced it.
- The code at b7d91de4 had a race in the snapshot save, fixed since. In 11 of 24 Nemotron runs and in 1 of 2 hr100 runs, one game's save raced. For all 650 games, the `benchmark.json` entry agrees with the game's event log in levels and actions, so the kit scores above are intact. Gym's own rollout rows read 0 levels for some raced games, which made Gym's own aggregate low by up to 0.22 (Nemotron) and by 1.07 (hr100 s1). The current code saves snapshots under a lock, and `summarize_rhae.py` reports such a run as `INCOMPLETE` instead of scoring it.

## Tests

```bash
ARC3_ENVIRONMENTS_DIR=/path/to/environment_files pytest resources_servers/arc_agi_3/tests
```

The tests on the real game server skip when the game files are absent.

## Licensing

- Code: Apache 2.0.
- Game files: ARC Prize Foundation, from the ARC Prize 2026 ARC-AGI-3 Kaggle competition (not redistributed here).
