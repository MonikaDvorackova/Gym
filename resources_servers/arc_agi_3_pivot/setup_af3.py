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
"""Install the two pinned external trees the pivot environment runs (not vendored: NVIDIA-internal repositories).

- ``agent-factory-3``: the AF3 ARC-AGI-3 window environment (``agent_factory_3.envs.arc3``) that AF3 RL trains on. Only
  that package is imported (numpy, pyarrow, arcengine; no torch), so the checkout is put on ``sys.path`` instead of being
  pip-installed with AF3's training dependencies.
- ``duck-harness``: the harness the pivots were recorded with (``harness-<revision>/kit2/{harness,config}``, the layout
  AF3's ``env_proxy`` resolves from ``harness_base``). Only ``kit2/harness`` and ``kit2/config`` are checked out.

Both are fetched at their pinned commit (``git fetch --depth 1 <url> <sha>``) from the repository URLs you give
(``af3_git_url`` / ``harness_git_url`` in the server config, or ``--af3-url`` / ``--harness-url``). An existing directory
is checked against its pin and never modified. A git submodule or any other checkout at the pin can be used instead
(``af3_src`` / ``harness_base`` in the server config).

    python resources_servers/arc_agi_3_pivot/setup_af3.py --af3-url URL --harness-url URL [--dest DIR]
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE / ".af3"

# agent-factory-3 main, CHANGELOG af3-stable-20261112u.4.43.29.
AF3_COMMIT = "1925d67293f4dfd70258923e28c9e33bed897b86"
# duck-harness jfp/e2-no-tool-call "24k trim", the harness of the Pub25 / V14 exact-host pivot pools.
HARNESS_COMMIT = "30d2cc8bb582c4126e1020d084a94468a057dea7"


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout.strip()


def _checkout(dest: Path, url: str, commit: str, sparse: tuple = ()) -> Path:
    if dest.exists():
        head = _git("rev-parse", "HEAD", cwd=dest)
        if head != commit:
            raise ValueError(f"{dest} is at {head}, not the pinned {commit}")
        return dest
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        subprocess.run(["rm", "-rf", str(partial)], check=True)
    _git("init", "-q", str(partial))
    if sparse:
        _git("sparse-checkout", "set", "--no-cone", *sparse, cwd=partial)
    _git("fetch", "-q", "--depth", "1", *(["--filter=blob:none"] if sparse else []), url, commit, cwd=partial)
    _git("checkout", "-q", "FETCH_HEAD", cwd=partial)
    partial.rename(dest)
    return dest


def ensure_af3(
    dest: Path = DEFAULT_DEST, *, af3_url: Optional[str] = None, harness_url: Optional[str] = None
) -> tuple[Path, Path]:
    """Return ``(<agent-factory-3 src dir>, <harness base>)``, fetching the pinned trees into ``dest`` when absent."""
    af3_dir, harness_dir = dest / "agent-factory-3", dest / f"harness-{HARNESS_COMMIT[:8]}"
    for path, url, name in ((af3_dir, af3_url, "af3_git_url"), (harness_dir, harness_url, "harness_git_url")):
        if not path.exists() and not url:
            raise FileNotFoundError(
                f"{path} is not installed: set {name} (the repository to fetch the pinned commit from), "
                "or af3_src and harness_base (existing checkouts)"
            )
    dest.mkdir(parents=True, exist_ok=True)
    af3 = _checkout(af3_dir, af3_url or "", AF3_COMMIT)
    harness = _checkout(harness_dir, harness_url or "", HARNESS_COMMIT, sparse=("/kit2/harness/", "/kit2/config/"))
    revision = harness / "REVISION"  # AF3's harness loader reads the tree's revision from this file
    if not revision.is_file():
        revision.write_text(HARNESS_COMMIT + "\n")
    return af3 / "src", dest


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--af3-url", help="agent-factory-3 repository URL")
    parser.add_argument("--harness-url", help="duck-harness repository URL")
    args = parser.parse_args(argv)
    src, base = ensure_af3(args.dest, af3_url=args.af3_url, harness_url=args.harness_url)
    print(f"agent-factory-3 {AF3_COMMIT[:8]} at {src}; duck-harness {HARNESS_COMMIT[:8]} under {base}")


if __name__ == "__main__":
    main()
