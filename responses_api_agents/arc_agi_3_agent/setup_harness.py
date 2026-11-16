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
"""Install the pinned ARC-AGI-3 harness (not vendored: it is third-party code) and check every byte of it.

An install is a directory with ``harness/`` (the Kaggle ``...-harness`` dataset tree) and
``config/harness_env.toml``. ``harness_manifest.json`` pins the sha256 of every file. Sources:

- a directory with that layout, or a ``.tar.gz`` of one (a single top-level directory is allowed);
- ``kaggle:<owner>/<prefix>``: downloads the ``<prefix>-harness`` and ``<prefix>-config`` Kaggle datasets (needs a
  token in ``KAGGLE_API_TOKEN`` or ``~/.kaggle/access_token`` and access to the datasets).

    python responses_api_agents/arc_agi_3_agent/setup_harness.py <source> [--dest DIR]
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "harness_manifest.json"
DEFAULT_DEST = HERE / ".arc3_harness"
KAGGLE_API = "https://www.kaggle.com/api/v1"
# Files Python creates when the harness is imported; not part of the pinned tree.
IGNORED_PARTS = ("__pycache__",)
IGNORED_SUFFIXES = (".pyc",)


def load_manifest(path: Optional[Path] = None) -> Dict:
    return json.loads((path or MANIFEST_PATH).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_files(root: Path) -> List[str]:
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_dir() or any(p in IGNORED_PARTS for p in rel.parts) or path.suffix in IGNORED_SUFFIXES:
            continue
        files.append(rel.as_posix())
    return files


def verify_install(root: Path, manifest: Optional[Dict] = None) -> None:
    """Raise unless ``root`` holds exactly the pinned harness files and harness_env.toml."""
    manifest = manifest or load_manifest()
    problems = []
    for subdir, pinned in (("harness", manifest["harness_files"]), ("config", manifest["config_files"])):
        base = root / subdir
        if not base.is_dir():
            raise FileNotFoundError(f"ARC-AGI-3 harness install {root} has no {subdir}/ directory")
        present = set(_tree_files(base)) - {"MANIFEST.json", "HARNESS-PROVENANCE.json"}
        if subdir == "config":
            present &= set(pinned)  # the config dataset's other files (profile.args, ...) are engine-side
        for rel, digest in pinned.items():
            path = base / rel
            if not path.is_file():
                problems.append(f"missing {subdir}/{rel}")
            elif _sha256(path) != digest:
                problems.append(f"sha256 differs: {subdir}/{rel}")
        problems += [f"unpinned file {subdir}/{rel}" for rel in sorted(present - set(pinned))]
    if problems:
        shown = "; ".join(problems[:10])
        raise ValueError(f"ARC-AGI-3 harness install {root} does not match harness_manifest.json: {shown}")


def _single_root(directory: Path) -> Path:
    if (directory / "harness").is_dir():
        return directory
    children = [p for p in directory.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "harness").is_dir():
        return children[0]
    raise FileNotFoundError(f"{directory} has no harness/ directory (expected harness/ and config/)")


def _kaggle_token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token
    path = Path.home() / ".kaggle" / "access_token"
    if not path.is_file():
        raise FileNotFoundError("kaggle: source needs KAGGLE_API_TOKEN or ~/.kaggle/access_token")
    return path.read_text().strip()


def _download_kaggle(ref: str, workdir: Path) -> Path:
    owner, prefix = ref.split("/", 1)
    token = _kaggle_token()
    root = workdir / "kaggle"
    for kind in ("harness", "config"):
        url = f"{KAGGLE_API}/datasets/download/{owner}/{prefix}-{kind}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(root / kind)
    return root


def _stage_source(source: str, workdir: Path) -> Path:
    if source.startswith("kaggle:"):
        return _download_kaggle(source[len("kaggle:") :], workdir)
    path = Path(source).expanduser()
    if path.is_dir():
        return _single_root(path)
    if path.is_file() and path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as archive:
            archive.extractall(workdir / "tar", filter="data")
        return _single_root(workdir / "tar")
    raise FileNotFoundError(f"ARC-AGI-3 harness source {source!r} is not a directory, a tarball or kaggle:<ref>")


def ensure_harness(source: Optional[str], dest: Path = DEFAULT_DEST) -> Path:
    """Return a verified harness install at ``dest``, installing it from ``source`` when ``dest`` is absent."""
    manifest = load_manifest()
    if dest.exists():
        verify_install(dest, manifest)
        return dest
    if not source:
        raise FileNotFoundError(
            f"The ARC-AGI-3 harness is not installed at {dest}; set harness_source (directory, tarball or kaggle:<ref>)"
        )
    with tempfile.TemporaryDirectory(prefix="arc3_harness_") as tmp:
        staged = _stage_source(source, Path(tmp))
        verify_install(staged, manifest)
        partial = dest.with_name(dest.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        shutil.copytree(staged / "harness", partial / "harness")
        (partial / "config").mkdir()
        for rel in manifest["config_files"]:
            shutil.copy2(staged / "config" / rel, partial / "config" / rel)
        verify_install(partial, manifest)
        partial.rename(dest)
    return dest


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="directory, .tar.gz, or kaggle:<owner>/<prefix>")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    root = ensure_harness(args.source, args.dest)
    print(f"ARC-AGI-3 harness verified at {root}")


if __name__ == "__main__":
    main()
