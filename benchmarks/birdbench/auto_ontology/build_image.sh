#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the BIRD dev catalog image for the bird_sql catalog server.
#
#   benchmarks/birdbench/auto_ontology/build_image.sh MODELS_ENV [TAG]
#
# Run it from Gym's environment (it downloads BIRD dev with bird_sql's setup when missing). MODELS_ENV is a KEY=VALUE
# file with the model keys the catalog build uses: DEFAULT_MODELS_API_KEY, and EMBED_* or *_MODEL settings where they
# differ from Auto Ontology's defaults. It reaches the build as a BuildKit secret, so no layer holds it. TAG defaults
# to nemo-gym/bird-catalog:dev, the image catalog_tools.yaml names.
#
# The build calls the embedding and compile models for all 11 databases; on Claude Opus 5.5 it takes about a quarter
# of an hour.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
MODELS_ENV="${1:?usage: build_image.sh MODELS_ENV [TAG]}"
TAG="${2:-nemo-gym/bird-catalog:dev}"
[[ -f "$MODELS_ENV" ]] || { echo "build_image.sh: no $MODELS_ENV" >&2; exit 1; }

# The pin in pyproject.toml: `auto-ontology-server @ git+<repository>@<commit>`.
read -r REPOSITORY COMMIT < <(python3 - "$HERE/pyproject.toml" <<'PY'
import re, sys, tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
[pin] = [dep for dep in deps if dep.startswith("auto-ontology-server")]
match = re.fullmatch(r"auto-ontology-server @ git\+(\S+)@([0-9a-f]{40})", pin)
if match is None:
    raise SystemExit(f"auto-ontology-server must be pinned to a full commit, got {pin!r}")
print(*match.groups())
PY
)

DATABASES="$(cd "$ROOT" && python -c 'from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql; print(ensure_bird_sql())')"
DATABASES="$(realpath "$DATABASES")"

DOCKER_BUILDKIT=1 docker build \
  --file "$HERE/Dockerfile" \
  --build-context bird="$DATABASES" \
  --build-arg AUTO_ONTOLOGY_REPOSITORY="$REPOSITORY" \
  --build-arg AUTO_ONTOLOGY_COMMIT="$COMMIT" \
  --secret "id=models,src=$MODELS_ENV" \
  --tag "$TAG" \
  "$ROOT"
