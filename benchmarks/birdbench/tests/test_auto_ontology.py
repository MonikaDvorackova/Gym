# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BIRD catalog image's inputs: Gym's pin of Auto Ontology, BIRD's descriptions, and Auto Ontology's prompt."""

import re
import tomllib
from pathlib import Path

import pytest

from benchmarks.birdbench.auto_ontology.build_catalog import column_descriptions


AUTO_ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "auto_ontology"


def test_auto_ontology_is_pinned_once_to_a_full_commit():
    """build_image.sh reads the commit from here; the prepare project resolves the same one."""
    dependencies = tomllib.loads((AUTO_ONTOLOGY_DIR / "pyproject.toml").read_text())["project"]["dependencies"]
    [pin] = [dependency for dependency in dependencies if dependency.startswith("auto-ontology-server")]
    match = re.fullmatch(r"auto-ontology-server @ git\+(\S+)@([0-9a-f]{40})", pin)
    assert match is not None, pin
    lock = (AUTO_ONTOLOGY_DIR / "uv.lock").read_text()
    assert f"rev={match.group(2)}#{match.group(2)}" in lock


def test_bird_descriptions_join_the_column_and_value_descriptions(tmp_path):
    (tmp_path / "frpm.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "CDSCode,CDS code,California school code,text,not useful\n"
        'Charter School (Y/N), ,Charter school,integer,"0: no\n1: yes"\n'
        ",,orphan,,\n"
    )
    (tmp_path / "schools.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "City,,Ciudad — city,text,\n".encode("cp1252")
    )

    assert column_descriptions(tmp_path) == [
        {"table_name": "frpm", "column_name": "CDSCode", "description": "California school code"},
        {"table_name": "frpm", "column_name": "Charter School (Y/N)", "description": "Charter school, 0: no 1: yes"},
        {"table_name": "schools", "column_name": "City", "description": "Ciudad — city"},
    ]


def test_the_rows_start_as_auto_ontologys_sql_episode():
    pytest.importorskip("auto_ontology", reason="Auto Ontology is installed by the auto_ontology prepare project")
    from auto_ontology.retrieval.text_to_sql.react import prompts

    from benchmarks.birdbench.auto_ontology.react_prompt import episode_input, tool_definitions

    system, user = episode_input("How many schools?", "school = row")

    assert system["content"].startswith("You answer a question about a database by writing one sqlite SQL query.")
    assert "SQLite" in system["content"]
    assert user["content"] == prompts.user_message("How many schools?", "school = row")
    assert [tool["name"] for tool in tool_definitions()] == ["retrieve", "run_sql", "submit"]
