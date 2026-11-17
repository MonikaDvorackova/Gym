# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What Auto Ontology's ReAct agent sends its model at the start of an episode, from the Auto Ontology pyproject.toml pins.

``run_episode`` (``auto_ontology/retrieval/text_to_sql/react/loop.py``) starts an SQL episode with a system prompt,
the SQL mode's prompt followed by the dialect's rules, and a user message with the question and its evidence apart,
and offers the SQL mode's tools. This module builds the same, from Auto Ontology's own functions, for the rows of
``catalog_tools.yaml``: the bird_sql catalog server serves those tools, so any agent starts where Auto Ontology's does.

It imports Auto Ontology, so it runs in this directory's project, the prepare project of ``catalog_tools.yaml``.
"""

from typing import Any, Optional


# BIRD's databases are SQLite; Auto Ontology's SQLite connector names its dialect this.
DIALECT = "sqlite"


def episode_input(question: str, evidence: Optional[str]) -> list[dict[str, str]]:
    """The system and user messages that start Auto Ontology's SQL episode on ``question``."""
    from auto_ontology.retrieval.text_to_sql.prompts import format_dialect_rules
    from auto_ontology.retrieval.text_to_sql.react import prompts
    from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode

    system = prompts.system_prompt_for(EpisodeMode.SQL, DIALECT, information_answers=False)
    rules = format_dialect_rules(DIALECT).strip()
    if rules:
        system = f"{system}\n\n{rules}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.user_message(question, evidence)},
    ]


def tool_definitions() -> list[dict[str, Any]]:
    """Auto Ontology's tools for an SQL episode, as Responses API function tools."""
    from auto_ontology.retrieval.text_to_sql.react import prompts
    from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode

    return [
        {
            "type": "function",
            "name": str(tool["function"]["name"]),
            "description": tool["function"].get("description", ""),
            "parameters": tool["function"]["parameters"],
            "strict": False,
        }
        for tool in prompts.tools_for(EpisodeMode.SQL, information_answers=False)
    ]
