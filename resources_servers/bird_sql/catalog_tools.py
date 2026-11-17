# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The arguments of the ``bird_sql`` catalog server's tools, and the reply of an accepted submission.

The tools are Auto Ontology's (https://github.com/NVIDIA/auto-ontology, ``auto_ontology/retrieval/text_to_sql/react``),
and so are their definitions: the rows offer the ones Auto Ontology's pinned version gives
(``benchmarks/birdbench/auto_ontology/react_prompt.py``). The routes take the arguments those definitions declare and
pass any others on, so Auto Ontology's function answers a wrong argument as its own loop does.
"""

from pydantic import BaseModel, ConfigDict


# What a submission the catalog accepted returns. verify grades the last submission that returned it.
ACCEPTED = "Accepted."


class RetrieveArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = ""
    kinds: list[str] | None = None


class RunSqlArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    sql: str = ""


class SubmitArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    sql: str = ""
    response: str = ""
    assumptions: str = ""
