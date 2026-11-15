# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench Verified OpenCode sessions recipe wires flat rows to a sandbox-lending server and OpenCode."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser
from resources_servers.swebench.app import SwebenchResourcesServerConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig


RECIPE = Path(__file__).parents[3] / "benchmarks/swebench/verified/opencode_sessions.yaml"


@pytest.fixture
def recipe():
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(RECIPE)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)
    return config


def test_recipe_runs_flat_rows_through_a_session_environment_whose_sandbox_outlives_the_episode(recipe) -> None:
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name="swebench_verified_opencode_sessions",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(
            recipe.swebench_verified_opencode_sessions.environment_servers.single_agent_turn_legacy, resolve=True
        ),
    )
    assert environment.resources_server.name == "swebench_verified_opencode_sessions_resources_server"
    assert environment.agent_server.name == "swebench_verified_opencode_sessions_agent"
    assert not environment.resources_tool_transports

    block = OmegaConf.to_container(recipe[environment.resources_server.name].resources_servers.swebench, resolve=True)
    resources = SwebenchResourcesServerConfig(
        name=environment.resources_server.name, host="localhost", port=8002, **block
    )
    [dataset] = block["datasets"]
    assert dataset["name"] == "swebench_verified" and dataset["type"] == "benchmark"
    assert block["allowed_agents"] == ["opencode_agent"]
    assert resources.apply_anti_cheating is False
    assert resources.sandbox_config["ready_timeout_s"] == 1200, "the override keeps the inherited sandbox settings"

    agent = recipe[environment.agent_server.name].responses_api_agents.opencode_agent
    # The flat rows reach the agent through the resources server it names.
    assert agent.resources_server.name == environment.resources_server.name
    pause_allowance = agent.checkpoint_pause_allowance_seconds
    assert pause_allowance == OpenCodeAgentConfig.model_fields["checkpoint_pause_allowance_seconds"].default
    # The session lives as long as the sandbox: it must outlive the episode and the time it may spend paused.
    assert environment.default_episode_timeout_seconds + pause_allowance < resources.sandbox_config["ttl_s"]
