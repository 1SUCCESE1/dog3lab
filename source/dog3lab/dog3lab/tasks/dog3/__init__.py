import gymnasium as gym

from .config import agents, dog3_env_cfg, dog3_flat_env_cfg

##
# Register Gym environments. Each task is wired to standard Instinct-RL PPO.
# NOTE: entry_point reuses instinctlab's InstinctRlEnv; only the task/asset/agent
# configs live in dog3lab (so dog3lab and instinctlab install side by side).
##

gym.register(
    id="Instinct-Locomotion-Flat-Dog3-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dog3_flat_env_cfg.Dog3FlatEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:Dog3FlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Flat-Dog3-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dog3_flat_env_cfg.Dog3FlatEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:Dog3FlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Rough-Dog3-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dog3_env_cfg.Dog3EnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:Dog3RoughPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Rough-Dog3-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dog3_env_cfg.Dog3EnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:Dog3RoughPPORunnerCfg",
    },
)
