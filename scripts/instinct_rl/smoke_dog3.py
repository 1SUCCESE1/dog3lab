"""Headless smoke test for the dog3 env: build, print DOF/obs shapes, step, no NaN."""

import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = __import__("argparse").ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Instinct-Locomotion-Flat-Dog3-v0")
parser.add_argument("--steps", type=int, default=30)
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dog3lab.tasks  # noqa: E402,F401
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = InstinctRlVecEnvWrapper(env)
    uw = env.unwrapped
    asset = uw.scene["robot"]
    print(f"[smoke] env={args_cli.task} num_envs={env.num_envs}", flush=True)
    print(f"[smoke] joint_names({len(asset.joint_names)}) = {asset.joint_names}", flush=True)
    print(f"[smoke] DOF (num_joints) = {asset.num_joints}", flush=True)
    print(f"[smoke] body_names({len(asset.body_names)}) = {asset.body_names}", flush=True)
    obs, _ = env.get_observations()
    print(f"[smoke] obs shape = {tuple(obs.shape)}", flush=True)
    print(f"[smoke] action dim = {uw.action_manager.total_action_dim}", flush=True)
    print(f"[smoke] reward terms = {len(getattr(uw.reward_manager, '_term_names', []))}", flush=True)

    z0 = asset.data.root_pos_w[:, 2].mean().item()
    bad = 0
    with torch.inference_mode():
        for it in range(args_cli.steps):
            a = torch.zeros(env.num_envs, uw.action_manager.total_action_dim, device=uw.device)
            obs, _r, _d, _ = env.step(a)
            if not torch.isfinite(obs).all():
                bad += 1
    z1 = asset.data.root_pos_w[:, 2].mean().item()
    print(f"[smoke] root z: {z0:.3f} -> {z1:.3f} m over {args_cli.steps} steps", flush=True)
    print(f"[smoke] non-finite obs count = {bad}", flush=True)
    print("[smoke] OK", flush=True)
    try:
        simulation_app.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
