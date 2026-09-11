"""Record dog3 joint trajectories in Isaac under the trained policy, for gait
comparison against the Gazebo sim2sim.  Writes name/position lines per step.

Usage:
  ./run.sh python scripts/instinct_rl/record_gait.py \
      --task=Instinct-Locomotion-Flat-Dog3-Play-v0 --headless \
      --load_run=20260910_154219 --checkpoint=model_6000.pt \
      --cmd_vx=0.5 --steps=300 --out=/tmp/isaac_gait.txt
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Instinct-Locomotion-Flat-Dog3-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cmd_vx", type=float, default=0.5)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, default="/tmp/isaac_gait.txt")
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
import dog3lab.tasks  # noqa: E402,F401
from instinct_rl.runners import OnPolicyRunner  # noqa: E402
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    # pin the command to a constant forward velocity
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    r.lin_vel_y = (0.0, 0.0)
    r.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False

    agent_cfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))

    log_root = os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))
    ckpt = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
    print(f"[record] loading {ckpt}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uw = env.unwrapped
    robot = uw.scene["robot"]
    names = list(robot.joint_names)
    obs, _ = env.get_observations()

    f = open(args_cli.out, "w")
    f.write("names: " + " ".join(names) + "\n")
    with torch.inference_mode():
        for k in range(args_cli.steps):
            obs, _, _, _ = env.step(policy(obs))
            q = robot.data.joint_pos[0].detach().cpu().tolist()
            f.write("q: " + " ".join(f"{v:.6f}" for v in q) + "\n")
    f.close()
    print(f"[record] wrote {args_cli.steps} steps to {args_cli.out}", flush=True)
    try:
        simulation_app.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
