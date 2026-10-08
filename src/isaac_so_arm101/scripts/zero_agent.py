# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    base_env = env.unwrapped
    step = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            obs, rew, terminated, truncated, extras = env.step(actions)
            
            if step % 10 == 0:
                print(f"\n===== step {step}  total reward(env0) = {rew[0].item():.4f} =====")
                # 報酬：各項の値（weight 掛け済み、dt は掛けない）
                for name, val in base_env.reward_manager.get_active_iterable_terms(0):
                    print(f"  [rew] {name:28s} {val[0]: .4f}")
                # 観測：critic グループの各項
                for name, val in base_env.observation_manager.get_active_iterable_terms(0):
                    if name.startswith("critic-"):
                        print(f"  [obs] {name:40s} {[round(v, 3) for v in val]}")
            step += 1
    # close the simulator
    env.close()

    return 0


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()