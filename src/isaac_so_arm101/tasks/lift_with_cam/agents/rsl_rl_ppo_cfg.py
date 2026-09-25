# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Learn log(std) instead of std directly so std stays positive.
        # This avoids: RuntimeError: normal expects all elements of std >= 0.0
        noise_std_type="log",
        actor_hidden_dims=[512, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2, 
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LiftCubeCameraPPORunnerCfg(LiftCubePPORunnerCfg):
    """RGB-D凍結特徴（512次元）を actor 入力に持つ非対称アクター・クリティック版。"""

    experiment_name = "lift_cube_with_camera"

    # --- 観測グループ → ネットワークの対応付け（rsl-rl 2.3系で必須）---
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # --- バッチ長：num_envs が 4096 → 128 に落ちる分を rollout 長で補う ---
    num_steps_per_env = 96
    max_iterations = 6000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        # actor は 512+α 次元を受けるので入口を広げる
        actor_hidden_dims=[1024, 256, 64],
        # critic は状態のみ（20次元程度）なので据え置きで十分
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        # 凍結特徴はスケール無調整。ここを True にしないと PPO が壊れる
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,          # 視覚探索は難しくなるので気持ち上げる
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,       # 観測正規化が入る前提で常識的な値に戻す
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LiftBottleCameraPPORunnerCfg(LiftCubePPORunnerCfg):
    """RGB-D凍結特徴（512次元）を actor 入力に持つ非対称アクター・クリティック版。"""

    experiment_name = "lift_bottle_with_camera"

    # --- 観測グループ → ネットワークの対応付け（rsl-rl 2.3系で必須）---
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # --- バッチ長：num_envs が 4096 → 128 に落ちる分を rollout 長で補う ---
    num_steps_per_env = 96
    max_iterations = 6000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        noise_std_type="log",
        # actor は 512+α 次元を受けるので入口を広げる
        actor_hidden_dims=[1024, 256, 64],
        # critic は状態のみ（20次元程度）なので据え置きで十分
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        # 凍結特徴はスケール無調整。ここを True にしないと PPO が壊れる
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,          # 視覚探索は難しくなるので気持ち上げる
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,       # 観測正規化が入る前提で常識的な値に戻す
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )