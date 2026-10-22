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
class LiftBottleCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RGB-D凍結特徴（512次元）を actor 入力に持つ非対称アクター・クリティック版。"""
    save_interval = 50
    experiment_name = "lift_bottle_with_camera"
    empirical_normalization = False
    
    # --- 観測グループ → ネットワークの対応付け（rsl-rl 2.3系で必須）---
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # --- バッチ長：num_envs が 4096 → 128 に落ちる分を rollout 長で補う ---
    num_steps_per_env = 20 # 2Hz x 10 sec/episode
    max_iterations = 6000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[1024, 512, 128, 64],
        critic_hidden_dims=[1024, 512, 128, 64],
        activation="elu",
        # 凍結特徴はスケール無調整。ここを True にしないと PPO が壊れる
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        # 値クリップは旧推定から ±clip_param(0.2) しか動けず、価値のスケールが大きいと
        # critic が追従できず value loss が発散したため無効化
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.008,          # 視覚探索は難しくなるので気持ち上げる
        # バッチ = num_envs x 10 = 640 (64env)。2 分割でミニバッチ 320、
        # 1 iter あたり 5 epoch x 2 = 10 回更新（各サンプルを 5 回再利用）
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-4,   
        schedule="adaptive",
        # 2Hz・20 step/episode。gamma=0 だと把持→持ち上げの因果を学べないので
        # 実効ホライズン 1/(1-gamma)=20 step（≒1 episode）にする
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LiftCubeCameraPPORunnerCfg(LiftBottleCameraPPORunnerCfg):
    experiment_name = "lift_cube_with_camera"
    policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="log",
            actor_hidden_dims=[512, 128, 64],
            critic_hidden_dims=[512, 128, 64],
            activation="elu",
            # 凍結特徴はスケール無調整。ここを True にしないと PPO が壊れる
            actor_obs_normalization=True,
            critic_obs_normalization=True,
        )