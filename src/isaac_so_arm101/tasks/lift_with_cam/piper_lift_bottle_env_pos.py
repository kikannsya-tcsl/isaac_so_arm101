# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg
# from isaaclab.managers NotImplementedError
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaac_so_arm101.tasks.lift_with_cam.piper_lift_bottle_env import LiftEnvCfg, TerminationsCfg
import isaac_so_arm101.tasks.lift_with_cam.mdp as mdp
from isaac_so_arm101.robots.piper_description.piper import PIPER_CFG
from isaac_so_arm101.model import MODEL_DIR


@configclass
class PiperLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
        # Set the wrist camera
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_depth_optical_frame/camera",
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93, 
                horizontal_aperture=2.14 * 640 / 360,   # 正方画素を保つため縦アパーチャ x アスペクト比 (水平FOV ≈ 89°)
                vertical_aperture=2.14,                 # 垂直FOV 58° (D405実機に合わせる)
                clipping_range=(0.05, 1.5)),
            width=640, height=360,
            depth_clipping_behavior="max",
        )

        # override actions
        # 初期姿勢(全関節0)は joint2 が下限0・joint3 が上限0 ちょうどにあるため、
        # use_default_offset=True だと joint3 は正の出力がすべて上限で潰れて勾配が消え、
        # 「joint3 が全く動かない方策」に収束した。オフセットを可動域の内側に置く。
        # (joint2=1.0, joint3=-0.8 で EE ≈ (0.35, 0.0, 0.35)、コマンド領域の中心・ボトル上方)
        # 初期姿勢(PIPER_CFG.init_state)は実機ホームに合わせて 0 のまま。
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6" ],
            scale=0.5,
            use_default_offset=False,
            offset={"joint1": 0.0, "joint2": 1.0, "joint3": -0.8, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0},
            # 目標角を URDF のハード可動域内に制限(可動域外で目標が発散するのを防ぐ)
            clip={"joint2": (0.0, 3.1415926), "joint3": (-2.9670597, 0.0)},
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["piper_finger_joint7","piper_finger_joint8"],
            open_command_expr={"piper_finger_joint8": -0.05,"piper_finger_joint7":0.05},
            close_command_expr={"piper_finger_joint8": -0.001,"piper_finger_joint7":0.001},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["link6"]

        # Set Bottle as object to lift
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1004)),
            spawn=UsdFileCfg(
                usd_path=f"{MODEL_DIR}/bottle_model/bottle.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.125],
                    ),
                ),
            ],
        )


@configclass
class PiperLiftEnvCfg_PLAY(PiperLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

# 把持前姿勢へ寄せる項と settled 項で共通にする object_in_contact_course のパラメータ
_APPROACH_PARAMS = {
    "std_x": 0.05,
    "max_x": 0.12,
    "robot_cfg": SceneEntityCfg("robot", body_names=["link6"], joint_names=["piper_finger_joint7", "piper_finger_joint8"]),
}


@configclass
class RewardsCfgP1:
    """Phase1: ボトルを倒さず・動かさずに把持前の位置まで近づける。

    通常状態（ボトルを乱していない）では1ステップの合計が負にならないようにしてある。
    負になると「早く倒して episode を終わらせる」方が得になるため。
    """
    # --- 誘導（正） ---
    # 遠方からの粗い誘導。std が大きいので、ボトルを押しのける方向の勾配は弱い
    reaching_coarse = RewTerm(func=mdp.object_ee_distance, params={"grasp_offset_z": 0.06, "std": 0.3}, weight=1.0)
    # 向き・高さ・指の間を満たしたときだけ出る。P1 の本体
    approach_pose = RewTerm(func=mdp.object_in_contact_course, params=_APPROACH_PARAMS, weight=5.0)
    # ボトルが直立かつ reset 位置から 1 cm 以内のまま approach_pose を満たしている状態
    settled = RewTerm(
        func=mdp.approach_settled,
        params={**_APPROACH_PARAMS, "max_displacement": 0.01, "max_tilt": 0.1},
        weight=5.0,
    )

    # --- ボトルを守る（負） ---
    object_tilt = RewTerm(func=mdp.object_tilt, weight=-5.0)
    # 2 cm ずらすと -0.6/step。2Hz では速度より状態で罰する方が確実
    object_displacement = RewTerm(func=mdp.object_displacement, weight=-30.0)
    # 転倒・落下で終了したステップに一度だけ課す
    terminated = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["object_tipped_over", "object_dropping"]},
        weight=-20.0,
    )
    # P1 ではグリッパは常に開いたまま（far_distance=0, std=0 で常時判定）
    gripper_closing = RewTerm(
        func=mdp.gripper_closed_far_from_object,
        params={"far_distance": 0.0, "std": 0.0, "lift_height": 0.05, "grasp_offset_z": 0.06},
        weight=-0.5,
    )

    # 名前は CurriculumCfg から参照されるので変えないこと
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfgP1(TerminationsCfg):
    """base の time_out / object_dropping に転倒終了を追加。"""
    object_tipped_over = DoneTerm(func=mdp.object_tipped_over, params={"max_angle": 0.5})


@configclass
class PiperLiftEnvCfgP1(PiperLiftEnvCfg):
    rewards: RewardsCfgP1 = RewardsCfgP1()
    terminations: TerminationsCfgP1 = TerminationsCfgP1()


@configclass
class PiperLiftEnvCfgP1_PLAY(PiperLiftEnvCfgP1):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
