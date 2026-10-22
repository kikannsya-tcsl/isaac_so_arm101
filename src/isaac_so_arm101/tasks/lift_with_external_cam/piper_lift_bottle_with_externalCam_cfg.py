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
from isaac_so_arm101.tasks.lift_with_external_cam.piper_lift_bottle_with_externalCam import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaac_so_arm101.robots.piper_description.piper import PIPER_CFG
from isaac_so_arm101.model import MODEL_DIR
from isaac_so_arm101.model.external_cam import return_camera_matrix

@configclass
class PiperLiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
        # Set the wrist camera
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_depth_optical_frame/camera",
            # data_types=["rgb", "distance_to_image_plane"],
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93, 
                horizontal_aperture=2.14 * 640 / 360,
                vertical_aperture=2.14,  
                clipping_range=(0.05, 1.5)),
            width=640, height=360,
            depth_clipping_behavior="max",
        )

        # Set the external camera
        self.scene.external_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_base/external_cam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0., 0.4, 0.176), # (0.4, 0, 0.185 - 0.009鋼板厚み)
                rot=(0.9347, 0.0, 0.0, -0.3554),  # 単位クォータニオン (w, x, y, z)。実機の取付姿勢が決まったら更新
                convention="world"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinstic_matrix=return_camera_matrix(),
                width=360, height=640,
                clipping_range=(0.05, 1.5)),
            width=640, height=360,
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

        # Set Cube as object
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
            debug_vis=False,
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
class PiperLiftCubeEnvCfg_PLAY(PiperLiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

