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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
# from isaac_so_arm101.tasks.lift.lift_env_cfg import LiftEnvCfg
from isaac_so_arm101.tasks.lift_with_cam.piper_lift_bottle_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaac_so_arm101.robots.piper_description.piper import PIPER_CFG
from isaaclab.sensors import TiledCameraCfg
from isaac_so_arm101.model import MODEL_DIR

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
                focal_length=1.93, horizontal_aperture=3.8,  # D405実機の内部パラメータに寄せる
                clipping_range=(0.05, 2.0)),
            width=224, height=224,
            depth_clipping_behavior="max",
        )

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            #joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6" ],
            joint_names=["joint1", "joint2", "joint3", "joint5", "joint6" ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["piper_finger_joint7","piper_finger_joint8"],
            open_command_expr={"piper_finger_joint8": -0.9,"piper_finger_joint7":0.9},
            close_command_expr={"piper_finger_joint8": -0.001,"piper_finger_joint7":0.001},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["link6"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.13], rot=[1, 0, 0, 0]),
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
class PiperLiftCubeEnvCfg_PLAY(PiperLiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

