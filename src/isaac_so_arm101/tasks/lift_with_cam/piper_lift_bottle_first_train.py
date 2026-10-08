
# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# from . import mdp
import isaac_so_arm101.tasks.lift_with_cam.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.utils.offset import OffsetCfg
#from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # d405 cam
    wrist_cam: TiledCameraCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(10, 10),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.35),
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 0.35),
            roll=(90.0, 90.0),
            pitch=(0.0, 0.0),
            yaw=(90.0, 90.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyObservationsCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        rgbd = ObsTerm(func=mdp.rgbd_features, 
                       params={"sensor_cfg": SceneEntityCfg("wrist_cam"),
                               "max_range": 1.5,
                               "verify_every": 5000,},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticObsercationsCfg(ObsGroup):
        """"特権情報を含む（Critic専用、非対称アクタークリティック）"""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        ee_pose_in_root = ObsTerm(func=mdp.ee_pose_in_robot_root_frame)
        object_pose_in_root = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        object_ee_relative_pos = ObsTerm(func=mdp.object_ee_relative_position)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    # observation groups
    policy: PolicyObservationsCfg = PolicyObservationsCfg()
    critic: CriticObsercationsCfg = CriticObsercationsCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.2), "y": (-0.1, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": (0.03, 0.55),
                "operation": "abs", "distribution": "uniform",
                "recompute_inertia": True},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Positive rewards for approching.
    # 初期姿勢の手先は狙い点から ~0.25 m 離れている。std=0.16 ではそこで勾配がほぼ無く
    # 「動かない」方策に収束したため、coarse を std=0.3 にして主たる誘導項にする。
    # fine は最終の詰め専用（遠方では飽和するので重みを lift より小さく抑える）。
    reaching_coarse = RewTerm(func=mdp.object_ee_distance, params={"grasp_offset_z": 0.06, "std": 0.3}, weight=20.0)  #旧 std=0.16, weight=5.0
    reaching_fine = RewTerm(func=mdp.object_ee_distance, params={"grasp_offset_z": 0.06, "std": 0.05}, weight=25.0)  #旧 std=0.08, weight=50.0

    contact_course = RewTerm(func=mdp.object_in_collision_course, 
                                         params={
                                             "std_x": 0.1,
                                             "max_x": 0.5,
                                             "robot_cfg": SceneEntityCfg("robot", body_names=["link6"],joint_names=["piper_finger_joint7", "piper_finger_joint8"],),
                                         }, weight=2.5e1)  #旧 weight=5.0e1

    # Positive rewards for gripping.
    # 狙い点は reaching_* と同じ grasp_offset_z=0.06 に揃える（None だと重心狙いになり、
    # reaching が誘導する高さと軸方向に ~5 cm ずれて proximity がほぼ 0 になる）。
    # miss_scale>0 で「狙い点付近で閉じる」こと自体にも小さく報酬を出し、held までの探索を誘導する。
    closed_near_obj = RewTerm(
        func=mdp.gripper_close_near_object,
        params={
            "std": 0.03,
            "axial_std": 0.04,
            "miss_scale": 0.2,
            "grasp_offset_z": 0.06,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["piper_finger_joint7", "piper_finger_joint8"]),
        },
        weight=5.0e1,
        )   


    # Penalty rewards
    object_tilt = RewTerm(func=mdp.object_tilt, weight=-1.5)  # 初期学習 weight=-1.0

    # -1e4 だと1回の落下(×dt=0.5 で -5000)がエピソード中の接近報酬の総和を大きく上回り、
    # 「ボトルに近づかない」方策が最適になっていた。終了条件でも打ち切られるので罰は控えめにする。
    object_dropping = RewTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
            weight=-1.0e2,
        )

    # 探索ノイズ由来のペナルティが接近報酬を埋もれさせないよう、初期は小さくする
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4) # 旧 weight=-1e-3

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4, # 旧 weight=-1e-2
        params={"asset_cfg": SceneEntityCfg("robot")},
    )   


    # Objective rewards
    # 接近 < 把持 < 持ち上げ ≦ 目標追従 の順に大きくし、「近くで待機」より「掴んで上げる」を得にする
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"lift_height":  0.1}, weight=2.0e2)  #旧 weight=1.0e2

    goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "lift_height": 0.1, "command_name": "object_pose"},
        weight=2.0e2,  #旧 weight=1.6e2
    )

    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # num_steps は env ステップ数（20 step/iter）。旧値 10000 = iter 500 で、接近を覚える前に
    # -0.1 が効いて報酬が崩壊したため、把持・持ち上げ習得後（~iter 3000）に -1e-2 へ上げる。
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 60000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 60000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=32, env_spacing=2.5)
    
    # Render settings
    # Using bottle.usd which is contained transprancy


    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
            render=sim_utils.RenderCfg(
                carb_settings={
                    "/rtx/raytracing/fractionalCutoutOpacity": True,
                    "/rtx/translucency/enabled": True,
                    "/rtx/translucency/maxRefractionBounces": 6,
                    "/rtx/translucency/reflectAtAllBounce": True,
                    "/rtx/material/translucencyAsOpacity": False,
                },
            ),
          )

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 10 # 1エピソード60step(2Hz)
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.drawing_freq = 60
        self.control_freq = 2 # 2Hzで制御信号を送信
        self.sim_freq = 200.0
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = int(self.sim_freq / self.control_freq) 
        self.sim.render_interval = int(self.sim_freq / self.drawing_freq)

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625