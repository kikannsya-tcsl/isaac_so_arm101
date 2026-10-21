
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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
# from . import mdp
import isaac_so_arm101.tasks.lift_with_cam.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaac_so_arm101.robots.piper_description.piper import PIPER_CFG
from isaac_so_arm101.model import MODEL_DIR

#from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


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
    # 手先(フィンガー)の接触センサ: 報酬計算用、agent env cfg で設定
    finger_contact: ContactSensorCfg = MISSING
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
        debug_vis=False,
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
        # 特権情報　object_ee_relative_pos　を含んだ状態で学習できるのか
        object_ee_relative_pos = ObsTerm(func=mdp.object_ee_relative_position)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
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
        # 指-ボトル間の接触（実機に指先力センサが無いため Critic のみ）
        finger_contact_forces = ObsTerm(func=mdp.finger_object_contact_forces, params={"window": 5, "scale": 1.0})
        finger_contact_flags = ObsTerm(func=mdp.finger_object_contact_flags, params={"threshold": 0.5, "window": 5})

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
    reaching_coarse = RewTerm(func=mdp.object_ee_distance, params={"grasp_offset_z": 0.06, "std": 0.5}, weight=0.1)  #初期学習 weight=1.0

    # 探索ノイズ由来のペナルティが接近報酬を埋もれさせないよう、初期は小さくする
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4) # 旧 weight=-1e-3

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )   

    # Neg Rew
    finger_non_object_contact = RewTerm(func=mdp.finger_non_object_contact, params={"threshold": 1.0}, weight=-0.5)

    # Objective rewards

    lifting_object = RewTerm(
        func=mdp.lifted_while_grasped,
        params={"lift_height": 0.1, "threshold": 0.5, "window": 5, "eval_grasp": False},
        weight=15.,
    )

    goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "lift_height": 0.1, "command_name": "object_pose"},
        weight=15.,  
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
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 60000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -5e-3, "num_steps": 60000}
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
        self.sim_freq = 100.0
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = int(self.sim_freq / self.control_freq) 
        self.sim.render_interval = int(self.sim_freq / self.drawing_freq)

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

@configclass
class PiperLiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
    
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
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
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

        # 手先(左右フィンガー)の接触センサ（報酬計算用）
        # filter_prim_paths_expr で対象物体との接触力のみを force_matrix_w (N, 2, 1, 3) に分離して取得する。
        # net_forces_w はテーブル・自己衝突を含む全接触力。
        # body の順序は USD の走査順なので、報酬側では body_names で解決すること。
        self.scene.finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link[12]",
            update_period=0.0,  # 毎物理ステップ更新
            history_length=self.decimation,  # 1制御ステップ分の履歴（瞬間的な接触の取りこぼし防止）
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            debug_vis=False,
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
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_base",
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
