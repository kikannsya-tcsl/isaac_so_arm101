# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import FrameTransformer, TiledCamera
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# ボトル形状定数
#
# Script Editorでの実測値（/Bottle, metersPerUnit=1.0, upAxis=Z）
#   local size   : (0.06650, 0.06650, 0.21146)
#   local center : (0, 0, 0.00584)   ← プリム原点から見た形状中心のオフセット
#
# したがってプリム原点は形状中心より 5.84 mm 下、底面より 99.89 mm 上にある。
# USD側で原点を移動した場合は下の3値を必ず更新すること。
# ---------------------------------------------------------------------------
BOTTLE_HEIGHT: float = 0.21146
BOTTLE_RADIUS: float = 0.03325

ORIGIN_TO_BOTTOM: float = 0.09989  # プリム原点 -> 底面
ORIGIN_TO_TOP: float = 0.11157     # プリム原点 -> 上端

# 横倒しになったときの底面中心の高さ（= 半径）。
# lift_height はこの値より大きくしないと「転倒＝持ち上げ」と誤判定される。
TIPPED_OVER_CLEARANCE: float = BOTTLE_RADIUS


# ---------------------------------------------------------------------------
# 内部ヘルパ
# ---------------------------------------------------------------------------
def _bottom_clearance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    surface_height: float,
) -> torch.Tensor:
    """ボトル底面中心の、載置面からの高さ [m] を返す。

    重心（COM）ではなくリンク原点を使う。COMはPhysXがコリジョン形状から
    自動計算するため値が不定で、しかも傾くと高さが変わってしまうため、
    「持ち上げたか」の判定基準には向かない。

    ボトルのローカルZ軸まわりの傾きも考慮するので、直立でも傾斜でも
    底面中心の実際のワールド高さが得られる。
    （厳密な最下点ではなく底面の中心点である点に注意。横倒し時は
     底面中心の高さ = 半径 になる。）
    """
    obj: RigidObject = env.scene[object_cfg.name]

    # ローカルZ軸のワールドZ成分（直立=1、真横=0）
    up_z = matrix_from_quat(obj.data.root_link_quat_w)[:, 2, 2]

    bottom_z_w = (
        obj.data.root_link_pos_w[:, 2]
        - ORIGIN_TO_BOTTOM * up_z
    )

    return (
        bottom_z_w
        - env.scene.env_origins[:, 2]
        - surface_height
    )


# ---------------------------------------------------------------------------
# 報酬関数
# ---------------------------------------------------------------------------
def object_in_camera_view(
    env,
    std: float = 0.6,
    margin: float = 4.0,
    min_depth: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
) -> torch.Tensor:
    cam: TiledCamera = env.scene[sensor_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # オブジェクト重心のワールド座標
    object_com_pos_w = obj.data.root_com_pos_w

    # world -> camera光学座標系
    # ROS optical frame: +x右、+y下、+z前方
    rel_w = object_com_pos_w - cam.data.pos_w
    pos_c = math_utils.quat_apply_inverse(
        cam.data.quat_w_ros,
        rel_w,
    )

    K = cam.data.intrinsic_matrices

    z = pos_c[:, 2]
    zs = z.clamp(min=1.0e-4)

    u = K[:, 0, 0] * pos_c[:, 0] / zs + K[:, 0, 2]
    v = K[:, 1, 1] * pos_c[:, 1] / zs + K[:, 1, 2]

    H, W = cam.image_shape

    visible = (
        (z > min_depth)
        & (u > margin)
        & (u < W - margin)
        & (v > margin)
        & (v < H - margin)
    )

    # 画面中心に近いほど高い連続報酬（0～1）
    du = (u - K[:, 0, 2]) / (0.5 * W)
    dv = (v - K[:, 1, 2]) / (0.5 * H)

    reward = torch.exp(
        -(du.square() + dv.square()) / (std**2)
    )

    return torch.where(
        visible,
        reward,
        torch.zeros_like(reward),
    )


def object_tilt(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    0 = 直立
    1 = 真横（横転）以上の傾き。倒立側も1で飽和させる

    ボトルのローカルZ軸とワールドZ軸のなす角 θ = arccos(up_z) を求め、
    θ / (π/2) を返す。転倒角に比例する（45°で0.5）。
    負の重みで使うと横転時に報酬が最小。
    """
    obj: RigidObject = env.scene[asset_cfg.name]

    # 物体形状のローカル軸を評価するため、COM姿勢ではなくlink姿勢を使う
    rotation_matrix = matrix_from_quat(
        obj.data.root_link_quat_w
    )

    # ローカルZ軸のワールドZ成分
    up_z = rotation_matrix[:, 2, 2]

    # 数値誤差で|up_z|>1になるとacosがNaNを返すのでクランプする
    tilt_angle = torch.acos(up_z.clamp(-1.0, 1.0))  # [rad], 0〜π

    # 直立(0rad)で0、真横(π/2)で1、それ以上傾いても1のまま
    return (tilt_angle / (0.5 * torch.pi)).clamp(0.0, 1.0)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    lift_height: float,
    surface_height: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """ボトル底面が載置面から lift_height 以上離れていれば1を返す。

    Args:
        lift_height: 載置面からの必要クリアランス [m]。
            転倒による誤判定を避けるため BOTTLE_RADIUS (0.033) より
            大きい値にすること。0.05 程度を推奨。
        surface_height: 載置面（机の天面）のenv原点からの高さ [m]。
            机の天面がenv原点と同じ高さなら 0.0。
    """
    clearance = _bottom_clearance(env, object_cfg, surface_height)

    return (clearance > lift_height).to(dtype=clearance.dtype)


def object_disturbance(
    env: ManagerBasedRLEnv,
    lift_height: float,
    std: float,
    surface_height: float = 0.0,
    distance_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # オブジェクト重心
    object_com_pos_w = obj.data.root_com_pos_w

    # エンドエフェクタ位置
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    object_ee_distance = torch.linalg.vector_norm(
        object_com_pos_w - ee_pos_w,
        dim=1,
    )

    # root_com_lin_vel_wは重心の並進速度
    linear_velocity = torch.linalg.vector_norm(
        obj.data.root_com_lin_vel_w,
        dim=-1,
    )
    angular_velocity = torch.linalg.vector_norm(
        obj.data.root_com_ang_vel_w,
        dim=-1,
    )

    disturbance_velocity = (
        linear_velocity + 0.1 * angular_velocity
    )

    clearance = _bottom_clearance(env, asset_cfg, surface_height)

    not_lifted = (
        clearance < lift_height
    ).to(dtype=object_com_pos_w.dtype)

    far = (
        object_ee_distance / std > distance_threshold
    ).to(dtype=object_com_pos_w.dtype)

    return disturbance_velocity * not_lifted
    # return disturbance_velocity * not_lifted * far


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_offset_z: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """手先とオブジェクト上の目標点との距離に基づく報酬。

    Args:
        grasp_offset_z: ボトルのローカルZ軸に沿った、プリム原点から
            把持狙い点までのオフセット [m]。None なら従来どおり重心を狙う。
            ボトルは細長いので、胴体の掴みやすい高さを明示したい場合に使う。
            例: 底面から60 mmを狙う -> 0.060 - ORIGIN_TO_BOTTOM = -0.040
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    if grasp_offset_z is None:
        target_pos_w = obj.data.root_com_pos_w
    else:
        # ボトルのローカルZ軸（回転行列の第3列）に沿ってオフセット
        local_z_w = matrix_from_quat(
            obj.data.root_link_quat_w
        )[:, :, 2]
        target_pos_w = (
            obj.data.root_link_pos_w
            + grasp_offset_z * local_z_w
        )

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.linalg.vector_norm(
        target_pos_w - ee_pos_w,
        dim=1,
    )

    return 1.0 - torch.tanh(distance / std)

def object_in_contact_course(
    env: ManagerBasedRLEnv,
    std_x: float = 0.10,
    max_x: float = 0.07,
    mouth_offset: float = 0.07,
    opening_width: float | None = 0.10,
    pad_thickness: float = 0.007,
    lateral_margin: float = 0.005,
    height_margin: float = 0.02,
    max_tilt: float = 0.6,
    k: float = 0.005,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["link6"],
        joint_names=["piper_finger_joint7", "piper_finger_joint8"],
    ),
) -> torch.Tensor:
    """
    ボトルの軸が指の間に入るように接近しているかを報酬化する。

    グリッパ座標系（右手系）と link6 の軸の対応:
        x_g = link6 +z : 接近方向（link6 からグリッパへ向かう向き）
        y_g = link6 +y : 指の開閉方向（指間の線分の向き）。
                         URDF 上 flange_link / gripper_base は link6 と同じ姿勢で、
                         joint7 (q>0) は -y、joint8 (q<0) は +y へ開く
        z_g = link6 -x : x_g × y_g。D405 が取り付けられている側
                         （camera_joint は gripper_base の x=-0.0425）

    面A = 指間の線分を x_g 方向に掃いた面 (z_g = 0)
    面B = 指間の線分を z_g 方向に掃いた面 (x_g = 0)。原点は link6 から
          +z 方向に mouth_offset の位置（指の付け根）に置く。

    ボトルを円柱とみなし、面Aで切った断面（傾いていれば楕円）について
    「面Aがボトル胴体を切っている」「断面が指の間に収まる」「断面が面Bより前方」
    の3条件をゲートとして掛け合わせ、面Bからの距離で整形した値を返す。
    戻り値は 0～1。

    Args:
        std_x: 前方距離 d（面Bから断面の手前端までの x_g 距離）の整形スケール [m]。
            報酬は 1 - tanh(d / std_x) で、指の付け根に近いほど大きい。
        max_x: 断面中心の x_g 座標（面Bからの距離）の上限 [m]。これより遠いと 0。
            硬判定で、k の影響を受けない。断面中心は少なくとも half_x（直立時は
            BOTTLE_RADIUS）だけ面Bより前にないと前方ゲートを通らないので、
            実際に報酬が出るのは half_x <= p_x < max_x の範囲になる。
        mouth_offset: link6 原点から面B（指の付け根）までの、link6 +z 方向の距離 [m]。
        opening_width: 指の開口幅 [m]。既定は全開時の値で固定し、把持のために
            閉じても報酬が消えないようにする。None なら現在の指関節位置
            |q7| + |q8|（指リンク原点間の距離）を使い、開いたまま接近することも
            要求する（閉じた瞬間に報酬が消える点に注意）。
        pad_thickness: 指1本あたりのパッド厚 [m]。開口幅から 2 倍を差し引いて
            指内面間の実効幅にする。
        lateral_margin: 断面の端と、近い側の指内面との間に要求する隙間 [m]。
            条件は |p_y| + half_y + lateral_margin <= (opening_width - 2*pad_thickness) / 2。
            指先がボトルの縁に当たって押し倒すのを防ぐ。
        height_margin: 面Aがボトルを切る位置の、底面・上端それぞれからの最小距離 [m]。
            ボトル軸に沿って測る（世界の鉛直ではない）。縁での把持、指と机の干渉、
            端で断面が欠けて半幅の式が成り立たなくなる領域を除外する。
        max_tilt: 許容角 [rad]。次の2つの硬判定に使う。
            (1) グリッパの z_g と世界の鉛直軸のなす角。接近方向が水平から
                ±max_tilt 以内のときだけ報酬を出す（真上からの接近を除外）。
            (2) z_g とボトル軸のなす角。これを超えると面Aがボトル軸とほぼ平行に
                なり交点・断面が定まらないので 0 にする。(1) と合わせて、
                倒れたボトルにも報酬は出ない。
        k: 各ゲート（sigmoid）の遷移幅 [m]。境界ちょうどでゲート値 0.5、
            およそ ±k で 0～1 に遷移する。0 ならステップ関数（硬判定）。
        object_cfg: ボトルの RigidObject。
        robot_cfg: body_names に link6、joint_names に指関節2本を指定したもの。
            既定値はマネージャに解決されず body_ids が slice のままになるため、
            RewTermCfg の params に必ず明示的に渡すこと。
    """
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    def gate(margin):
        return torch.sigmoid(margin / k) if k > 0 else (margin > 0).float()

    # グリッパ座標系 (列 = x_g, y_g, z_g) を link6 から構成
    bid = robot_cfg.body_ids[0]
    R6 = matrix_from_quat(robot.data.body_link_quat_w[:, bid])
    x_g = R6[:, :, 2]   # link6 +z: 接近方向
    y_g = R6[:, :, 1]   # link6 +y: 指の開閉方向
    z_g = -R6[:, :, 0]  # link6 -x: x_g × y_g
    R_g = torch.stack([x_g, y_g, z_g], dim=-1)
    origin = robot.data.body_link_pos_w[:, bid] + mouth_offset * x_g

    # ボトル軸と底面中心をグリッパ座標へ
    a_w = matrix_from_quat(obj.data.root_link_quat_w)[:, :, 2]
    bottom_w = obj.data.root_link_pos_w - ORIGIN_TO_BOTTOM * a_w
    a = torch.einsum("nij,ni->nj", R_g, a_w)
    b = torch.einsum("nij,ni->nj", R_g, bottom_w - origin)

    # 面A(z_g=0)との交点
    cos_min = math.cos(max_tilt)
    # z_g が世界の鉛直に近いこと（= 接近方向がほぼ水平）
    aligned = (z_g[:, 2].abs() > cos_min).float()
    # ボトル軸が面Aを横切り、断面が定まること
    section_valid = (a[:, 2].abs() > cos_min).float()
    a_z = torch.where(a[:, 2] >= 0, 1.0, -1.0) * a[:, 2].abs().clamp(min=cos_min)
    s = -b[:, 2] / a_z
    p = b + s.unsqueeze(-1) * a
    hit_a = gate(s - height_margin) * gate(BOTTLE_HEIGHT - height_margin - s)

    # 斜め断面（楕円）の半幅
    half_y = BOTTLE_RADIUS * torch.sqrt(a[:, 1] ** 2 + a_z ** 2) / a_z.abs()
    half_x = BOTTLE_RADIUS * torch.sqrt(a[:, 0] ** 2 + a_z ** 2) / a_z.abs()

    # 指の間
    if opening_width is None:
        w = robot.data.joint_pos[:, robot_cfg.joint_ids].abs().sum(-1)
    else:
        w = torch.full_like(s, opening_width)
    w = w - 2.0 * pad_thickness
    between = gate(0.5 * w - half_y - p[:, 1].abs() - lateral_margin)

    # 面Bより前方
    d = p[:, 0] - half_x
    front = gate(d) * (p[:, 0] < max_x).float()
    shaping = 1.0 - torch.tanh(d.clamp(min=0.0) / std_x)

    # 各因子は 0～1。スケールは RewTerm の weight 側で調整する
    return aligned * section_valid * hit_a * between * front * shaping


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    lift_height: float,
    command_name: str,
    surface_height: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """オブジェクト重心と目標位置との距離に基づく報酬。

    Note:
        commandはロボットルート座標系での目標位置で、ここでは
        ボトルの重心と比較している。ボトルの重心は静置時でも机上面から
        90～106 mm 程度の高さにあるため、コマンドのz範囲がキューブ用の
        ままだと「机に埋め込む」ような到達不能な目標になる。
        commands cfg の pos_z 範囲も併せて見直すこと。
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)

    # コマンドはロボットルート座標系
    desired_pos_b = command[:, :3]

    # ロボットルート座標系 -> ワールド座標系
    desired_pos_w, _ = combine_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        desired_pos_b,
    )

    object_com_pos_w = obj.data.root_com_pos_w

    distance = torch.linalg.vector_norm(
        desired_pos_w - object_com_pos_w,
        dim=1,
    )

    clearance = _bottom_clearance(env, object_cfg, surface_height)

    lifted = (
        clearance > lift_height
    ).to(dtype=distance.dtype)

    return lifted * (1.0 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    lift_height: float,
    surface_height: float = 0.0,
    grasp_offset_z: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """接近報酬と持ち上げ判定を組み合わせる。"""
    reach_reward = object_ee_distance(
        env=env,
        std=std,
        grasp_offset_z=grasp_offset_z,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )

    lift_reward = object_is_lifted(
        env=env,
        lift_height=lift_height,
        surface_height=surface_height,
        object_cfg=object_cfg,
    )

    return reach_reward * lift_reward


def gripper_close_near_object(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    axial_std: float = 0.04,
    min_finger_pos: float = 0.005,
    miss_scale: float = 0.0,
    action_name: str = "gripper_action",
    grasp_offset_z: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["piper_finger_joint7", "piper_finger_joint8"]
    ),
) -> torch.Tensor:
    """ボトルを指の間に挟んだ状態でグリッパを閉じていることを報酬化する。

    戻り値は 0～1。次の3つの積で決まる。

    1. closing   : 前回アクションが close（BinaryJointPositionAction は負が close）
    2. proximity : 手先と狙い点の近さ。ボトル軸からの半径方向と軸方向を分けて評価する。
                   球状の距離だと、ボトルの横（指の外側）で閉じても半径 33 mm の
                   位置で exp(-(0.033/0.04)^2)≈0.5 と報酬の半分が出てしまうため、
                   半径方向は std をボトル半径より小さくして「指の間に軸が来ている」
                   ことを要求し、縦長の胴体に沿った軸方向は axial_std で緩めに許容する。
    3. held      : close 指令にもかかわらず両指とも閉じ切っていない
                   （= 何かを挟んで止まっている）。指を閉じ切ると関節位置は
                   close_command (±0.001) まで戻るので、両指の |位置| が
                   min_finger_pos を超えていれば物体で止められていると判定する。
                   片指だけ止まる「横から押しているだけ」の状態は min で弾く。
                   held でないときは miss_scale 倍（既定 0 = 空振りは無報酬）。

    制御が 2 Hz（1ステップ 0.5 s）で、指の速度上限 0.2 m/s なら 50 mm のストロークは
    同じステップ内で閉じ切るため、報酬計算時点で指位置は収束している前提でよい。

    Args:
        std: ボトル軸からの半径方向距離のスケール [m]。BOTTLE_RADIUS より小さくする。
        axial_std: ボトル軸方向の狙い点からのずれのスケール [m]。
        min_finger_pos: 「挟んでいる」と判定する各指関節の最小 |位置| [m]。
            close_command (0.001) より十分大きく、ボトル把持時の指位置より小さくする。
        miss_scale: held でない（空振り / 片指だけ当たっている）ときの報酬倍率。
            初期学習で閉じる動作自体の誘導がほしい場合だけ 0.1～0.3 程度にする。
        grasp_offset_z: object_ee_distance と同じ意味の狙い点オフセット。
            None なら重心を狙い点にする（軸方向はリンク姿勢のローカルZ）。
        gripper_cfg: 指関節。joint_names の順序は問わない（|位置| で評価する）。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene[gripper_cfg.name]

    # ボトルのローカルZ軸（回転行列の第3列）
    local_z_w = matrix_from_quat(
        obj.data.root_link_quat_w
    )[:, :, 2]

    if grasp_offset_z is None:
        target_pos_w = obj.data.root_com_pos_w
    else:
        target_pos_w = (
            obj.data.root_link_pos_w
            + grasp_offset_z * local_z_w
        )

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # 狙い点 -> 手先 をボトル軸方向と半径方向に分解
    rel = ee_pos_w - target_pos_w
    axial = (rel * local_z_w).sum(dim=-1)
    radial = torch.linalg.vector_norm(
        rel - axial.unsqueeze(-1) * local_z_w,
        dim=-1,
    )

    proximity = torch.exp(
        -torch.square(radial / std)
        - torch.square(axial / axial_std)
    )

    # BinaryJointPositionActionでは負の値がclose
    gripper_action = (
        env.action_manager
        .get_term(action_name)
        .raw_actions[:, 0]
    )

    closing = (gripper_action < 0.0).to(proximity.dtype)

    # 両指とも物体で止められているか
    finger_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids].abs()
    held = (
        finger_pos.min(dim=-1).values > min_finger_pos
    ).to(proximity.dtype)

    grasp_quality = held + miss_scale * (1.0 - held)

    return closing * proximity * grasp_quality *1.0e2


def gripper_closed_far_from_object(
    env: ManagerBasedRLEnv,
    far_distance: float,
    lift_height: float,
    std: float = 0.01,
    surface_height: float = 0.0,
    action_name: str = "gripper_action",
    grasp_offset_z: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """物体から遠い位置でグリッパを閉じる行動を罰する。

    戻り値は 0～1（1 = 遠いのに閉じる行動をとっている）。負の重みで使うこと。
    物体近傍（far_distance 以内）と持ち上げ後は 0 を返すので、
    gripper_close_near_object による「近くで閉じる」報酬とは干渉しない。

    開閉の判定は指関節の実位置ではなく前回アクションで行う。
    実位置だと開指令を出してから指が開き切るまでの数ステップも
    罰が残り、行動との対応が曖昧になるため。

    Args:
        far_distance: これより遠いと罰の対象になる距離 [m]。
            gripper_close_near_object が報酬を出す範囲（半径方向 std, 軸方向 axial_std）
            より外側に置くこと。
        lift_height: 持ち上げ判定の高さ [m]。object_is_lifted と揃える。
        std: far_distance 付近の遷移幅 [m]。0 ならステップ関数。
        action_name: グリッパの BinaryJointPositionAction の term 名。
        grasp_offset_z: 距離を測る狙い点のオフセット。object_ee_distance と同じ意味。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 狙い点（object_ee_distance と同じ定義）
    if grasp_offset_z is None:
        target_pos_w = obj.data.root_com_pos_w
    else:
        local_z_w = matrix_from_quat(
            obj.data.root_link_quat_w
        )[:, :, 2]
        target_pos_w = (
            obj.data.root_link_pos_w
            + grasp_offset_z * local_z_w
        )

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.linalg.vector_norm(
        target_pos_w - ee_pos_w,
        dim=1,
    )

    # BinaryJointPositionActionでは負の値がclose
    gripper_action = (
        env.action_manager
        .get_term(action_name)
        .raw_actions[:, 0]
    )

    closing = (gripper_action < 0.0).to(distance.dtype)

    # 遠さゲート
    if std > 0.0:
        far = torch.sigmoid(
            (distance - far_distance) / std
        )
    else:
        far = (distance > far_distance).to(distance.dtype)

    # 持ち上げ後は罰しない
    clearance = _bottom_clearance(env, object_cfg, surface_height)

    on_ground = (
        clearance < lift_height
    ).to(dtype=distance.dtype)

    return closing * far * on_ground


class object_displacement(ManagerTermBase):
    """reset 時のボトル位置からの水平変位 [m] を返す。負の重みで使うこと。

    2Hz 制御では報酬は 0.5 秒に1回しか評価されず、ボトルの速度はその瞬間に
    ほぼ 0 に戻っていることが多い。変位は状態量なので、押しずらした結果を
    評価タイミングに関係なく捉えられる。

    基準位置は reset() で記録する。RewardManager.reset は reset イベント
    (reset_root_state_uniform) の後に呼ばれ、write_root_pose_to_sim は
    data バッファも即時更新するので、ランダム化後の位置が取れる。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        obj: RigidObject = env.scene[self._object_cfg.name]
        self._init_pos_xy = obj.data.root_link_pos_w[:, :2].clone()

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        obj: RigidObject = self._env.scene[self._object_cfg.name]
        self._init_pos_xy[env_ids] = obj.data.root_link_pos_w[env_ids, :2]

    def displacement(self) -> torch.Tensor:
        obj: RigidObject = self._env.scene[self._object_cfg.name]
        return torch.linalg.vector_norm(
            obj.data.root_link_pos_w[:, :2] - self._init_pos_xy,
            dim=1,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        return self.displacement()


class approach_settled(object_displacement):
    """ボトルを乱さずに把持前の位置に収まっている状態への報酬。0～1。

    object_in_contact_course の値に「ボトルが直立している」「reset 位置から
    ほとんど動いていない」の2つの硬判定を掛ける。押し倒しながら・押しずらしながら
    接触コースに入っても報酬は出ない。

    Args:
        max_displacement: 許容する水平変位 [m]。
        max_tilt: 許容する傾き。object_tilt と同じ単位（真横 = 1）。0.1 ≈ 9°。
        その他: object_in_contact_course と同じ。approach 側の項と揃えること。
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_displacement: float = 0.01,
        max_tilt: float = 0.1,
        std_x: float = 0.10,
        max_x: float = 0.07,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=["link6"],
            joint_names=["piper_finger_joint7", "piper_finger_joint8"],
        ),
    ) -> torch.Tensor:
        course = object_in_contact_course(
            env, std_x=std_x, max_x=max_x,
            object_cfg=object_cfg, robot_cfg=robot_cfg,
        )
        undisturbed = (self.displacement() < max_displacement).to(course.dtype)
        upright = (object_tilt(env, object_cfg) < max_tilt).to(course.dtype)
        return course * undisturbed * upright
