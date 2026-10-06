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
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ee_pose_in_robot_root_frame(
    env,
    only_quaternion: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE pose expressed in the robot root frame.

    Returns:
        (num_envs, 7):
        [x, y, z, qw, qx, qy, qz]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # robot root pose in world
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    # EE pose in world
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids][:, 0]  # 1 body_id のみを想定
    ee_quat_w = robot.data.body_quat_w[:, ee_cfg.body_ids][:, 0]  # 1 body_id のみを想定

    # world -> robot root
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )

    return torch.cat((ee_pos_b, ee_quat_b), dim=-1)

def object_pose_in_robot_root_frame(
    env,
    only_quaternion: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object pose expressed in the robot root frame.

    Returns:
        (num_envs, 7):
        [x, y, z, qw, qx, qy, qz]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    obj_pos_b, obj_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        obj_pos_w,
        obj_quat_w,
    )

    if only_quaternion:
        return obj_quat_b
    return torch.cat((obj_pos_b, obj_quat_b), dim=-1)

def object_ee_relative_position(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object position relative to EE, expressed in robot root frame.

    Returns:
        (num_envs, 3)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids][:, 0]  # 1 body_id のみを想定
    obj_pos_w = obj.data.root_pos_w

    # Both positions -> robot root frame
    ee_pos_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
    )

    obj_pos_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        obj_pos_w,
    )

    return obj_pos_b - ee_pos_b

# (N,H,W,C) にブロードキャストする形で保持
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)


class SpatialSoftmax(torch.nn.Module):
    """(N,C,H,W) -> (N,2C) 各チャネルの注目位置(x,y)"""

    def __init__(self, h: int, w: int, temperature: float = 1.0):
        super().__init__()
        px, py = torch.meshgrid(
            torch.linspace(-1.0, 1.0, w), torch.linspace(-1.0, 1.0, h), indexing="xy"
        )
        self.register_buffer("px", px.reshape(1, 1, -1))
        self.register_buffer("py", py.reshape(1, 1, -1))
        self.t = temperature

    def forward(self, f):
        n, c, h, w = f.shape
        a = torch.softmax(f.reshape(n, c, h * w) / self.t, dim=-1)
        return torch.cat([(a * self.px).sum(-1), (a * self.py).sum(-1)], dim=-1)

class rgbd_features(ManagerTermBase):
    """RGB-D を凍結 ResNet18(4ch入力) で 512 次元特徴に変換する観測項。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # --- センサ設定の事前検証：実行時KeyErrorを起動時に前倒しする ---
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        data_types = set(env.scene.sensors[sensor_cfg.name].cfg.data_types)
        missing = {"rgb", "distance_to_image_plane"} - data_types
        if missing:
            raise ValueError(
                f"[rgbd_features] sensor '{sensor_cfg.name}' の data_types に "
                f"{sorted(missing)} が不足しています（現在: {sorted(data_types)}）"
            )

        import torchvision

        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        w = m.conv1.weight.data                                    # (64,3,7,7)
        conv = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        conv.weight.data[:, :3] = w
        conv.weight.data[:, 3:] = w.mean(dim=1, keepdim=True)      # depth ch は RGB 平均で初期化
        m.conv1, m.fc = conv, torch.nn.Identity()
        
        self.model = m.to(env.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = _IMAGENET_MEAN.to(env.device)
        self._std = _IMAGENET_STD.to(env.device)

        self._fp0 = self._fingerprint()
        self._step = 0

    # ------------------------------------------------------------------ #
    def _fingerprint(self) -> torch.Tensor:
        parts = [p.detach().flatten().float() for p in self.model.parameters()]
        parts += [b.detach().flatten().float() for b in self.model.buffers()]  # BN統計量が肝
        v = torch.cat(parts)
        return torch.stack([v.sum(), v.abs().sum(), v.pow(2).sum()])

    def verify_frozen(self, raise_on_fail: bool = True) -> dict:
        rep = {
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_training_mode": self.model.training,
            "weights_or_bn_stats_changed": not torch.equal(self._fingerprint(), self._fp0),
        }
        if raise_on_fail and any(rep.values()):
            raise RuntimeError(f"[rgbd_features] encoder is NOT frozen: {rep}")
        return rep

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        max_range: float = 1.5,
        verify_every: int = 0,          # params に入れるなら必ずここにも並べる
    ) -> torch.Tensor:
        self.model.eval()

        cam: TiledCamera = env.scene[sensor_cfg.name]
        rgb = cam.data.output["rgb"][..., :3].float() / 255.0      # 版により4ch返るので :3 で保険
        rgb = (rgb - self._mean) / self._std
        d = torch.nan_to_num(cam.data.output["distance_to_image_plane"], posinf=max_range)
        d = d.clamp(0.0, max_range) / max_range
        x = torch.cat([rgb, d], dim=-1).permute(0, 3, 1, 2).contiguous()   # (N,4,H,W)

        with torch.no_grad():
            feat = self.model(x)

        self._step += 1
        if verify_every and self._step % verify_every == 0:
            self.verify_frozen()
        return feat.detach()                                        # (N,512)