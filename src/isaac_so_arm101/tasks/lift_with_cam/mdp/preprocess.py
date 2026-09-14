import torch

from isaaclab.envs import mdp


def wrist_depth_observation(
    env,
    sensor_cfg,
    min_depth: float = 0.05,
    max_depth: float = 1.5,
) -> torch.Tensor:
    depth = mdp.image(
        env,
        sensor_cfg=sensor_cfg,
        data_type="distance_to_image_plane",
        normalize=False,
    )

    # NaNやinfを最大距離として処理
    depth = torch.nan_to_num(
        depth,
        nan=max_depth,
        posinf=max_depth,
        neginf=min_depth,
    )

    # 有効範囲へクリップ
    depth = depth.clamp(min_depth, max_depth)

    # メートル値を0～1へ正規化
    depth = (depth - min_depth) / (max_depth - min_depth)

    # NHWC → NCHW
    depth = depth.permute(0, 3, 1, 2).contiguous()

    return depth