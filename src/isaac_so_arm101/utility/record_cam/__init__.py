from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_camera_frame(
    env,
    step: int,
    env_id: int = 0,
    output_dir: str = "camera_debug",
    save_interval: int = 100,
    min_depth: float = 0.07,
    max_depth: float = 0.50,
):
    if step % save_interval != 0:
        return

    camera = env.unwrapped.scene["wrist_cam"]

    rgb = camera.data.output["rgb"][env_id]
    depth = camera.data.output["distance_to_image_plane"][env_id]

    # -------------------------
    # RGB
    # -------------------------
    rgb = rgb.detach().cpu().numpy()

    # float [0, 1] の場合
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)

    # RGBA の場合は RGB のみにする
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # -------------------------
    # Depth
    # -------------------------
    depth = depth.detach().cpu().numpy()

    if depth.ndim == 3:
        depth = depth[..., 0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 同じ step 番号で保存
    rgb_path = output_path / f"rgb_{step:08d}.png"
    depth_path = output_path / f"depth_{step:08d}.png"
    depth_raw_path = output_path / f"depth_{step:08d}.npy"

    # RGB 保存
    Image.fromarray(rgb).save(rgb_path)

    # 深度の生データ保存
    np.save(depth_raw_path, depth)

    # 深度可視化
    valid_depth = np.ma.masked_invalid(depth)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        valid_depth,
        vmin=min_depth,
        vmax=max_depth,
        interpolation="nearest",
    )
    plt.colorbar(label="Depth [m]")
    plt.title(
        f"distance_to_image_plane\n"
        f"step={step}, env={env_id}"
    )
    plt.tight_layout()
    plt.savefig(depth_path, dpi=150)
    plt.close()

    print(
        f"[Camera Debug] step={step} "
    )