uv run play --task Piper-Lift-Bottle-With-Camera-phase1  \
    --num_envs $1 \
    --enable_camera \
    --checkpoint /workspace/isaac_so_arm101/logs/rsl_rl/lift_bottle_with_camera/2026-10-15_08-00-01/model_$2.pt
