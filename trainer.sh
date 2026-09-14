# uv run train --task Isaac-Piper-Lift-Bottle-With-Camera-v1 \
# --headless \
# --num_envs 128 \
# --enable_camera \
# --resume \
# --load_run 2026-09-04_02-52-09 \
# --checkpoint model_300.pt 

uv run train --task Isaac-Piper-Lift-Bottle-With-Camera-v1 \
--headless \
--num_envs 128 \
--enable_camera \
--resume \
--load_run 2026-09-09_01-04-56 \
--checkpoint model_10400.pt
