# uv run train --task Isaac-Piper-Lift-Bottle-With-Camera-v1 \
# --headless \
# --num_envs 128 \
# --max_iterations 2000 \
# --enable_camera \
# --resume \
# --load_run 2026-09-04_02-52-09 \
# --checkpoint model_300.pt 

num=${1:-2000}

uv run train --task Isaac-Piper-Lift-Bottle-With-Camera-v1 \
--headless \
--num_envs 200 \
--max_iterations $num \
--enable_camera \
