GPU_ID=1 docker compose -p gpu1_trainer up -d

# Scaled version
# GPU_ID=1 docker compose -p gpu1_trainer up -d --scale piper_trainer=2