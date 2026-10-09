import statistics
import torch

NUMEL = 64 * 1024 * 1024  # 256 MiB (float32)
NUM_RUNS = 20
WARMUP = 5


def measure_transfer(op, size_bytes):
    for _ in range(WARMUP):
        op()
    torch.cuda.synchronize()

    times = []
    for _ in range(NUM_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        op()
        end.record()
        end.synchronize()

        times.append(start.elapsed_time(end) / 1000)

    elapsed = statistics.median(times)
    return size_bytes / elapsed / 1e9


print("Number of CUDA devices:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        device = f"cuda:{i}"

        cpu_src = torch.empty(NUMEL, pin_memory=True)
        cpu_dst = torch.empty(NUMEL, pin_memory=True)
        gpu = torch.empty(NUMEL, device=device)

        cpu_src.fill_(1.0)
        gpu.fill_(1.0)

        size_bytes = gpu.numel() * gpu.element_size()

        h2d = measure_transfer(
            lambda: gpu.copy_(cpu_src, non_blocking=True),
            size_bytes,
        )

        d2h = measure_transfer(
            lambda: cpu_dst.copy_(gpu, non_blocking=True),
            size_bytes,
        )

        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        print(f"CPU -> GPU: {h2d:.2f} GB/s")
        print(f"GPU -> CPU: {d2h:.2f} GB/s")