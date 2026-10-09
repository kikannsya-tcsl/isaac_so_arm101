import sys
import time
import torch

print("Number of CUDA devices:", torch.cuda.device_count())
tensor =  torch.randn(50000, 10000, device='cpu')
tensor_size = tensor.element_size() * tensor.nelement() / 10**9   # Size in GB
num_runs = 10
print(f"Tensor size: {tensor_size:.2f} GB, averaged over {num_runs} runs")
avg_time = []
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"==== Tensor size: {tensor_size:.2f} GB, CUDA -> CPU -> CUDA transfer speed ====")
    total_time = 0.0
    for j in range(num_runs):
        tensor = torch.randn(50000, 10000, device=f'cuda:{i}')
        start_time = time.perf_counter()
        tensor_cpu = tensor.cpu()
        tensor_gpu = tensor_cpu.to(f'cuda:{i}')
        end_time = time.perf_counter()
        total_time += end_time - start_time
    print(f"Average transfer time: {total_time / num_runs:.6f} seconds")
    avg_time.append(total_time / num_runs)

