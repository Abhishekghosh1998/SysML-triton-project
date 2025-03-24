import torch
import triton
import triton.language as tl
import time

# ✅ Low IPC Triton Kernel
# @triton.jit
# def low_ipc_kernel(dummy_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     lane = tl.arange(0, BLOCK_SIZE)

#     # Initialize values
#     acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
#     val = tl.load(dummy_ptr + lane)

#     # Fake workload to reduce IPC
#     for _ in range(num_iters):
#         acc = (acc + val) * 1.00001  # Redundant computation
#         acc = tl.math.log(acc + 1.0)  # Serial dependency to slow execution
#         acc = tl.math.exp(acc) - 1.0  # More dependencies to prevent parallelism

#     # Store result (unnecessary but forces memory access latency)
#     tl.store(dummy_ptr + lane, acc)

### IPC 0.33    
@triton.jit
def low_ipc_kernel(dummy_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    # Force dependencies by initializing values
    acc = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)  
    val = tl.load(dummy_ptr + lane)
    
    # Fake workload with forced serial dependency
    for _ in range(num_iters):
        acc = acc + val  # Forces dependency
        acc = acc * 1.00001
        acc = acc / 1.00001
        acc = acc - val  # Maintains dependency

        # Introduce bitwise integer ops to confuse FP pipeline
        acc_int = acc.to(tl.int32)
        acc_int = (acc_int << 3) | (acc_int >> 2)  # Unaligned shifts
        acc = acc_int.to(tl.float32)

        # Force memory stalls
        tl.store(dummy_ptr + lane, acc)  # Unnecessary store
        acc = tl.load(dummy_ptr + lane)  # Unnecessary reload

    # Store final result (ensures execution)
    tl.store(dummy_ptr + lane, acc)


num_tb=132 # TODO: set to number of SMs present on GPU
num_threads_per_tb=1024 # TODO: set to max_threads_per_sm / 2
num_itrs=100000

# ✅ Set Parameters
BLOCK_SIZE = num_threads_per_tb
NUM_ITERS = num_itrs  # Large iteration count to reduce IPC

num_warps = num_threads_per_tb // 32

# ✅ Allocate Memory
dummy = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")

# ✅ Synchronize CUDA Before Timing
torch.cuda.synchronize()

# ✅ Measure Kernel Execution Time
start_time = time.time()
for _ in range(2):
    low_ipc_kernel[(num_tb,)](dummy, num_iters=NUM_ITERS, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
torch.cuda.synchronize()  # Ensure kernel execution finishes
end_time = time.time()

print(f"Kernel execution time: {end_time - start_time:.6f} seconds")
