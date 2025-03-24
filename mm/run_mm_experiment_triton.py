import argparse
import torch
import threading

from ctypes import *

import triton
import triton.language as tl

@triton.jit
def fma_kernel_ilp4(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP32 for ILP-4)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float32)

    # Initialize ILP-4 variables
    op1, op2 = a, b
    op3 = tl.zeros(block_start.shape, dtype=tl.float32)  
    op4 = tl.zeros(block_start.shape, dtype=tl.float32)
    op5 = tl.zeros(block_start.shape, dtype=tl.float32)
    op6 = tl.zeros(block_start.shape, dtype=tl.float32)

    # Compute loop with ILP-4 using FMA
    for _ in range(num_iters):
        op3 = tl.math.fma(op1, op2, op3)  # op3 = (op1 * op2) + op3
        op4 = tl.math.fma(op1, op2, op4)
        op5 = tl.math.fma(op1, op2, op5)
        op6 = tl.math.fma(op1, op2, op6)

    # Store the result
    c = op3 + op4 + op5 + op6
    tl.store(c_ptr + block_start, c, mask=mask)

def run_fp32_fma_kernel(num_tb, num_threads_per_tb, num_itrs, num_runs):
    
    BLOCK_SIZE = num_threads_per_tb
    # Allocate memory
    num_elems = num_threads_per_tb
    a, b, c = [torch.ones(num_elems, device="cuda", dtype=torch.float32) for _ in range(3)]

    # Define the grid
    grid = (num_tb,)

    # Define the number of warps
    num_warps = num_threads_per_tb // 32

    # Define the kernel arguments
    args_compute = { "a_ptr": a, "b_ptr": b, "c_ptr": c, "num_iters": num_itrs}

    # Create CUDA streams
    stream = torch.cuda.Stream()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        with torch.cuda.stream(stream):
            # Record the start event
            start_event.record(stream)
            fma_kernel_ilp4[grid](**args_compute, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            # Record the end event
            end_event.record(stream)
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        print(f"[FP32] Run {_}: Latency: {times[-1]} ms")

    avg_time = sum(times) / (num_runs * 1000)
    print(f"fma_kernel_ilp4 Execution Latency: {avg_time:.6f} sec")

##########################################################################################

def is_cuda():
    return True


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_autotune_config():
    return get_cuda_autotune_config()
    

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float32)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, start_event, end_event, stream):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    start_event.record(stream)
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    end_event.record(stream)
    
    return c

##########################################################################################

def alloc_mats(dim1, dim2, dim3):
    mat1 = torch.randn(dim1, dim2, pin_memory=True).cuda(non_blocking=True)
    mat2 = torch.randn(dim2, dim3, pin_memory=True).cuda(non_blocking=True)
    return mat1, mat2


# def run_mm(m11, m12, num_runs, stream):
#     start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
#     end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

#     with torch.cuda.stream(stream):
#         for i in range(num_runs):
#             start_events[i].record()
#             torch.mm(m11, m12)
#             end_events[i].record()

#     # need to synchronize at the very end to make sure, all operations end up in nsys trace
#     end_events[-1].synchronize()

#     lats = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
#     sort_lats = sorted(lats)

#     if num_runs > 1:
#         mid = num_runs // 2
#         avg_lat = (sort_lats[mid] + sort_lats[mid + 1]) / 2
#     else:
#         avg_lat = sort_lats[0]

#     for i in range(num_runs):
#         print(f"[MM PyTorch] Run {i}, Latency: {lats[i]} ms")
#     print(f"AVG MED time is {avg_lat} ms")

#     return avg_lat

def run_triton_matmul(m11, m12, num_runs, stream):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.cuda.stream(stream):
        for i in range(num_runs):
            # start_events[i].record()
            matmul(m11, m12, start_events[i], end_events[i], stream)
            # end_events[i].record()

    # need to synchronize at the very end to make sure, all operations end up in nsys trace
    end_events[-1].synchronize()

    lats = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
    sort_lats = sorted(lats)

    if num_runs > 1:
        mid = num_runs // 2
        avg_lat = (sort_lats[mid] + sort_lats[mid + 1]) / 2
    else:
        avg_lat = sort_lats[0]

    for i in range(num_runs):
        print(f"[MM Triton] Run {i}, Latency: {lats[i]} ms")
    print(f"AVG MED time is {avg_lat} ms")

    return avg_lat



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024, help="Matrix A rows")
    parser.add_argument(
        "--k", type=int, default=1024, help="Matrix A cols / Matrix B rows"
    )
    parser.add_argument("--n", type=int, default=1024, help="Matrix B cols")
    parser.add_argument(
        "--iters_interf",
        type=int,
        default=300000,
        help="Number of iterations for the interference kernel",
    )
    parser.add_argument(
        "--runs_mm", type=int, default=100, help="Number of runs for the mm kernel"
    )
    parser.add_argument(
        "--runs_interf",
        type=int,
        default=4,
        help="Number of runs for the interference kernel",
    )

    args = parser.parse_args()

    stream = torch.cuda.Stream()

    # run the mm kernel once, to preload it into memory
    # otherwise risk of lazy loading leading to sequential kernel execution (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#possible-issues-when-adopting-lazy-loading)
    m11, m12 = alloc_mats(args.m, args.k, args.n)
    # run_mm(m11, m12, 1, stream)
    run_triton_matmul(m11, m12, 1, stream)
    torch.cuda.synchronize()
    
    num_tb = 132
    num_threads = 128
    launch_config = f"fp32_fma launch Config: ({num_tb}, {num_threads})"

    # warmup
    run_fp32_fma_kernel(num_tb, num_threads, args.iters_interf, 1)

    # print("------------------")
    # print("Running MM PyTorch alone")
    # run_mm(m11, m12, args.runs_mm, stream)

    print("------------------")
    print("Running MM Triton alone")
    run_triton_matmul(m11, m12, args.runs_mm, stream)

    print("------------------")
    print(f"Running fp32_fma kernel alone - {launch_config}")
    run_fp32_fma_kernel(
        num_tb, num_threads, args.iters_interf, args.runs_interf
    )

    print("------------------")
    # print(f"Running MM PyTorch and fp32_fma kernel collocated - {launch_config}")
    print(f"Running MM Triton and fp32_fma kernel collocated - {launch_config}")
    interf_thread = threading.Thread(
        target=run_fp32_fma_kernel,
        args=(num_tb, num_threads, args.iters_interf, args.runs_interf),
    )
    # mm_thread = threading.Thread(target=run_mm, args=(m11, m12, args.runs_mm, stream))

    triton_mm_thread = threading.Thread(target=run_triton_matmul, args=(m11, m12, args.runs_mm, stream))

    interf_thread.start()
    # mm_thread.start()
    triton_mm_thread.start()

    triton_mm_thread.join()
    # mm_thread.join()
    interf_thread.join()

    print("Done!")
