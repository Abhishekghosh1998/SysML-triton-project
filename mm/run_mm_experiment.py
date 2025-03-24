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

def alloc_mats(dim1, dim2, dim3):
    mat1 = torch.randn(dim1, dim2, pin_memory=True).cuda(non_blocking=True)
    mat2 = torch.randn(dim2, dim3, pin_memory=True).cuda(non_blocking=True)
    return mat1, mat2


def run_mm(m11, m12, num_runs, stream):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.cuda.stream(stream):
        for i in range(num_runs):
            start_events[i].record()
            torch.mm(m11, m12)
            end_events[i].record()

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
        print(f"[MM PyTorch] Run {i}, Latency: {lats[i]} ms")
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
    run_mm(m11, m12, 1, stream)
    torch.cuda.synchronize()
    
    num_tb = 132
    num_threads = 128
    launch_config = f"fp32_fma launch Config: ({num_tb}, {num_threads})"

    # warmup
    run_fp32_fma_kernel(num_tb, num_threads, args.iters_interf, 1)

    print("------------------")
    print("Running MM PyTorch alone")
    run_mm(m11, m12, args.runs_mm, stream)

    print("------------------")
    print(f"Running fp32_fma kernel alone - {launch_config}")
    run_fp32_fma_kernel(
        num_tb, num_threads, args.iters_interf, args.runs_interf
    )

    print("------------------")
    print(f"Running MM PyTorch and fp32_fma kernel collocated - {launch_config}")
    interf_thread = threading.Thread(
        target=run_fp32_fma_kernel,
        args=(num_tb, num_threads, args.iters_interf, args.runs_interf),
    )
    mm_thread = threading.Thread(target=run_mm, args=(m11, m12, args.runs_mm, stream))

    interf_thread.start()
    mm_thread.start()

    mm_thread.join()
    interf_thread.join()

    print("Done!")
