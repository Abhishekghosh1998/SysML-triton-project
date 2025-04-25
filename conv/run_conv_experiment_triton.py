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


##############################################################################################
def get_autotune_config():
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
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
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


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'D', 'H', 'W', 'K', 'D_out', 'H_out', 'W_out', 'T', 'R', 'S', 'stride_d', 'stride_h', 'stride_w', 'pad_d', 'pad_h', 'pad_w', 'dila_d', 'dila_h', 'dila_w']
)
@triton.jit
def conv3d_kernel(x_ptr, w_ptr, y_ptr, N, C, D, H, W, K, D_out, H_out, W_out, T, R, S, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dila_d, dila_h, dila_w, 
                  GEMM_M, GEMM_N, GEMM_K,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (D_out * H_out * W_out)
    ndhw_residual = gemm_i % (D_out * H_out * W_out)
    d_out = ndhw_residual // (H_out * W_out)
    dhw_residual = ndhw_residual % (H_out * W_out)
    h_out = dhw_residual // W_out
    w_out = dhw_residual % W_out
    k = gemm_j

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) # % GEMM_K
        t = gemm_k // (R * S * C)
        trsc_residual = gemm_k % (R * S * C)
        r = trsc_residual // (S * C)
        rsc_residual = gemm_k % (S * C)
        s = rsc_residual // C
        c = rsc_residual % C
        d = d_out[:, None] * stride_d + t[None, :] * dila_d - pad_d
        h = h_out[:, None] * stride_h + r[None, :] * dila_h - pad_h
        w = w_out[:, None] * stride_w + s[None, :] * dila_w - pad_w
        mask_x = (d >= 0) & (d < D) & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_w = (t < T) & (r < R) & (s < S) & (c < C)
        offs_x = n[:, None] * D * H * W * C + d * H * W * C + h * W * C + w * C + c
        offs_w = k[None, :] * T * R * S * C + t[:, None] * R * S * C + r[:, None] * S * C + s[:, None] * C + c[:, None]

        x_ptrs = x_ptr + offs_x
        w_ptrs = w_ptr + offs_w

        x_data = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_data = tl.load(w_ptrs, mask=mask_w[:, None], other=0.0)
        accumulator = tl.dot(x_data, w_data, accumulator)
    c_data = accumulator.to(tl.float16)

    offs_y = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_y = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    y_ptrs = y_ptr + offs_y
    tl.store(y_ptrs, c_data, mask=mask_y)


def triton_conv3d(start_event, end_event, stream, x: torch.Tensor, w: torch.Tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    N, D, H, W, C = x.shape
    K, T, R, S, C = w.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dila_d, dila_h, dila_w = dilation
    D_out = (D + 2 * pad_d - dila_d * (T - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dila_h * (R - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dila_w * (S - 1) - 1) // stride_w + 1
    y = torch.empty((N, D_out, H_out, W_out, K), device=x.device, dtype=torch.float16)
    GEMM_M = N * D_out * H_out * W_out
    GEMM_N = K
    GEMM_K = T * R * S * C

    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )

    start_event.record(stream)
    conv3d_kernel[grid](x, w, y, N, C, D, H, W, K, D_out, H_out, W_out, T, R, S, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dila_d, dila_h, dila_w, GEMM_M, GEMM_N, GEMM_K)
    end_event.record(stream)
    return y

def run_triton_conv3d(x, w, num_runs, stream):

    x_channel_last = x.permute(0, 2, 3, 4, 1).contiguous()
    w_channel_last = w.permute(0, 2, 3, 4, 1).contiguous()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.cuda.stream(stream):
        for i in range(num_runs):
            triton_conv3d(start_events[i], end_events[i], stream, x_channel_last, w_channel_last, stride=(stride_d, stride_h, stride_w), padding=(pad_d, pad_h, pad_w), dilation=(dila_d, dila_h, dila_w))
            
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
        print(f"[Conv Triton] Run {i}, Latency: {lats[i]} ms")
    print(f"AVG MED time is {avg_lat} ms")

    return avg_lat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters_interf",
        type=int,
        default=300000,
        help="Number of iterations for the interference kernel",
    )
    parser.add_argument(
        "--runs_conv", type=int, default=100, help="Number of runs for the conv kernel"
    )
    parser.add_argument(
        "--runs_interf",
        type=int,
        default=4,
        help="Number of runs for the interference kernel",
    )

    args = parser.parse_args()

    stream = torch.cuda.Stream()

    # N = 1; C = 64; D = 56; H = 56; W = 56; K = 128; T = 3; R = 3; S = 3; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    # N = 1; C = 16; D = 6; H = 6; W = 6; K = 16; T = 1; R = 1; S = 1; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    N = 1; C = 64; D = 56; H = 56; W = 56; K = 64; T = 1; R = 1; S = 1; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1

    x = torch.randn(N, C, D, H, W).cuda().half()
    w = torch.randn(K, C, T, R, S).cuda().half()
    # run the conv kernel once, to preload it into memory
    # otherwise risk of lazy loading leading to sequential kernel execution (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#possible-issues-when-adopting-lazy-loading)
    run_triton_conv3d(x, w, 1, stream)
    torch.cuda.synchronize()
    
    num_tb = 132
    num_threads = 128
    launch_config = f"fp32_fma launch Config: ({num_tb}, {num_threads})"

    # warmup
    run_fp32_fma_kernel(num_tb, num_threads, args.iters_interf, 1)

    print("------------------")
    print("Running conv Triton alone")
    run_triton_conv3d(x, w, args.runs_conv, stream)

    print("------------------")
    print(f"Running fp32_fma kernel alone - {launch_config}")
    run_fp32_fma_kernel(
        num_tb, num_threads, args.iters_interf, args.runs_interf
    )

    print("------------------")
    print(f"Running conv Triton and fp32_fma kernel collocated - {launch_config}")
    interf_thread = threading.Thread(
        target=run_fp32_fma_kernel,
        args=(num_tb, num_threads, args.iters_interf, args.runs_interf),
    )
    
    triton_conv_thread = threading.Thread(target=run_triton_conv3d, args=(x, w, args.runs_conv, stream))

    interf_thread.start()
    triton_conv_thread.start()

    triton_conv_thread.join()
    interf_thread.join()

    print("Done!")
