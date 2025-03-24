import torch
import triton
import triton.language as tl
import time

# ---------------------- COMPUTE KERNEL (FP64 ILP4) ----------------------

@triton.jit
def compute_kernel(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP32 for ILP-4)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float32)

    # Initialize ILP-4 variables
    op1, op2 = a, b
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float32)  
    op4 = tl.full(block_start.shape, 1.0, dtype=tl.float32)
    op5 = tl.full(block_start.shape, 1.0, dtype=tl.float32)
    op6 = tl.full(block_start.shape, 1.0, dtype=tl.float32)

    # Compute loop with ILP-4
    for _ in range(num_iters):
        op3 = op1 * op3
        op4 = op2 * op4
        op5 = op1 * op5
        op6 = op2 * op6

    # Store the result
    c = op3 + op4 + op5 + op6
    tl.store(c_ptr + block_start, c, mask=mask)


# ---------------------- DRIVER FUNCTION ----------------------
def run_kernel(kernel, args, grid, block_size, num_warps, num_runs=10, stream=None):
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.cuda.stream(stream):
            kernel[grid](**args, BLOCK_SIZE=block_size, num_warps=num_warps)

    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_runs

def main(mode, num_threads_per_tb, num_iters_comp):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    BLOCK_SIZE = num_threads_per_tb

    # Allocate memory for compute kernel 1
    num_comp_elems = num_threads_per_tb
    a1, b1, c1 = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float64) for _ in range(3)]

    # Allocate memory for compute kernel 2
    a2, b2, c2 = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float64) for _ in range(3)]


    # number of warps 
    num_warps = num_threads_per_tb // 32
    # Define kernel arguments
    args_compute_1 = {"a_ptr": a1, "b_ptr": b1, "c_ptr": c1, "num_iters": num_iters_comp}
    args_compute_2 = {"a_ptr": a2, "b_ptr": b2, "c_ptr": c2, "num_iters": num_iters_comp}
    
    # Define grid
    grid = (num_sms,)

    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        print("\n[Running Compute Kernel Alone]")
        time_compute = run_kernel(compute_kernel, args_compute_1, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        print(f"Compute Kernel Latency: {time_compute:.6f} sec")

    elif mode == 2:
        print("\n[Running two compute kernel sequentially]")
        num_runs = 10
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_runs):
            compute_kernel[grid](**args_compute_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            compute_kernel[grid](**args_compute_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        print(f"Sequential Execution Latency: {avg_time:.6f} sec")

    elif mode == 3:
        print("\n[Running two compute kernel colocated using CUDA streams]")
        num_runs = 10
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_runs):
            with torch.cuda.stream(stream1):
                compute_kernel[grid](**args_compute_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            with torch.cuda.stream(stream2):
                compute_kernel[grid](**args_compute_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        print(f"Concurrent Execution Latency: {avg_time:.6f} sec")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python script.py <mode> <num_threads_per_tb> <num_iters_comp>")
        sys.exit(1)

    mode = int(sys.argv[1])
    num_threads_per_tb = int(sys.argv[2])
    num_iters_comp = int(sys.argv[3])

    main(mode, num_threads_per_tb, num_iters_comp)
