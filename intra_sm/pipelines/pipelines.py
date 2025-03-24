import torch
import triton
import triton.language as tl
import time

# ---------------------- COMPUTE KERNEL (FP64 ILP1) ----------------------
@triton.jit
def compute_kernel_ilp1(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP64 for ILP-1)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float64)
   
    # Initialize ILP-1 variables
    op1 = a
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float64)

    # Compute loop with ILP-1
    for _ in range(num_iters):
        op3 = op1 * op3

    # Store the result
    c = op3
    tl.store(c_ptr + block_start, c, mask=mask)

# ---------------------- COMPUTE KERNEL (FP64 ILP2) ----------------------
@triton.jit
def compute_kernel_ilp2(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP64 for ILP-2)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float64)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float64)

    # Initialize ILP-2 variables
    op1, op2 = a, b
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float64)
    op4 = tl.full(block_start.shape, 1.0, dtype=tl.float64)

    # Compute loop with ILP-2
    for _ in range(num_iters):
        op3 = op1 * op3
        op4 = op2 * op4

    # Store the result
    c = op3 + op4
    tl.store(c_ptr + block_start, c, mask=mask)

# ---------------------- COMPUTE KERNEL (FP64 ILP3) ----------------------
@triton.jit
def compute_kernel_ilp3(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP64 for ILP-3)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float64)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float64)

    # Initialize ILP-3 variables
    op1, op2 = a, b
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float64)
    op4 = tl.full(block_start.shape, 1.0, dtype=tl.float64)
    op5 = tl.full(block_start.shape, 1.0, dtype=tl.float64)

    # Compute loop with ILP-3
    for _ in range(num_iters):
        op3 = op1 * op3
        op4 = op2 * op4
        op5 = op1 * op5

    # Store the result
    c = op3 + op4 + op5
    tl.store(c_ptr + block_start, c, mask=mask)

# ---------------------- COMPUTE KERNEL (FP64 ILP4) ----------------------

@triton.jit
def compute_kernel_ilp4(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP32 for ILP-4)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float64)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float64)

    # Initialize ILP-4 variables
    op1, op2 = a, b
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float64)  
    op4 = tl.full(block_start.shape, 1.0, dtype=tl.float64)
    op5 = tl.full(block_start.shape, 1.0, dtype=tl.float64)
    op6 = tl.full(block_start.shape, 1.0, dtype=tl.float64)

    # Compute loop with ILP-4
    for _ in range(num_iters):
        op3 = op1 * op3
        op4 = op2 * op4
        op5 = op1 * op5
        op6 = op2 * op6

    # Store the result
    c = op3 + op4 + op5 + op6
    tl.store(c_ptr + block_start, c, mask=mask)

def get_compute_kernel(ilp):
    if ilp == 1:
        return compute_kernel_ilp1
    elif ilp == 2:
        return compute_kernel_ilp2
    elif ilp == 3:
        return compute_kernel_ilp3
    elif ilp == 4:
        return compute_kernel_ilp4
    else:
        raise ValueError(f"Invalid ILP value: {ilp}")   


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

def main(mode, ilp_level, num_threads_per_tb, num_iters):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # display the number of SMs
    print(f"Number of SMs: {num_sms}")
    BLOCK_SIZE = num_threads_per_tb
    
    # Allocate memory for compute kernel
    num_comp_elems = num_threads_per_tb
    a1, b1, c1 = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float64) for _ in range(3)]
    a2, b2, c2 = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float64) for _ in range(3)]

    # number of warps 
    num_warps = num_threads_per_tb // 32
    
    # Define kernel arguments
    args_compute_1 = {"a_ptr": a1, "b_ptr": b1, "c_ptr": c1, "num_iters": num_iters}
    args_compute_2 = {"a_ptr": a2, "b_ptr": b2, "c_ptr": c2, "num_iters": num_iters}

    compute_kernel = get_compute_kernel(ilp_level)
    # Define grid
    grid = (num_sms,)

    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        print(f"\n[Running Compute Kernel Alone with ILP {ilp_level}]")
        time_compute = run_kernel(compute_kernel, args_compute_1, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        print(f"Compute Kernel Latency: {time_compute:.6f} sec")

    elif mode == 2:
        print(f"\n[Running two compute kernel with ILP {ilp_level} sequentially]")
        num_runs = 10
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            # Record the start event
            start_event.record()
            compute_kernel[grid](**args_compute_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            compute_kernel[grid](**args_compute_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            # Record the end event
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        avg_time = sum(times) / (num_runs * 1000)
        print(f"Sequential Execution Latency: {avg_time:.6f} sec")

    elif mode == 3:
        print(f"\n[Running two compute kernel with ILP {ilp_level} colocated]")
        num_runs = 10
        # Create CUDA events
        start1 = torch.cuda.Event(enable_timing=True)
        stop1 = torch.cuda.Event(enable_timing=True)
        start2 = torch.cuda.Event(enable_timing=True)
        stop2 = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            with torch.cuda.stream(stream1):
                start1.record(stream1)
                compute_kernel[grid](**args_compute_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                stop1.record(stream1)
            
            with torch.cuda.stream(stream2):
                start2.record(stream2)
                compute_kernel[grid](**args_compute_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                stop2.record(stream2)

            torch.cuda.synchronize()
            time1 = start1.elapsed_time(stop1)
            time2 = start2.elapsed_time(stop2)
            makespan1 = start1.elapsed_time(stop2)
            makespan2 = start2.elapsed_time(stop1)
            # get max of time1, time2, makespan1, makespan2
            span = max(time1, time2, makespan1, makespan2)
            times.append(span)
        
        avg_time = sum(times) / (num_runs * 1000)
        print(f"Concurrent Execution Latency: {avg_time:.6f} sec")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python script.py <mode> <ilp_level> <num_threads_per_tb> <num_iters>")
        sys.exit(1)

    mode = int(sys.argv[1])
    ilp_level = int(sys.argv[2])
    num_threads_per_tb = int(sys.argv[3])
    num_iters = int(sys.argv[4])

    main(mode, ilp_level, num_threads_per_tb, num_iters)
