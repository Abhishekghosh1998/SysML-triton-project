import torch
import triton
import triton.language as tl
import time

# ---------------------- COMPUTE KERNEL (FP32 ILP1) ----------------------
@triton.jit
def compute_kernel_ilp1(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP32 for ILP-1)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float32)
   
    # Initialize ILP-1 variables
    op1 = a
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float32)

    # Compute loop with ILP-1
    for _ in range(num_iters):
        op3 = op1 * op3

    # Store the result
    c = op3
    tl.store(c_ptr + block_start, c, mask=mask)

# ---------------------- COMPUTE KERNEL (FP32 ILP2) ----------------------
@triton.jit
def compute_kernel_ilp2(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP32 for ILP-2)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float32)

    # Initialize ILP-2 variables
    op1, op2 = a, b
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float32)
    op4 = tl.full(block_start.shape, 1.0, dtype=tl.float32)

    # Compute loop with ILP-2
    for _ in range(num_iters):
        op3 = op1 * op3
        op4 = op2 * op4

    # Store the result
    c = op3 + op4
    tl.store(c_ptr + block_start, c, mask=mask)

# ---------------------- COMPUTE KERNEL (FP32 ILP3) ----------------------
@triton.jit
def compute_kernel_ilp3(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < num_iters

    # Load inputs (FP32 for ILP-3)
    a = tl.load(a_ptr + block_start, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + block_start, mask=mask).to(tl.float32)

    # Initialize ILP-3 variables
    op1, op2 = a, b
    op3 = tl.full(block_start.shape, 1.0, dtype=tl.float32)
    op4 = tl.full(block_start.shape, 1.0, dtype=tl.float32)
    op5 = tl.full(block_start.shape, 1.0, dtype=tl.float32)

    # Compute loop with ILP-3
    for _ in range(num_iters):
        op3 = op1 * op3
        op4 = op2 * op4
        op5 = op1 * op5

    # Store the result
    c = op3 + op4 + op5
    tl.store(c_ptr + block_start, c, mask=mask)

# ---------------------- COMPUTE KERNEL (FP32 ILP4) ----------------------

@triton.jit
def compute_kernel_ilp4(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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


# ---------------------- MEMORY COPY KERNEL --------------------------------
@triton.jit
def memory_kernel(in_ptr, out_ptr, num_floats: tl.constexpr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  
    step = BLOCK_SIZE * tl.num_programs(0)  # The step size to move forward
    
    for _ in range(num_iters):
        block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        # Iterate over memory in steps until all elements are processed
        for _ in range(0, (num_floats + step - 1) // step):  # Ensures full coverage
            mask = block_start < num_floats  # Ensure we don't read out-of-bounds
            val = tl.load(in_ptr + block_start, mask=mask, other=0.0)  
            tl.store(out_ptr + block_start, val, mask=mask)
            
            block_start += step  # Move to the next chunk


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

def main(mode, kernel_id, num_threads_per_tb_copy, num_threads_per_tb_comp, num_iters_copy, num_iters_comp, num_bytes):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # display the number of SMs
    print(f"Number of SMs: {num_sms}")
    BLOCK_SIZE_COMP = num_threads_per_tb_comp
    BLOCK_SIZE_COPY = num_threads_per_tb_copy

    # Allocate memory for compute kernel
    num_comp_elems = num_threads_per_tb_comp
    a, b, c = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float32) for _ in range(3)]

    # Allocate memory for memory kernel
    # calculate num_copy_elems from num_bytes using size of torch.float32
    num_copy_elems = num_bytes // 4
    in_tensor = torch.ones(num_copy_elems, device="cuda", dtype=torch.float32)
    out_tensor = torch.empty_like(in_tensor)

    # number of warps 
    num_warps_comp = num_threads_per_tb_comp // 32
    num_warps_copy = num_threads_per_tb_copy // 32

    # Define kernel arguments
    args_compute = {"a_ptr": a, "b_ptr": b, "c_ptr": c, "num_iters": num_iters_comp}
    args_memory = {"in_ptr": in_tensor, "out_ptr": out_tensor, "num_floats": num_copy_elems, "num_iters": num_iters_copy}

    # Define grid
    grid = (num_sms,)

    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        if kernel_id == 0:
            print("\n[Running Copy Kernel Alone]")
            time_Copy = run_kernel(memory_kernel, args_memory, grid, BLOCK_SIZE_COPY, num_warps_copy, num_runs=10, stream=None)
            print(f"Copy Kernel Latency: {time_Copy:.6f} sec")    
        else:
            compute_kernel = get_compute_kernel(kernel_id)
            print(f"\n[Running Compute Kernel Alone with ILP {kernel_id}]")
            time_compute = run_kernel(compute_kernel, args_compute, grid, BLOCK_SIZE_COMP, num_warps_comp, num_runs=10, stream=None)
            print(f"Compute Kernel Latency: {time_compute:.6f} sec")

    elif mode == 2:
        print(f"\n[Running copy kernel and compute kernel with ILP {kernel_id} sequentially]")
        num_runs = 10
        compute_kernel = get_compute_kernel(kernel_id)
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            # Record the start event
            start_event.record()
            compute_kernel[grid](**args_compute, BLOCK_SIZE=BLOCK_SIZE_COMP, num_warps=num_warps_comp)
            memory_kernel[grid](**args_memory, BLOCK_SIZE=BLOCK_SIZE_COPY, num_warps=num_warps_copy)
            # Record the end event
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        avg_time = sum(times) / (num_runs * 1000)
        print(f"Sequential Execution Latency: {avg_time:.6f} sec")

    elif mode == 3:
        print(f"\n[Running copy kernel and compute kernel with ILP {kernel_id} colocated]")
        num_runs = 10
        compute_kernel = get_compute_kernel(kernel_id)
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
                compute_kernel[grid](**args_compute, BLOCK_SIZE=BLOCK_SIZE_COMP, num_warps=num_warps_comp)
                stop1.record(stream1)
            
            with torch.cuda.stream(stream2):
                start2.record(stream2)
                memory_kernel[grid](**args_memory, BLOCK_SIZE=BLOCK_SIZE_COPY, num_warps=num_warps_copy)
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
    if len(sys.argv) < 8:
        print("Usage: python script.py <mode> <kernel_id> <num_threads_per_tb_copy> "
              "<num_threads_per_tb_comp> <num_iters_copy> <num_iters_comp> <num_bytes>")
        sys.exit(1)

    mode = int(sys.argv[1])
    kernel_id = int(sys.argv[2])
    num_threads_per_tb_copy = int(sys.argv[3])
    num_threads_per_tb_comp = int(sys.argv[4])
    num_iters_copy = int(sys.argv[5])
    num_iters_comp = int(sys.argv[6])
    num_bytes = int(sys.argv[7])

    main(mode, kernel_id, num_threads_per_tb_copy, num_threads_per_tb_comp, num_iters_copy, num_iters_comp, num_bytes)
