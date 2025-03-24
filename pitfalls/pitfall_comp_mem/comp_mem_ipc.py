import torch
import triton
import triton.language as tl
import time

# ---------------------- COMPUTE KERNEL (FP64 ILP4) ----------------------
# @triton.jit
# def compute_kernel(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = block_start < num_iters

#     # Load inputs
#     a = tl.load(a_ptr + block_start, mask=mask, other=1.0)
#     b = tl.load(b_ptr + block_start, mask=mask, other=1.0)

#     # Compute loop (ILP-4 style)
#     op1, op2, op3, op4 = a, b, 1.0, 1.0
#     for _ in range(num_iters):
#         op3 = op1 * op3
#         op4 = op2 * op4

#     # Store result
#     c = op3 + op4
#     tl.store(c_ptr + block_start, c, mask=mask)

# @triton.jit
# def compute_kernel(a_ptr, b_ptr, c_ptr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = block_start < num_iters

#     # Load inputs (FP64)
#     a = tl.load(a_ptr + block_start, mask=mask).to(tl.float64)
#     b = tl.load(b_ptr + block_start, mask=mask).to(tl.float64)

#     # Compute loop (ILP-4 style) using FP64
#     op1, op2 = a, b
#     op3 = tl.full(block_start.shape, 1.0, dtype=tl.float64)  # FP64 constant
#     op4 = tl.full(block_start.shape, 1.0, dtype=tl.float64)

#     for _ in range(num_iters):
#         op3 = op1 * op3
#         op4 = op2 * op4

#     # Store result
#     c = op3 + op4
#     tl.store(c_ptr + block_start, c, mask=mask)


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


# ---------------------- MEMORY COPY KERNEL ----------------------
# @triton.jit
# def memory_kernel(in_ptr, out_ptr, num_floats: tl.constexpr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     step = tl.num_programs(0) * BLOCK_SIZE
    
#     for _ in range(num_iters):
#         for i in range(block_start, num_floats, step):
#             val = tl.load(in_ptr + i, mask=i < num_floats)
#             tl.store(out_ptr + i, val, mask=i < num_floats)

# @triton.jit
# def memory_kernel(in_ptr, out_ptr, num_floats: tl.constexpr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)  
#     block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
#     mask = block_start < num_floats  # Prevent out-of-bounds access

#     for _ in range(num_iters):
#         val = tl.load(in_ptr + block_start, mask=mask, other=0.0)  
#         tl.store(out_ptr + block_start, val, mask=mask)
        
#         block_start += BLOCK_SIZE * tl.num_programs(0)  # Stride forward
#         mask = block_start < num_floats  # Update mask

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
    print(f"args: {args}")
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.cuda.stream(stream):
            kernel[grid](**args, BLOCK_SIZE=block_size, num_warps=num_warps)

    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_runs

def main(mode, num_threads_per_tb, num_iters_comp, num_iters_copy):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    BLOCK_SIZE = num_threads_per_tb

    # Allocate memory for compute kernel
    num_comp_elems = num_threads_per_tb
    a, b, c = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float32) for _ in range(3)]

    # Allocate memory for memory kernel
    num_copy_elems = 1024 * 1024 * 1024  # 4GB
    in_tensor = torch.ones(num_copy_elems, device="cuda", dtype=torch.float32)
    out_tensor = torch.empty_like(in_tensor)

    # number of warps 
    num_warps = num_threads_per_tb // 32
    # Define kernel arguments
    args_compute = {"a_ptr": a, "b_ptr": b, "c_ptr": c, "num_iters": num_iters_comp}
    args_memory = {"in_ptr": in_tensor, "out_ptr": out_tensor, "num_floats": num_copy_elems, "num_iters": num_iters_copy}

    # Define grid
    grid = (num_sms,)

    print(f"num_program = {num_sms}, BLOCK_SIZE = {BLOCK_SIZE}, num_copy_elems = {num_copy_elems}")
    
    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        print("\n[Running Compute Kernel Alone]")
        time_compute = run_kernel(compute_kernel, args_compute, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        print(f"Compute Kernel Latency: {time_compute:.6f} sec")

        print("\n[Running Memory Copy Kernel Alone]")
        time_memory = run_kernel(memory_kernel, args_memory, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        print(f"Memory Kernel Latency: {time_memory:.6f} sec")

    elif mode == 3:
        print("\n[Running Compute and Memory Kernel Concurrently]")
        num_runs = 10
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_runs):
            with torch.cuda.stream(stream1):
                compute_kernel[grid](**args_compute, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            with torch.cuda.stream(stream2):
                memory_kernel[grid](**args_memory, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            torch.cuda.synchronize()
        
        # with torch.cuda.stream(stream1):
        #     compute_kernel[grid](**args_compute, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        # with torch.cuda.stream(stream2):
        #     memory_kernel[grid](**args_memory, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        print(f"Concurrent Execution Latency: {avg_time:.6f} sec")
    print(f"args: {args_memory}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python script.py <mode> <num_threads_per_tb> <num_iters_comp> <num_iters_copy>")
        sys.exit(1)

    mode = int(sys.argv[1])
    num_threads_per_tb = int(sys.argv[2])
    num_iters_comp = int(sys.argv[3])
    num_iters_copy = int(sys.argv[4])

    main(mode, num_threads_per_tb, num_iters_comp, num_iters_copy)
