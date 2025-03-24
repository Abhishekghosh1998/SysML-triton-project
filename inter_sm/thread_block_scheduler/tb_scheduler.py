# think about it
# the sleep kernel is not possible in triton
# think of an alternative in triton

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

def main(mode, num_tb, num_threads_per_tb, num_iters):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # display the number of SMs
    print(f"Number of SMs: {num_sms}")
    BLOCK_SIZE = num_threads_per_tb

    num_comp_elems = num_threads_per_tb
    # Allocate memory for kernel 1
    # a1, b1, c1 = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float32) for _ in range(3)]
    dummy1 = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
    # Allocate memory for kernel 2
    # a2, b2, c2 = [torch.ones(num_comp_elems, device="cuda", dtype=torch.float32) for _ in range(3)]
    dummy2 = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
    

    # number of warps 
    num_warps = num_threads_per_tb // 32
    # Define kernel arguments
    kernel_args_1 = {"dummy_ptr": dummy1, "num_iters": num_iters}
    kernel_args_2 = {"dummy_ptr": dummy2, "num_iters": num_iters}
    
    # Define grid
    grid = (num_tb,)

    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        print("\n[Running Low IPC Kernel Alone]")
        time_kernel = run_kernel(low_ipc_kernel, kernel_args_1, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        print(f"Low IPC Kernel Latency: {time_kernel:.6f} sec")

    elif mode == 2:
        print("\n[Running two Low IPC Kernel sequentially]")
        num_runs = 10

        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            # Record the start event
            start_event.record()
            low_ipc_kernel[grid](**kernel_args_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            low_ipc_kernel[grid](**kernel_args_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            # Record the end event
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        avg_time = sum(times) / (num_runs * 1000)
        print(f"Sequential Execution Latency: {avg_time:.6f} sec")

    elif mode == 3:
        print("\n[Running two Low IPC Kernel colocated using CUDA streams]")
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
                low_ipc_kernel[grid](**kernel_args_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                stop1.record(stream1)
            
            with torch.cuda.stream(stream2):
                start2.record(stream2)
                low_ipc_kernel[grid](**kernel_args_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
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
        print("Usage: python script.py <mode> <num_tb> <num_threads_per_tb> <num_iters>")
        sys.exit(1)

    mode = int(sys.argv[1])
    num_tb = int(sys.argv[2])
    num_threads_per_tb = int(sys.argv[3])
    num_iters = int(sys.argv[4])

    main(mode, num_tb, num_threads_per_tb, num_iters)
