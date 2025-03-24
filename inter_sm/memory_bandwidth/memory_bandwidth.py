import torch
import triton
import triton.language as tl
import time


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
    print(f"args: {args}")
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.cuda.stream(stream):
            kernel[grid](**args, BLOCK_SIZE=block_size, num_warps=num_warps)

    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_runs

def main(mode, num_tb, num_threads_per_tb, num_iters, num_bytes):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # display the number of SMs
    print(f"Number of SMs: {num_sms}")
    BLOCK_SIZE = num_threads_per_tb

    num_mb = num_bytes / (1024 * 1024)
    # Allocate memory for memory kernel
    # calculate num_copy_elems from num_bytes using size of torch.float32
    num_copy_elems = num_bytes // 4
    in_tensor1 = torch.ones(num_copy_elems, device="cuda", dtype=torch.float32)
    out_tensor1 = torch.empty_like(in_tensor1)
    in_tensor2 = torch.ones(num_copy_elems, device="cuda", dtype=torch.float32)
    out_tensor2 = torch.empty_like(in_tensor2)

    # number of warps 
    num_warps = num_threads_per_tb // 32
    # Define kernel arguments
    args_memory_1 = {"in_ptr": in_tensor1, "out_ptr": out_tensor1, "num_floats": num_copy_elems, "num_iters": num_iters}
    args_memory_2 = {"in_ptr": in_tensor2, "out_ptr": out_tensor2, "num_floats": num_copy_elems, "num_iters": num_iters}

    # Define grid
    grid = (num_tb,)

    print(f"Launch Configuration {grid = }, {BLOCK_SIZE = }, {num_warps = } - copying {num_mb} MB data")

    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        print("\n[Running Copy Kernel Alone]")
        time_Copy = run_kernel(memory_kernel, args_memory_1, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        lat_sec = time_Copy
        tot_num_bytes = 2 * 1.0 * num_iters * num_bytes
        bw = (tot_num_bytes / lat_sec) / (1024 * 1024 * 1024) # GB/s
        print(f"Average achieved bandwidth: {bw:.6f} GB/s")
    elif mode == 2:
        print("\n[Running two Copy Kernel sequentially]")
        num_runs = 10

        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            # Record the start event
            start_event.record()
            memory_kernel[grid](**args_memory_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            memory_kernel[grid](**args_memory_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            # Record the end event
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        avg_time = sum(times) / (num_runs * 1000)
        print(f"Sequential Execution Latency: {avg_time:.6f} sec")

    elif mode == 3:
        print("\n[Running two Copy Kernel colocated using CUDA streams]")
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
                memory_kernel[grid](**args_memory_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                stop1.record(stream1)
            
            with torch.cuda.stream(stream2):
                start2.record(stream2)
                memory_kernel[grid](**args_memory_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
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
    if len(sys.argv) < 6:
        print("Usage: python script.py <mode> <num_tb> <num_threads_per_tb> <num_iters> <num_bytes>")
        sys.exit(1)

    mode = int(sys.argv[1])
    num_tb = int(sys.argv[2])
    num_threads_per_tb = int(sys.argv[3])
    num_iters = int(sys.argv[4])
    num_bytes = int(sys.argv[5])

    main(mode, num_tb, num_threads_per_tb, num_iters, num_bytes)
