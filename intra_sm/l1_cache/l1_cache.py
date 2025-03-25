import torch
import triton
import triton.language as tl
import time


# ---------------------- MEMORY COPY KERNEL --------------------------------
# @triton.jit
# def memory_kernel(in_ptr, out_ptr, num_floats: tl.constexpr, num_iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)  
#     step = BLOCK_SIZE * tl.num_programs(0)  # The step size to move forward
    
#     for _ in range(num_iters):
#         block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
#         # Iterate over memory in steps until all elements are processed
#         for _ in range(0, (num_floats + step - 1) // step):  # Ensures full coverage
#             mask = block_start < num_floats  # Ensure we don't read out-of-bounds
#             val = tl.load(in_ptr + block_start, mask=mask, other=0.0)  
#             tl.store(out_ptr + block_start, val, mask=mask)
            
#             block_start += step  # Move to the next chunk

@triton.jit
def memory_kernel_per_tb(
    in_ptr, out_ptr, num_floats_per_tb: tl.constexpr, num_iters: tl.constexpr, 
    region_size_bytes: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Thread block and thread indexing
    pid = tl.program_id(0)  # Thread block ID
    lane = tl.arange(0, BLOCK_SIZE)  # Thread index within a block

    # Calculate floats per region
    floats_per_region = region_size_bytes // 4  # sizeof(float) = 4 bytes

    # Calculate regions per thread block to avoid overlaps
    regions_per_tb = (num_floats_per_tb + floats_per_region - 1) // floats_per_region

    # Compute the range of indices this thread block should operate on
    block_begin = pid * regions_per_tb * floats_per_region
    block_end = block_begin + num_floats_per_tb

    stride = BLOCK_SIZE  # How far each thread moves per step

    # Outer loop for multiple iterations
    for _ in range(num_iters):
        # Initialize index at block_begin
        index = block_begin + lane
        # Inner loop to shift indices dynamically
        for _ in range(0, (num_floats_per_tb + stride - 1) // stride):
            # tl.device_print(" ")
            mask = index < block_end  # Ensure we don't read out-of-bounds
            val = tl.load(in_ptr + index, mask=mask, other=0.0)
            tl.store(out_ptr + index, val, mask=mask)

            index += stride  # Shift index dynamically


# ---------------------- DRIVER FUNCTION ----------------------
def run_kernel(kernel, args, grid, block_size, num_warps, num_runs=10, stream=None):
    # print(f"args: {args}")
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.cuda.stream(stream):
            kernel[grid](**args, BLOCK_SIZE=block_size, num_warps=num_warps)

    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_runs

def main(mode, num_threads_per_tb, num_bytes_per_tb, unified_l1_cache_size, num_iters):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # display the number of SMs
    print(f"Number of SMs: {num_sms}")
    BLOCK_SIZE = num_threads_per_tb

    num_floats_per_tb = num_bytes_per_tb // 4 # divide by 4 bytes per float
    unified_l1_cache_size_bytes = unified_l1_cache_size * 1024

    # allocate separate non overlapping memory regions for each thread block
    # the size of each region is equal to the unified L1 cache size
    # if num_bytes_per_tb > unified_l1_cache_size, then we need to allocate multiple regions per thread block
    num_regions_per_block = (num_bytes_per_tb + unified_l1_cache_size_bytes - 1) // unified_l1_cache_size_bytes
    tot_elems = num_sms * num_regions_per_block * unified_l1_cache_size_bytes // 4 # divide by 4 bytes per float
    print(f"Total number of elements allocated: {tot_elems}")
    # Allocate memory for memory kernel
    # calculate num_copy_elems from num_bytes using size of torch.float32
    in_tensor1 = torch.ones(tot_elems, device="cuda", dtype=torch.float32)
    out_tensor1 = torch.empty_like(in_tensor1)
    in_tensor2 = torch.ones(tot_elems, device="cuda", dtype=torch.float32)
    out_tensor2 = torch.empty_like(in_tensor2)

    # number of warps 
    num_warps = num_threads_per_tb // 32

    # Define kernel arguments
    args_memory_1 = {"in_ptr": in_tensor1, "out_ptr": out_tensor1, "num_floats_per_tb": num_floats_per_tb, 
                     "num_iters": num_iters, "region_size_bytes": unified_l1_cache_size_bytes}
    args_memory_2 = {"in_ptr": in_tensor2, "out_ptr": out_tensor2, "num_floats_per_tb": num_floats_per_tb, 
                     "num_iters": num_iters, "region_size_bytes": unified_l1_cache_size_bytes}
    # Define grid
    grid = (num_sms,)

    print(f"Launch Configuration {grid = }, {BLOCK_SIZE = }, {num_warps = } - copying {num_bytes_per_tb / 1024} KB per thread block")

    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    if mode == 1:
        print("\n[Running Copy Kernel Alone]")
        time_Copy = run_kernel(memory_kernel_per_tb, args_memory_1, grid, BLOCK_SIZE, num_warps, num_runs=10, stream=None)
        print(f"Copy Kernel Latency: {time_Copy:.6f} sec")

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
            memory_kernel_per_tb[grid](**args_memory_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            memory_kernel_per_tb[grid](**args_memory_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            # Record the end event
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        # print(f"{args_memory_1 = }")
        # print(f"{sum(args_memory_1['out_ptr']) = }")
        # print(f"{args_memory_2 = }")
        # print(f"{sum(args_memory_2['out_ptr']) = }")
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
                memory_kernel_per_tb[grid](**args_memory_1, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                stop1.record(stream1)
            
            with torch.cuda.stream(stream2):
                start2.record(stream2)
                memory_kernel_per_tb[grid](**args_memory_2, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                stop2.record(stream2)

            torch.cuda.synchronize()
            time1 = start1.elapsed_time(stop1)
            time2 = start2.elapsed_time(stop2)
            makespan1 = start1.elapsed_time(stop2)
            makespan2 = start2.elapsed_time(stop1)
            # get max of time1, time2, makespan1, makespan2
            span = max(time1, time2, makespan1, makespan2)
            times.append(span)
        
        # print(f"{args_memory_1 = }")
        # print(f"{sum(args_memory_1['out_ptr']) = }")
        # print(f"{args_memory_2 = }")
        # print(f"{sum(args_memory_2['out_ptr']) = }")
        
        avg_time = sum(times) / (num_runs * 1000)
        print(f"Concurrent Execution Latency: {avg_time:.6f} sec")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("Usage: python script.py <mode> <num_threads_per_tb> <num_bytes_per_tb> <unified_l1_cache_size> <num_itrs>")
        sys.exit(1)

    mode = int(sys.argv[1])
    num_threads_per_tb = int(sys.argv[2])
    num_bytes_per_tb = int(sys.argv[3])
    unified_l1_cache_size = int(sys.argv[4])
    num_iters = int(sys.argv[5])

    main(mode, num_threads_per_tb, num_bytes_per_tb, unified_l1_cache_size, num_iters)

    # main(mode, num_tb, num_threads_per_tb, num_iters, num_bytes)
