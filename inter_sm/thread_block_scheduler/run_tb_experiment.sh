#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_tb=132 # TODO: set to number of SMs present on GPU
num_threads_per_tb=1024 # TODO: set to max_threads_per_sm / 2
num_itrs=100000

# Measuring latnecy of single low ipc kernel
python3 tb_scheduler.py 1 $num_tb $num_threads_per_tb $num_itrs

# Measuring sequential latency of two low ipc kernels
python3 tb_scheduler.py 2 $num_tb $num_threads_per_tb $num_itrs

# Measuring colocated latency of two low ipc kernels
python3 tb_scheduler.py 3 $num_tb $num_threads_per_tb $num_itrs

# DOUBLING NUMBER OF THREAD BLOCKS

# Measuring latnecy of single low ipc kernel with double the number of thread blocks
python3 tb_scheduler.py 1 $((num_tb * 2)) $num_threads_per_tb $num_itrs

# Measuring sequential latency of two low ipc kernels with double the number of thread blocks
python3 tb_scheduler.py 2 $((num_tb * 2)) $num_threads_per_tb $num_itrs

# Measuring colocated latency of two low ipc kernels with double the number of thread blocks
python3 tb_scheduler.py 3 $((num_tb * 2)) $num_threads_per_tb $num_itrs