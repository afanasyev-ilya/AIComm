# AIComm


Runtime:
```
export LD_LIBRARY_PATH=/home/i.afanasyev/nccl/build/lib:$LD_LIBRARY_PATH

```

Profiling:

Using nsys 2025.6.1+ is mandatory
```
export PATH="/home/i.afanasyev/opt/nsys-cli/extract/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"
nsys profile --trace=cuda -o new_trace ./tp_col.bin --gpus 2 --iters 50 --check
```