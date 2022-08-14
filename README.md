# Is the 3090 good for computer vision

This repo holds a set of personal benchmark I've done on my system.


```
OS: Ubuntu 22.04.1 LTS x86_64 
CPU: AMD Ryzen 7 2700X (16) @ 3.700GHz 
GPU: NVIDIA GeForce RTX 3090 
Memory: 6823MiB / 64260MiB 
```

You can find all the benchmarks inside [benchmarks](/benchmarks)

The following figure shows an average across all the benchmarks, the **RTX 3090** is **~2 times faster** than the 1080ti. 


![img](/plots/all.png)

## Benchmark settings

For most benchmarks I've used the [latest driver from nvidia](https://www.nvidia.com/Download/driverResults.aspx/190414/en-us/) and [their containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) with different version of CUDA, Cudnn and PyTorch.