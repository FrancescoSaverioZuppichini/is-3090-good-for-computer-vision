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

Since I am mostly interested in vision, I've used a very vgg-like conv net.

```python
def get_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        # stage 1
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        # stage 2
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        # stage 3
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
```

Motivated by the recent progress in vision since [RepVGG](https://arxiv.org/abs/2101.03697).