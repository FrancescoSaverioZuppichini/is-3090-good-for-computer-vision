from time import perf_counter

import pandas as pd
import torch
from torch import nn

torch.manual_seed(0)

N_REPEAT = 32


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


def benchmark(batches_sizes, device="cpu", benchmark=False):
    records = []
    # test the models
    torch.backends.cudnn.benchmark = benchmark

    for batch_size in batches_sizes:
        x = torch.randn((batch_size, 3, 224, 224), device=torch.device(device))
        model = get_model().to(device=torch.device(device)).eval()
        with torch.no_grad():
            # warmup
            for _ in range(4):
                model(x)
            torch.cuda.synchronize()
            start = perf_counter()
            for _ in range(N_REPEAT):
                model(x)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            records.append(
                {
                    "Type": "Model",
                    "Time (s)": elapsed,
                    "Time (s/iter)": elapsed / N_REPEAT,
                    "batch size": batch_size,
                    "device": device,
                }
            )
            print(
                f"Elapsed {perf_counter() - start:.4f}s {elapsed / N_REPEAT:.4f} s/iter"
            )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    batches_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    cudnn_version = torch.backends.cudnn.version()
    print(f"cudnn version: {torch.backends.cudnn.version()}")
    df = benchmark(batches_sizes, "cuda", benchmark=True)
    # df.to_csv(f"./torch={torch.__version__}_cuda={torch.version.cuda}_cudnn={cudnn_version}_gpu=1080ti_cudnn-benchmark=True.csv", index=False)
    print(df)
