import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(["science"])


@dataclass
class BenchmarkMetadata:
    torch: str
    cuda: str
    cudnn: str
    gpu: str
    cudnn_benchmark: Optional[bool] = False
    driver: Optional[bool] = "515.57"

    @classmethod
    def from_filename(cls, filename: str):
        parts = filename.split("_")

        kwargs = {}

        for part in parts:
            key, val = part.split("=")
            # replace 'foo-bar' with a python 'foo_bar'
            key = key.replace("-", "_")
            kwargs[key] = val

        return cls(**kwargs)


@dataclass
class Benchmark:
    metadata: BenchmarkMetadata
    df: pd.DataFrame

    @classmethod
    def from_filepath(cls, filepath: Path):
        return Benchmark(
            metadata=BenchmarkMetadata.from_filename(filepath.stem),
            df=pd.read_csv(filepath),
        )


root = Path(os.getcwd()) / "benchmarks"
csvs = root.glob("*.csv")

benchmarks = [Benchmark.from_filepath(filepath) for filepath in csvs]


def make_comparison_with_all_benchmarks():
    # for this one we want to ignore the old 1080ti driver
    gpus_to_dfs = defaultdict(list)

    for benchmark in benchmarks:
        if benchmark.metadata.gpu == "470":
            continue
        gpus_to_dfs[benchmark.metadata.gpu].append(benchmark.df)

    gpus_to_stats = {}

    for gpu, dfs in gpus_to_dfs.items():
        # I don't like pandas and I suck
        data = np.array([df["Time (s/iter)"].values for df in dfs])
        # get mean and std across all benchmarks for that gpu
        stats = data.mean(0), data.std(0)
        gpus_to_stats[gpu] = stats

    # matplotlib goes brumm brumm
    fig, ax = plt.subplots()
    batch_sizes = benchmarks[0].df["batch size"].values
    # x_axis = np.delete(x_axis, 1)
    offset = [x for x in range(len(batch_sizes))]
    bar_width = 0.33
    for gpu, stats in gpus_to_stats.items():
        mean, std = stats
        mean *= 1000
        std *= 1000

        ax.bar(
            offset,
            mean,
            width=bar_width,
            yerr=std,
            error_kw=dict(lw=0.33, capsize=3, capthick=0.33),
            label=gpu,
        )
        offset = [el + bar_width for el in offset]
    _, labels = plt.xticks(
        [el + bar_width for el in range(len(batch_sizes))],
        [str(batch_size) for batch_size in batch_sizes],
    )

    plt.title("GTX 1080ti vs RTX 3090 - Conv Model")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    fig.savefig("./plots/all", dpi=800)


make_comparison_with_all_benchmarks()
