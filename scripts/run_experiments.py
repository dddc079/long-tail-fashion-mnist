"""实验总调度（支持分片并行）。

用法：
    python scripts/run_experiments.py main     0 3
    python scripts/run_experiments.py ablation 1 3
第 2、3 个参数是分片编号与分片总数，用于多进程并行。
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from common import ensure_dirs
from runner import run_single

METHODS = ["ce", "focal", "cbce", "oversample", "la_loss", "crt"]


def jobs_main():
    out = []
    for m in METHODS:
        for s in (0, 1, 2):
            out.append(dict(method=m, seed=s, imbalance_ratio=100,
                            sampler="balanced" if m == "oversample" else "instance",
                            save_embed=(m in ("ce", "crt") and s == 0)))
    return out


def jobs_ablation():
    out = []
    for ir in (10, 50):
        for m in ("ce", "cbce", "crt"):
            for s in (0,):
                out.append(dict(method=m, seed=s, imbalance_ratio=ir))
    return out


def jobs_all():
    return jobs_main() + jobs_ablation()


def main():
    ensure_dirs()
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    threads = int(os.environ.get("TORCH_THREADS", "4"))
    torch.set_num_threads(threads)

    jobs = {"main": jobs_main, "ablation": jobs_ablation, "all": jobs_all}[what]()
    mine = [j for i, j in enumerate(jobs) if i % nshard == shard]
    print(f"[{what}] shard {shard}/{nshard}: {len(mine)} runs, threads={threads}",
          flush=True)
    t0 = time.time()
    for k, j in enumerate(mine):
        print(f"  [{k+1}/{len(mine)}] start", flush=True)
        run_single(j, verbose=True)
    print(f"[{what}] shard {shard} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
