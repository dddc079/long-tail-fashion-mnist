"""用修正后的第二阶段配置重跑 cRT 相关实验。

原实现中第二阶段仅 10 个 epoch、每轮 4 个 batch（共 40 步更新）、lr 1e-3，
分类器从随机初始化出发明显欠训练。修正为 60 epoch、lr 1e-2、batch 64。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from common import ensure_dirs
from runner import run_single


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    ensure_dirs()
    jobs = [dict(method="crt", seed=s, imbalance_ratio=100) for s in (0, 1, 2)]
    jobs += [dict(method="crt", seed=0, imbalance_ratio=ir) for ir in (10, 50)]
    for j in jobs:
        print(f"[crt-fix] {j}", flush=True)
        run_single(dict(j, save_embed=(j["imbalance_ratio"] == 100 and j["seed"] == 0)),
                   verbose=True)


if __name__ == "__main__":
    main()
