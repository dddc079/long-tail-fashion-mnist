"""补算 CE 基线的推理期后处理 Logit Adjustment 结果。

原因：主实验早期的 post-hoc 偏置符号写反（应为 -tau*log(pi)），本脚本用修正后的
实现重新训练同样的 CE 模型（相同种子与配置），把后处理指标写回原结果 JSON。
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch

from common import set_seed, save_json, RUN_DIR, ensure_dirs
from data import build_loaders
from models import build_model
from losses import posthoc_bias
from engine import train_one_epoch, evaluate, compute_metrics


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    ensure_dirs()
    for seed in (0, 1, 2):
        cfg = dict(dataset="fashion", num_classes=10, imbalance_ratio=100,
                   batch_size=512, epochs=10, lr=1e-3, wd=5e-4, width=32,
                   dropout=0.3, sampler="instance")
        set_seed(seed)
        bundle = build_loaders(cfg, seed)
        model = build_model(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        for ep in range(cfg["epochs"]):
            tl = train_one_epoch(model, bundle["train"], torch.nn.CrossEntropyLoss(),
                                 opt, torch.device("cpu"))
            sched.step()
        _, logits, y = evaluate(model, bundle["test"], torch.device("cpu"))
        np.savez(os.path.join(RUN_DIR, f"logits_ce_ir100_s{seed}.npz"),
                 logits=logits, label=y)

        counts = bundle["class_counts"]
        bias = posthoc_bias(counts, tau=1.0)
        _, logits_ph, _ = evaluate(model, bundle["test"], torch.device("cpu"), bias=bias)
        ph = compute_metrics(y, logits_ph, counts, cfg["num_classes"])

        path = os.path.join(RUN_DIR, f"fashion_ir100_ce_s{seed}.json")
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r["posthoc"] = ph
        r["posthoc_tau"] = 1.0
        save_json(r, path)
        print(f"seed {seed}: CE acc {r['metrics']['acc']:.4f} -> posthoc {ph['acc']:.4f} "
              f"(tail {ph['group_acc']['few']:.4f})", flush=True)


if __name__ == "__main__":
    main()
