"""单次实验的执行逻辑。"""
import os
import time
import numpy as np
import torch
import torch.nn as nn

from common import set_seed, save_json, ensure_dirs, RUN_DIR
from data import build_loaders
from models import build_model
from losses import build_criterion, posthoc_bias
from engine import (train_one_epoch, evaluate, collect_logits, compute_metrics,
                    weight_norms)

DEFAULT_CFG = dict(
    dataset="fashion",
    num_classes=10,
    imbalance_ratio=100,
    batch_size=512,
    epochs=10,
    lr=1e-3,
    wd=5e-4,
    width=32,
    dropout=0.3,
    sampler="instance",
    method="ce",
    seed=0,
    save_embed=False,
)


def train_stage(model, train_loader, val_loader, criterion, opt, sched, epochs,
                device, tag="", verbose=True):
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        tl = train_one_epoch(model, train_loader, criterion, opt, device)
        if sched is not None:
            sched.step()
        _, logits, y = evaluate(model, val_loader, device)
        va = float((logits.argmax(1) == y).mean())
        hist["train_loss"].append(tl)
        hist["val_acc"].append(va)
        if verbose and ((ep + 1) % 4 == 0 or ep == epochs - 1):
            print(f"    [{tag}] ep {ep+1:2d}/{epochs} loss {tl:.4f} acc {va:.4f}",
                  flush=True)
    return hist


def run_single(cfg, verbose=True):
    cfg = {**DEFAULT_CFG, **cfg}
    ensure_dirs()
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    t0 = time.time()

    bundle = build_loaders(cfg, cfg["seed"])
    model = build_model(cfg).to(device)
    counts = bundle["class_counts"]

    criterion = build_criterion(cfg["method"], counts, cfg["num_classes"])
    epochs = cfg["epochs"]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if verbose:
        print(f"  run {cfg['method']} seed={cfg['seed']} IR={cfg['imbalance_ratio']} "
              f"n_train={bundle['meta']['n_train']}", flush=True)

    hist = {"stage1": train_stage(model, bundle["train"], bundle["val"], criterion,
                                  opt, sched, epochs, device, tag="s1",
                                  verbose=verbose)}

    # 两阶段解耦（cRT）：冻结特征提取器，在类别均衡子集上重训练分类器
    if cfg["method"] == "crt":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        nn.init.kaiming_uniform_(model.fc.weight, a=np.sqrt(5))
        nn.init.zeros_(model.fc.bias)
        # 分类器从随机初始化重新学习，需要足够的更新步数与更大的学习率
        opt2 = torch.optim.AdamW(model.fc.parameters(), lr=1e-2, weight_decay=cfg["wd"])
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=60)
        hist["stage2"] = train_stage(model, bundle["balanced"], bundle["val"],
                                     nn.CrossEntropyLoss(), opt2, sched2, 60,
                                     device, tag="s2", verbose=verbose)
        for p in model.parameters():
            p.requires_grad = True

    # 测试集评估
    _, logits, y = evaluate(model, bundle["test"], device)
    metrics = compute_metrics(y, logits, counts, cfg["num_classes"])

    # CE 基线额外给出推理期后处理（post-hoc logit adjustment）结果
    posthoc = None
    if cfg["method"] == "ce":
        bias = posthoc_bias(counts, tau=1.0)
        _, logits_ph, _ = evaluate(model, bundle["test"], device, bias=bias)
        posthoc = compute_metrics(y, logits_ph, counts, cfg["num_classes"])

    result = {
        "cfg": cfg,
        "meta": bundle["meta"],
        "history": hist,
        "metrics": metrics,
        "posthoc": posthoc,
        "weight_norms": weight_norms(model),
        "time_sec": round(time.time() - t0, 1),
        "params": int(sum(p.numel() for p in model.parameters())),
    }

    if cfg["save_embed"]:
        _, _, feats = collect_logits(model, bundle["test"], device, max_samples=4000)
        np.savez(os.path.join(RUN_DIR, f"embed_{cfg['method']}_ir{cfg['imbalance_ratio']}_s{cfg['seed']}.npz"),
                 feat=feats.numpy()[:4000], label=y[:4000])

    name = f"{cfg['dataset']}_ir{cfg['imbalance_ratio']}_{cfg['method']}_s{cfg['seed']}"
    save_json(result, os.path.join(RUN_DIR, name + ".json"))
    if verbose:
        m = metrics
        print(f"  -> acc {m['acc']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"many {m['group_acc']['many']:.4f} med {m['group_acc']['medium']:.4f} "
              f"few {m['group_acc']['few']:.4f}  ({result['time_sec']}s)", flush=True)
    return result
