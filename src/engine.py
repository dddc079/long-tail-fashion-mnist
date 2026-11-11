"""训练与评测引擎。"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model, loader, device, bias=None, max_samples=None):
    model.eval()
    logits_all, y_all, feat_all = [], [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        h = model.features(x)
        out = model.fc(h)
        if bias is not None:
            out = out + bias.to(device)
        logits_all.append(out.cpu())
        y_all.append(y)
        if max_samples is not None and n < max_samples:
            feat_all.append(h.cpu())
            n += h.size(0)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    feats = torch.cat(feat_all) if feat_all else None
    return logits, y, feats


@torch.no_grad()
def evaluate(model, loader, device, bias=None):
    logits, y, _ = collect_logits(model, loader, device, bias)
    loss = F.cross_entropy(logits, y).item()
    return loss, logits.numpy(), y.numpy()


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=None):
    model.train()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    return tot / max(n, 1)


# ---------------- 指标 ----------------

def confusion_matrix(y, pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def compute_metrics(y, logits, class_counts, num_classes):
    pred = logits.argmax(1)
    cm = confusion_matrix(y, pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    acc = tp.sum() / max(support.sum(), 1)

    recall = tp / np.maximum(support, 1)
    precision = tp / np.maximum(predicted, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    # 按训练样本数排序，等分为 头部/中部/尾部 三组
    counts = np.asarray(class_counts, dtype=np.float64)
    order = np.argsort(-counts)
    k = num_classes // 3
    groups = {
        "many": order[:k],
        "medium": order[k:2 * k],
        "few": order[2 * k:],
    }
    group_acc = {}
    for gname, idx in groups.items():
        mask = np.isin(y, idx)
        group_acc[gname] = float((pred[mask] == y[mask]).mean()) if mask.sum() else float("nan")

    # 期望校准误差 ECE（15 个等宽置信度区间）
    prob = torch.softmax(torch.tensor(logits), dim=1).numpy()
    conf = prob.max(1)
    correct = (pred == y).astype(np.float64)
    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())

    return {
        "acc": float(acc),
        "macro_f1": float(f1.mean()),
        "group_acc": group_acc,
        "per_class_recall": recall.tolist(),
        "ece": float(ece),
        "confusion": cm.tolist(),
        "groups": {k2: [int(i) for i in v] for k2, v in groups.items()},
    }


def weight_norms(model):
    """分类器各类别权重向量的 L2 范数，用于分析分类器偏置。"""
    with torch.no_grad():
        w = model.fc.weight.detach().cpu().numpy()
    return np.linalg.norm(w, axis=1).tolist()
