"""不平衡学习相关的损失函数。

- FocalLoss : Lin et al., ICCV 2017
- ClassBalancedCE : Cui et al., CVPR 2019
- LogitAdjustedCE : Menon et al., ICLR 2021 (训练期版本)
- 后处理 logit adjustment 通过 eval 时给 logits 加 tau*log(prior) 实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight.numel() > 0:
            loss = loss * self.weight[target]
        return loss.mean()


class ClassBalancedCE(nn.Module):
    """类别平衡损失：权重 (1-beta)/(1-beta^{n_c})。"""

    def __init__(self, counts, beta=0.9999, num_classes=10):
        super().__init__()
        counts = np.asarray(counts, dtype=np.float64)
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / np.maximum(eff, 1e-12)
        w = w / w.sum() * num_classes
        self.register_buffer("weight", torch.tensor(w, dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight)


class LogitAdjustedCE(nn.Module):
    """训练期 logit adjustment：CE(logits + tau * log(prior))。"""

    def __init__(self, counts, tau=1.0):
        super().__init__()
        prior = np.asarray(counts, dtype=np.float64)
        prior = prior / prior.sum()
        self.register_buffer("bias", tau * torch.tensor(np.log(prior), dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.bias, target)


def build_criterion(name, counts, num_classes):
    if name == "focal":
        return FocalLoss(gamma=2.0)
    if name == "cbce":
        return ClassBalancedCE(counts, beta=0.9999, num_classes=num_classes)
    if name == "la_loss":
        return LogitAdjustedCE(counts, tau=1.0)
    return nn.CrossEntropyLoss()


def posthoc_bias(counts, tau=1.0):
    """推理期后处理 logit adjustment：argmax f_y - tau*log(pi_y)。

    注意符号：训练集先验已被模型隐式编码进 logits，需要**减去**它才能
    得到均衡先验下的决策规则（Menon et al., ICLR 2021 的 post-hoc 形式）。
    """
    prior = np.asarray(counts, dtype=np.float64)
    prior = prior / prior.sum()
    return torch.tensor(-tau * np.log(prior), dtype=torch.float32)
