"""生成报告所需图表。"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from aggregate import load_runs, summarize, ORDER, DISPLAY  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

CLASS_CN = ["T恤", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
COLORS = {
    "ce": "#4C72B0", "focal": "#DD8452", "cbce": "#55A868", "oversample": "#C44E52",
    "la_loss": "#8172B3", "crt": "#937860", "ce_la": "#DA8BC3",
}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png")
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print("saved", name)


# ---------------- 图 1：长尾分布与样本示意 ----------------
def fig_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axes[0]
    base = df[(df.ir == 100) & (df.method == "ce")].iloc[0]
    c100 = [int(base.n_train) for _ in range(1)]  # 占位，真实分布从 run json 读取
    with open(sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*ir100_ce_*.json")))[0],
              encoding="utf-8") as f:
        r = json.load(f)
    counts100 = [r["meta"]["class_counts_train"][str(c)] for c in range(10)]
    with open(sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*ir10_ce_*.json")))[0],
              encoding="utf-8") as f:
        r10 = json.load(f)
    counts10 = [r10["meta"]["class_counts_train"][str(c)] for c in range(10)]

    x = np.arange(10)
    ax.bar(x - 0.2, counts10, 0.4, label="IR = 10", color="#9ECAE1", edgecolor="#4C72B0",
           linewidth=0.6)
    ax.bar(x + 0.2, counts100, 0.4, label="IR = 100", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_CN, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("训练样本数")
    ax.set_title("(a) 人工长尾分布（按类别频次降序）")
    ax.legend(frameon=False)
    for xi, v in zip(x + 0.2, counts100):
        ax.text(xi, v + 110, str(v), ha="center", fontsize=5.4, color="#22456e")

    # 样本示意
    ax = axes[1]
    from data import load_base
    imgs, labels, _, _ = load_base("fashion", True)
    grid = np.zeros((2, 5, 28, 28), dtype=np.float32)
    for k in range(10):
        i, j = divmod(k, 5)
        idx = np.where(labels.numpy() == k)[0]
        grid[i, j] = imgs[idx[np.random.RandomState(k).randint(len(idx))], 0].float() / 255.0
    canvas = np.zeros((28 * 2 + 6, 28 * 5 + 8 * 4), dtype=np.float32)
    for i in range(2):
        for j in range(5):
            canvas[i * 34:i * 34 + 28, j * 36:j * 36 + 28] = grid[i, j]
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("(b) 十类样本示例（0→9）", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig1_distribution")


# ---------------- 图 6：训练动态 ----------------
def fig_curves():
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    for m in ["ce", "focal", "cbce", "oversample", "la_loss", "crt"]:
        files = glob.glob(os.path.join(ROOT, "results", "runs", f"*ir100_{m}_s*.json"))
        A = []
        for pth in files:
            with open(pth, encoding="utf-8") as f:
                A.append(json.load(f)["history"]["stage1"]["val_acc"])
        if not A:
            continue
        A = np.array(A) * 100
        ep = np.arange(1, A.shape[1] + 1)
        ax.plot(ep, A.mean(0), color=COLORS[m], label=DISPLAY[m], linewidth=1.3)
        ax.fill_between(ep, A.mean(0) - A.std(0), A.mean(0) + A.std(0),
                        color=COLORS[m], alpha=0.12, linewidth=0)
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("验证集准确率 (%)")
    ax.set_ylim(45, 97)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="lower right")
    fig.tight_layout()
    save(fig, "fig6_curves")


# ---------------- 图 3：主结果 ----------------
def fig_main(df):
    s = summarize(df, 100)
    metrics = [("acc_mean", "acc_std", "总体准确率"), ("f1_mean", "f1_std", "宏平均 F1"),
               ("many_mean", None, "头部类"), ("medium_mean", None, "中部类"),
               ("few_mean", "few_std", "尾部类")]
    fig, axes = plt.subplots(1, 5, figsize=(6.8, 2.45), sharey=True)
    for ax, (mk, sk, title) in zip(axes, metrics):
        vals = s[mk].values * 100
        errs = s[sk].values * 100 if sk else None
        bars = ax.bar(np.arange(len(s)), vals, color=[COLORS[m] for m in s.index],
                      width=0.72, yerr=errs, capsize=2,
                      error_kw=dict(lw=0.8, ecolor="#555555"))
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(np.arange(len(s)))
        short = {"ce": "CE", "focal": "Focal", "cbce": "CB-CE", "oversample": "ROS",
                 "la_loss": "LA", "crt": "cRT", "ce_la": "CE+LA"}
        ax.set_xticklabels([short.get(m, m) for m in s.index],
                           rotation=90, ha="center", fontsize=6.2)
        ax.grid(axis="x", visible=False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}", ha="center",
                    fontsize=5.2, rotation=90)
        ax.set_ylim(45, 103)
    axes[0].set_ylabel("测试准确率 (%)")
    fig.tight_layout()
    save(fig, "fig2_main")


# ---------------- 图 4：混淆矩阵 ----------------
def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.85))
    for ax, m, title in zip(axes, ["ce", "crt"], ["(a) CE (ERM)", "(b) cRT（两阶段）"]):
        p = sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"*ir100_{m}_s0.json")))[0]
        with open(p, encoding="utf-8") as f:
            cm = np.array(json.load(f)["metrics"]["confusion"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True) * 100
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_CN, rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(CLASS_CN, fontsize=6)
        ax.set_xlabel("预测类别")
        ax.set_title(title, fontsize=8.5)
        ax.grid(visible=False)
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 20:
                    ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=4.6,
                            color="white" if cm[i, j] > 55 else "#222222")
    axes[0].set_ylabel("真实类别")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("行归一化 (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig3_confusion")


# ---------------- 图 5：不平衡比敏感性 ----------------
def fig_ir(df):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
    for m in ["ce", "cbce", "crt"]:
        for ax, key, lab in zip(axes, ["acc", "few"], ["总体准确率 (%)", "尾部类准确率 (%)"]):
            xs, ys = [], []
            for ir in (10, 50, 100):
                sub = df[(df.ir == ir) & (df.method == m)]
                if len(sub):
                    xs.append(ir)
                    ys.append(sub[key].mean() * 100)
            ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.4,
                    color=COLORS[m], label=DISPLAY[m])
            ax.set_xscale("log")
            ax.set_xticks([10, 50, 100])
            ax.set_xticklabels(["10", "50", "100"])
            ax.set_xlabel("不平衡比 IR")
            ax.set_ylabel(lab)
    axes[0].set_title("(a) 总体准确率随 IR 的变化")
    axes[1].set_title("(b) 尾部类准确率随 IR 的变化")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig4_ir")


# ---------------- 图 5：机制分析 ----------------
def _load(method, ir=100, seed=0):
    pth = sorted(glob.glob(os.path.join(
        ROOT, "results", "runs", f"*_ir{ir}_{method}_s{seed}.json")))[0]
    with open(pth, encoding="utf-8") as f:
        return json.load(f)


def _mean_rho(method, ir=100):
    rhos = []
    for s in (0, 1, 2):
        try:
            r = _load(method, ir, s)
        except IndexError:
            continue
        counts = np.array([r["meta"]["class_counts_train"][str(c)] for c in range(10)])
        rhos.append(np.corrcoef(np.log(counts), np.array(r["weight_norms"]))[0, 1])
    return float(np.mean(rhos)) if rhos else float("nan")


def fig_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))

    # (a) 各类别召回率（3 个种子平均）
    ax = axes[0]
    for m in ["ce", "cbce", "crt"]:
        recs = []
        for s in (0, 1, 2):
            recs.append(_load(m, 100, s)["metrics"]["per_class_recall"])
        rec = np.mean(np.array(recs), axis=0) * 100
        ax.plot(range(10), rec, marker="o", markersize=3, linewidth=1.3,
                color=COLORS[m], label=DISPLAY[m])
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(c) for c in range(10)], fontsize=6)
    ax.set_xlabel("类别（0 = 头部类, 9 = 尾部类）")
    ax.set_ylabel("召回率 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) 各类别召回率")
    ax.legend(frameon=False, fontsize=6, loc="lower left")

    # (b) 权重范数 vs 类别频次（CB-CE 与 cRT：尾部表现相近但范数相关性相反）
    ax = axes[1]
    for m, mk in (("cbce", "o"), ("crt", "^")):
        r = _load(m, 100, 0)
        counts = np.array([r["meta"]["class_counts_train"][str(c)] for c in range(10)])
        wn = np.array(r["weight_norms"])
        ax.scatter(counts, wn, s=26, color=COLORS[m], marker=mk, edgecolor="white",
                   linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("类训练样本数（对数）")
    ax.set_ylabel("‖w_c‖")
    ax.set_title("(b) 权重范数与类别频次")

    # (c) t-SNE：CE 模型特征（cRT 冻结同一骨干，特征完全一致）
    ax = axes[2]
    from sklearn.manifold import TSNE
    z = np.load(os.path.join(ROOT, "results", "runs", "embed_ce_ir100_s0.npz"))
    X, y = z["feat"][:2500], z["label"][:2500]
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
               learning_rate="auto").fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=2.2, alpha=0.8, vmin=0, vmax=9)
    ax.set_title("(c) CE 特征 t-SNE")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, ticks=range(10))
    cb.ax.tick_params(labelsize=5)
    cb.set_label("类别", fontsize=6)
    fig.tight_layout()
    save(fig, "fig5_mechanism")


def main():
    df = load_runs()
    print(df.groupby(["ir"]).size())
    fig_distribution(df)
    fig_curves()
    fig_main(df)
    fig_confusion()
    if len(df[df.ir == 10]):
        fig_ir(df)
    fig_mechanism()


if __name__ == "__main__":
    main()
