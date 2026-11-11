"""汇总 results/runs 下的试验结果，输出 CSV 与报告用的关键数字。"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "results", "runs")
OUT_DIR = os.path.join(ROOT, "results")

DISPLAY = {
    "ce": "CE (ERM)",
    "focal": "Focal Loss",
    "cbce": "CB-CE",
    "oversample": "重采样 ROS",
    "la_loss": "LA (训练期)",
    "crt": "cRT (两阶段)",
    "ce_la": "CE + LA (后处理)",
}
ORDER = ["ce", "focal", "cbce", "oversample", "la_loss", "crt", "ce_la"]


def load_runs():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUN_DIR, "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        cfg, m = r["cfg"], r["metrics"]
        base = dict(ir=cfg["imbalance_ratio"], method=cfg["method"], seed=cfg["seed"],
                    acc=m["acc"], macro_f1=m["macro_f1"],
                    many=m["group_acc"]["many"], medium=m["group_acc"]["medium"],
                    few=m["group_acc"]["few"], ece=m["ece"],
                    n_train=r["meta"]["n_train"], time_sec=r["time_sec"],
                    params=r["params"])
        rows.append(base)
        if r.get("posthoc") and cfg["imbalance_ratio"] == 100:
            ph = r["posthoc"]
            rows.append(dict(base, method="ce_la", acc=ph["acc"],
                             macro_f1=ph["macro_f1"],
                             many=ph["group_acc"]["many"],
                             medium=ph["group_acc"]["medium"],
                             few=ph["group_acc"]["few"], ece=ph["ece"]))
    return pd.DataFrame(rows)


def summarize(df, ir=None):
    d = df if ir is None else df[df.ir == ir]
    g = d.groupby("method")
    out = pd.DataFrame({
        "acc_mean": g.acc.mean(), "acc_std": g.acc.std(ddof=0),
        "f1_mean": g.macro_f1.mean(), "f1_std": g.macro_f1.std(ddof=0),
        "many_mean": g.many.mean(), "medium_mean": g.medium.mean(),
        "few_mean": g.few.mean(), "few_std": g.few.std(ddof=0),
        "ece_mean": g.ece.mean(), "n_seeds": g.acc.count(),
    })
    out = out.reindex([m for m in ORDER if m in out.index])
    return out


def main():
    df = load_runs()
    df.to_csv(os.path.join(OUT_DIR, "runs_all.csv"), index=False, encoding="utf-8-sig")

    for ir in sorted(df.ir.unique()):
        s = summarize(df, ir)
        s.to_csv(os.path.join(OUT_DIR, f"summary_ir{ir}.csv"), encoding="utf-8-sig")
        if ir == 100:
            pd.set_option("display.width", 120)
            print(f"=== IR={ir} ===")
            cols = ["acc_mean", "acc_std", "f1_mean", "many_mean", "medium_mean",
                    "few_mean", "ece_mean"]
            print((s[cols] * 100).round(2).to_string())
            print("seeds per method:", dict(s["n_seeds"]))

    # 权重范数与类频的相关性（偏置诊断）
    corr = []
    for p in sorted(glob.glob(os.path.join(RUN_DIR, "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r["cfg"]["imbalance_ratio"] != 100:
            continue
        counts = np.array([r["meta"]["class_counts_train"][str(c)] for c in range(10)])
        wn = np.array(r["weight_norms"])
        corr.append(dict(method=r["cfg"]["method"], seed=r["cfg"]["seed"],
                         pearson=float(np.corrcoef(np.log(counts), wn)[0, 1]),
                         norm_ratio=float(wn.max() / wn.min())))
    if corr:
        cdf = pd.DataFrame(corr)
        cdf.to_csv(os.path.join(OUT_DIR, "weight_norm_corr.csv"), index=False,
                   encoding="utf-8-sig")
        print("\n=== 分类器权重范数 vs log(类样本数) 的相关系数 ===")
        print(cdf.groupby("method")[["pearson", "norm_ratio"]].mean().round(3))


if __name__ == "__main__":
    main()
