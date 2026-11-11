# 长尾分布下的图像分类：七种不平衡学习策略的受控对比与机制分析

深度学习课程设计（2026 秋季）配套代码与实验产物。全部实验在**纯 CPU** 环境完成，
一次完整复现约 2 小时。

## 研究问题

在**完全相同的骨干网络、数据划分、优化器与训练预算**下，横向比较长尾分类中六条技术路线
（重采样 / 损失重加权 / 样本重加权 / 决策规则调整 / 训练流程解耦 / 推理期后处理）的收益，
并回答"收益来自特征表征还是分类器偏置"。

## 目录结构

```
data/            数据集（Fashion-MNIST 原始 idx 文件）
src/
  common.py      路径、随机种子、JSON 落盘
  data.py        指数衰减长尾构造、均衡子集、批量张量化数据增强
  models.py      三段式 CNN 骨干（297K 参数）
  losses.py      Focal Loss / CB-CE / 训练期 Logit Adjustment / 后处理偏置
  engine.py      训练循环、分组指标、宏 F1、ECE、权重范数
  runner.py      单次实验编排（含 cRT 两阶段）
scripts/
  run_experiments.py  24 组实验统一调度（支持分片）
  aggregate.py        结果聚合为 CSV 与汇总表
  make_figures.py     生成报告图表
  build_docx.py       报告 Markdown → 排版好的 DOCX
results/         每次实验的 JSON（逐轮损失/准确率、测试指标、混淆矩阵、权重范数）
figures/         报告用图
build/           build/report.md（报告源文）
```

## 复现

```bash
python -m pip install -r requirements.txt

python scripts/run_experiments.py all 0 1      # 主实验 18 组 + IR 消融 6 组
python scripts/aggregate.py                    # 生成 results/summary_ir*.csv
python scripts/make_figures.py                 # 生成 figures/*.png|pdf
python scripts/build_docx.py "2026439182-雷东曦.docx"       # 由 build/report.md 重建报告
```

数据集不随仓库分发（`data/` 已在 `.gitignore` 中忽略）。首次运行时
`torchvision.datasets.FashionMNIST(root="data/raw", download=True)` 会自动下载；
若网络受限，也可手动下载以下 4 个文件放入 `data/raw/FashionMNIST/raw/`：

```
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz
```

## 实验设定

| 项目 | 取值 |
|---|---|
| 数据集 | Fashion-MNIST（10 类，28×28，训练 60k / 测试 10k） |
| 长尾构造 | 指数衰减 n_i = n_max·(1/IR)^(i/(C−1))，IR=100 时 5700→54 |
| 训练集规模 | 14147（含 5% 监控验证集） |
| 骨干 | 3×(Conv3×3+BN+ReLU)×2 + MaxPool，GAP + FC，297,450 参数 |
| 优化 | AdamW，lr 1e-3，wd 5e-4，余弦退火，bs 512，10 epoch |
| 增强 | 随机裁剪(padding=4, reflect) + 随机水平翻转 |
| 随机种子 | 主实验 3 个（0/1/2），IR 消融 1 个 |
| 评价 | 总体 Acc、宏 F1、头/中/尾三组 Acc、ECE(15 bins) |

## 方法清单

| 名称 | 类型 | 关键超参 |
|---|---|---|
| CE (ERM) | 基线 | — |
| Focal Loss | 样本重加权 | γ=2 |
| CB-CE | 损失重加权 | β=0.9999 |
| ROS | 数据重采样 | 逆频率采样 |
| LA (训练期) | 决策规则调整 | τ=1 |
| cRT | 两阶段解耦 | 阶段二仅重训分类器，lr 1e-3，10 epoch |
| CE + LA (后处理) | 推理期修正 | τ=1，零训练成本 |

## 致谢与说明

方法均实现自公开论文，出处见报告参考文献。数据集来自 Fashion-MNIST 官方发布
（Xiao et al., arXiv:1708.07747）。
