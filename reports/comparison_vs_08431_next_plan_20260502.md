# Comparison With Previous 0.8431 AUC Run And Next Plan

对比对象：

- 上一次 `0.8431 AUC` 版本：`graph/outputs/yelpzip_fake_extractor_only`
- 本次 clean senior 版本：`graph/outputs/yelpzip_senior_backbone_clean_formal_20260502_101931`
- 本文只做结果分析和下一步实验方案，不涉及代码层面修改。

## 1. 总体结论

本次确实比上一次差，而且差距不是单点随机波动。

如果按你说的 `0.8431` 版本对比：

```text
上一次:
llm_masked_logic_initial + Full
AUC = 0.843126

本次:
llm_masked_logic + Full
AUC = 0.822576

下降:
-0.020550 AUC
```

如果按上一次目录里的最佳结果对比：

```text
上一次最佳:
llm_masked_logic_graph_reweighted + Base_LogicAE_CB
AUC = 0.844066

本次最佳:
llm_masked_logic + Full
AUC = 0.822576

下降:
-0.021491 AUC
```

这次下降主要不是因为 LLM，也不是因为 SKEP，因为两版都是 `mask_source=full_text`，且本次 clean 版已经关闭 SKEP 和 Graph Reweighting。真正原因是：

```text
数据口径变难 + full senior UPU/UTU 边太噪 + 当前 mean aggregation 图模型抗噪不足。
```

## 2. 两次实验配置差异

| 项目 | 上一次 0.8431 版本 | 本次 clean senior 版本 |
|---|---:|---:|
| output | `yelpzip_fake_extractor_only` | `yelpzip_senior_backbone_clean_formal_20260502_101931` |
| 数据口径 | 非平衡全量用户 | 1:1 平衡用户 |
| 用户数 | 48558 | 6664 |
| review 数 | 348243 | 39667 |
| split | 0.70 / 0.15 / 0.15 | 0.64 / 0.16 / 0.20 |
| test fake 用户 | 500 / 7284 | 667 / 1334 |
| fake 用户比例 | 6.86% | 50.00% |
| graph mode | old current top-k reliability graph | senior full UPU/UTU/USU |
| Graph Reweighting | enabled，但 0.8431 是 initial Full | disabled |
| SKEP | disabled | disabled |

所以这两个结果不是严格同口径对照。AUC 可以参考，但不能把下降全部归因到模型结构。

## 3. 指标拆解

### 3.1 上一次 0.8431 版本

| Edge set | AUC | AP |
|---|---:|---:|
| MLP_no_graph | 0.828716 | 0.277440 |
| Base | 0.841596 | 0.318538 |
| Base_LogicAE_CB | 0.843944 | 0.326993 |
| Full | 0.843126 | 0.302035 |

关键点：

```text
Full - MLP_no_graph = +0.014410 AUC
Base - MLP_no_graph = +0.012880 AUC
```

上一次图结构确实带来了明显增益。

### 3.2 本次 clean senior 版本

| Edge set | AUC | AP |
|---|---:|---:|
| MLP_no_graph | 0.812890 | 0.800352 |
| Base | 0.809233 | 0.807422 |
| Base_TextSim | 0.814864 | 0.810321 |
| Base_CB | 0.815898 | 0.811471 |
| Base_LogicAE_CB | 0.810377 | 0.807588 |
| Full | 0.822576 | 0.819364 |

关键点：

```text
Full - MLP_no_graph = +0.009686 AUC
Base - MLP_no_graph = -0.003657 AUC
```

本次 `Base = UPU + UTU + USU` 反而低于 no-graph，说明 full senior 行为图没有被当前聚合器有效利用。

## 4. 为什么本次更差

### 4.1 数据口径改变让 no-graph 自身下降

上一次 no-graph：

```text
AUC = 0.828716
```

本次 no-graph：

```text
AUC = 0.812890
```

仅 no-graph 就下降：

```text
-0.015826 AUC
```

这说明一部分下降来自数据口径变化：从全量非平衡用户变成 1:1 平衡用户后，负样本集合变了，测试任务也变了。上一次测试集有大量普通正常用户，本次正常用户是从全量正常用户中抽样出来与 fake 用户配平，区分难度和分布都不一样。

### 4.2 full senior UPU/UTU 边完整，但噪声大

本次 senior 边数已经对齐：

| Edge | directed edges | undirected estimate | avg degree |
|---|---:|---:|---:|
| UPU | 1827992 | 913996 | 274.35 |
| UTU | 2081272 | 1040636 | 312.41 |
| USU | 444222 | 222111 | 666.00 |

但边质量不理想：

| Edge | same-label ratio | P(dst fake | src fake) | P(dst real | src real) |
|---|---:|---:|---:|
| UPU | 0.5349 | 0.4085 | 0.6167 |
| UTU | 0.5469 | 0.3310 | 0.6575 |
| USU | 0.7262 | 0.8363 | 0.1622 |
| TextSim / CB | 0.7570 | 0.7297 | 0.7843 |
| LogicAE_CB | 0.6772 | 0.6556 | 0.6983 |

UPU 和 UTU 是完整图，但完整不等于有效。对 fake 用户来说：

```text
UPU: fake 用户连到 fake 的概率只有 0.4085
UTU: fake 用户连到 fake 的概率只有 0.3310
```

在 1:1 平衡用户集里，随机连边的 fake 概率约是 0.5。因此 UPU/UTU 对 fake 传播不是正信号，反而偏噪。

### 4.3 当前图模型是 mean aggregation，不是师兄论文的 GAT

这是最核心的结构原因。

师兄论文有效的不是单纯“UPU/UTU/USU 完整边”，而是：

```text
行为特征 + 多关系图 + GAT / 关系注意力 / 邻居筛选
```

我们当前实现更接近：

```text
用户自特征
+ 每种关系的邻居特征加权均值
+ relation-level attention 分类器
```

也就是说，我们现在有 relation-level attention，但没有真正的 neighbor-level attention。面对 UPU/UTU 这种百万级噪声边，均值聚合会把大量正常用户和虚假用户混在一起，导致用户表示被冲淡。

这解释了为什么：

```text
Base = UPU + UTU + USU
AUC 0.8092

MLP_no_graph
AUC 0.8129
```

Base 不但没帮忙，还略微拖低。

### 4.4 上一次旧图为什么反而更好

上一次虽然不是 senior full graph，但它使用的是更保守的 top-k reliability graph。它相当于先过滤邻居，再聚合：

```text
旧版 current graph:
UPU/UTU/USU 都控制 top-k，边更稀疏，噪声更少。
```

这和当前 mean aggregation 更匹配。因此旧版能达到：

```text
Base AUC = 0.8416
Full AUC = 0.8431
```

而本次 full senior graph 是：

```text
先把所有 UPU/UTU 邻居放进来，再简单平均。
```

这对当前模型是不友好的。

### 4.5 行为统计本身比当前 Full 图还强

本次我额外看了用户行为信号：

| Signal | AUC | AP |
|---|---:|---:|
| behavior_anomaly_score | 0.758146 | 0.751814 |
| LR on 5 behavior indicators | 0.801121 | 0.784440 |
| LR on behavior stats | 0.830980 | 0.823086 |
| 当前 Full 图 | 0.822576 | 0.819364 |

这说明：

```text
当前图模型没有充分利用行为统计。
甚至一个简单 behavior_stats Logistic Regression 都高于 Full 图。
```

所以问题不是数据里没有行为信号，而是当前图聚合把行为信号用差了。

## 5. 最终原因排序

按影响大小排序：

1. full senior UPU/UTU 对当前 mean aggregation 太噪，Base 直接低于 no-graph。
2. 当前图模型不是师兄 GAT，缺少 neighbor-level attention / 邻居筛选。
3. 平衡数据口径让 no-graph 自身从 0.8287 降到 0.8129。
4. LogicAE_CB 在 no-LLM/full-text 模式下仍然受异常向量相似度饱和影响，不能弥补 Base 噪声。
5. Graph Reweighting 这次已关闭，所以它不是本次差的主因。

一句话总结：

```text
上一次高，是因为 top-k reliability graph 和 mean aggregation 匹配；
这次低，是因为 full senior graph 需要 GAT/邻居注意力，而我们当前只是邻居均值。
```

## 6. 下一步方案

### 方案 A：最快止损，回到“平衡数据 + old current top-k graph”

目的：确认在同样 1:1 平衡数据上，旧版 top-k reliability graph 是否能恢复到接近 0.84。

这一步最重要，因为它能隔离变量：

```text
如果 balanced + current graph 明显高于 0.8226，
说明问题主要是 senior full graph 不适配 mean aggregation。
```

建议运行：

```bash
export OUTPUT_DIR=graph/outputs/yelpzip_balanced_current_graph_no_reweight
export FAKE_EXTRACTOR_ONLY=1
export RUN_LLM_CACHE=0
export BALANCE_USER_LABELS=1
export BALANCED_USER_COUNT=6742
export GRAPH_MODE=current
export DISABLE_GRAPH_REWEIGHTING=1

bash graph/run_all.sh \
  --train_ratio 0.64 \
  --val_ratio 0.16 \
  --test_ratio 0.20
```

注意：这里不要开 `SENIOR_PROTOCOL=1`，因为该开关会默认把 `graph_mode` 改成 `senior`。我们只要平衡采样和 6.4/1.6/2.0 划分，不要 senior full graph。

预期判断：

- 如果 AUC 回到 0.835 附近或更高：放弃 full UPU/UTU mean aggregation，使用 top-k reliability graph 做工程底座。
- 如果仍然只有 0.82 左右：说明主要是平衡数据口径下文本/用户表示本身不足，需要加强用户特征。

### 方案 B：把行为统计 baseline 正式加入主表

本次 `behavior_stats LR` 已经有：

```text
AUC = 0.830980
AP  = 0.823086
```

它应该进入正式实验表，作为强行为 baseline。否则我们会错误地认为图模型已经超过行为特征，实际上没有。

建议主表至少包含：

```text
Behavior-LR
Behavior-MLP
Text/User Vector MLP
Current Top-k Graph
Senior Full Graph MeanAgg
```

### 方案 C：如果坚持师兄底座，必须补 GAT/邻居注意力

如果目标是“以师兄底座为骨架”，那不能只复刻 full UPU/UTU/USU 边，还要补模型侧的邻居选择能力。

最低要求：

```text
relation-specific GAT
或
per-relation neighbor attention
或
top-k attention sampling before aggregation
```

否则 full UPU/UTU 会持续把表示平均糊掉。

### 方案 D：LLM/LogicAE 暂时不要作为主要救火点

当前 full-text review extractor 在本次平衡数据上 review-level AUC 已经有：

```text
p_fake_review AUC = 0.852307
```

所以短期瓶颈不是“review 编码器完全没信号”，而是 user-level graph aggregation。LLM mask 后面仍有意义，但现在先救图底座。

## 7. 推荐的下一轮实验顺序

我建议下一轮只跑三个，不要再一次混太多变量：

```text
Run 1:
balanced + current top-k graph + no SKEP + no Graph Reweighting

Run 2:
balanced + behavior_stats LR / MLP

Run 3:
balanced + senior full graph + neighbor attention/GAT
```

如果 Run 1 明显好于本次 clean senior，那么最终工程底座就用 current top-k graph，不硬贴 full senior。师兄论文结果仍可作为外部基线引用，我们的方法强调“在多关系行为图上引入异常证据关系”，不需要机械复刻 full UPU/UTU。

如果 Run 3 做出来后超过 Run 1，再考虑把 senior full graph 作为主底座。

## 8. 最终建议

当前不要继续围绕这版 `0.8226` 微调。

它已经说明：

```text
full senior graph + mean aggregation 不是我们的好底座。
```

下一步最稳的是：

```text
先回到 balanced + current top-k reliability graph，
确认同口径下能不能恢复 0.84 附近；
同时把 behavior_stats baseline 纳入主表；
再决定是否值得实现真正 GAT。
```

这比继续调 LLM、调 LogicAE-CB、调 graph reweighting 更优先。
