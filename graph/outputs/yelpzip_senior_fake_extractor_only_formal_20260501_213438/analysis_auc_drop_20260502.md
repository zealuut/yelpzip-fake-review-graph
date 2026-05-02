# Senior-Protocol Run AUC Drop Analysis

分析对象：

- 本次结果目录：`graph/outputs/yelpzip_senior_fake_extractor_only_formal_20260501_213438`
- 上一次对照目录：`graph/outputs/yelpzip_fake_extractor_only`
- 关注指标：以 AUC 为主，同时参考 AP、边同质性、数据规模和配置差异。

## 1. 结论先行

这次 AUC 下降不是单纯说明“师兄底座不如旧版”，而是由三个问题叠加造成的：

1. 本次和上一次不是同一实验口径：本次是 1:1 平衡用户集，上一次是完整非平衡用户集。
2. 本次目录虽然叫 fake extractor only，但实际启用了 SKEP 双塔，上一次没有启用 SKEP。
3. 最关键：`graph_mode=senior` 虽然写进了配置，但输出的 `UPU_edges.csv` 和 `UTU_edges.csv` 实际仍是 top-k 截断后的旧式关系图，不是师兄论文那种完整 UPU/UTU 图。

其中第 3 点是主问题。当前结果不能作为“复刻师兄底座”的有效结果，也不能据此判断师兄图底座是否无效。

## 2. 指标对比

| Run | 数据口径 | 用户数 | 测试集 fake 用户 | 最佳 AUC | 最佳 AP | 最佳 edge set |
|---|---:|---:|---:|---:|---:|---|
| 上一次 `yelpzip_fake_extractor_only` | 非平衡全量 | 48558 | 500 / 7284 | 0.8441 | 0.3000 | Base_LogicAE_CB |
| 本次 `senior_fake_extractor_only` | 1:1 平衡 | 6664 | 667 / 1334 | 0.8232 | 0.8143 | Full |

不能直接把 AP、Precision、Accuracy 横向比较，因为测试集 fake 比例从约 6.86% 变成 50%。AUC 可以比较，但也要注意负样本集合变了：上一次有大量容易区分的正常用户，本次只抽了 3332 个正常用户与全部 fake 用户配平，pairwise ranking 难度会变化。

本次最佳结果：

```text
llm_masked_logic_graph_reweighted + Full
AUC = 0.823196
AP  = 0.814256
F1  = 0.752909
```

但本次 `MLP_no_graph` 在 graph-reweighted 表示上已经有：

```text
AUC = 0.823176
```

也就是说，最佳 Full 图只比 no-graph 高约 `0.00002`。这说明这次图关系几乎没有真正贡献，主要性能来自用户自身表示。

## 3. 关键问题：UPU/UTU 没有真正按师兄图落地

本次配置写的是：

```json
"senior_protocol": true,
"graph_mode": "senior",
"balance_user_labels": true
```

但实际边文件显示：

```text
UPU_edges.csv header:
src_user_id,dst_user_id,edge_type,edge_weight,shared_entity_count

UTU_edges.csv header:
src_user_id,dst_user_id,edge_type,edge_weight,shared_entity_count
```

而且实际边数为：

| Edge | 实际 directed rows | 折算 undirected | avg degree |
|---|---:|---:|---:|
| UPU | 129868 | 64934 | 19.49 |
| UTU | 132614 | 66307 | 19.91 |
| USU | 444222 | 222111 | 666.00 |

`UPU/UTU` 的 avg degree 接近 20，说明它们仍然被 `top_k=20` 截断了。真正的师兄 UPU/UTU 不应该是这个量级。

我用本次 `review_scores_enriched.csv` 重新估算了不截断的 senior 图：

| Edge | 当前数据可得到的 collapsed directed pairs | 折算 undirected | 师兄论文边数 |
|---|---:|---:|---:|
| UPU | 1827992 | 913996 | 940686 |
| UTU | 2081272 | 1040636 | 1054600 |
| USU | 444222 | 222111 | 227475 |

这个结果非常关键：在当前平衡数据上，只要 UPU/UTU 不被 top-k 截断，边数本来几乎可以对上师兄论文。现在 AUC 低，很大程度是因为我们以为跑了 senior UPU/UTU，实际上跑的是 top-k 稀疏版。

## 4. 图为什么没有带来收益

本次各边同质性如下：

| Edge | fake-fake ratio | fake-real ratio | real-real ratio | 说明 |
|---|---:|---:|---:|---|
| USU | 0.6997 | 0.2738 | 0.0265 | 很强，但只覆盖 top 10% burst 用户，5997 个用户无 USU 出边 |
| UPU | 0.2920 | 0.4717 | 0.2364 | 由于 top-k 截断，结构信息不足且混入大量异类边 |
| UTU | 0.3438 | 0.4950 | 0.1612 | 同样被 top-k 截断，fake-real 边接近一半 |
| LogicAE_CB | 0.3552 | 0.3132 | 0.3317 | 有一定信号，但向量相似度过饱和 |
| GraphSupport | 0.3422 | 0.2954 | 0.3625 | 没有形成明显 fake 聚团 |

USU 看起来很强，但它只覆盖 667 个 burst 用户，约 90% 用户没有 USU 出边。因此如果 UPU/UTU 没有完整铺开，整体图模型无法得到师兄论文中那种多关系覆盖效果。

## 5. Graph Reweighting 在 no-LLM/full-text 模式下仍然偏饱和

本次 review-level 分数统计：

| Score | review-level AUC | 说明 |
|---|---:|---|
| `p_fake_review` | 0.8561 | 文本提取器本身并不差 |
| `evidence_score` | 0.8561 | full-text 模式下基本等价于 review classifier 分数 |
| `graph_support_score` | 0.6616 | 有弱信号，但不够干净 |
| `corrected_evidence_score` | 0.8534 | 比原始 evidence 略降 |

`graph_support_score` 均值很高：

```text
real review: 0.9880
fake review: 0.9945
```

这说明在 no-LLM/full-text 模式下，review abnormal vectors 仍然高度相似，图支持分数接近饱和。它不会严重毁掉 AUC，但也不能提供干净排序信号。Graph Reweighting 应作为 ablation，而不是当前 strong backbone 的默认必开模块。

## 6. 另一个混杂因素：本次启用了 SKEP

上一次配置：

```json
"secondary_model_name_or_path": null
```

本次配置：

```json
"secondary_model_name_or_path": ".../pretrain_model/skep-base"
```

所以本次并不是纯粹的“平衡数据 + 师兄图”对照，也不是严格的 fake extractor only。好消息是，本次 review-level `p_fake_review` AUC 达到 0.8561，高于上次 full-text review-level 约 0.7092，说明文本提取器没有崩。用户级 AUC 没上去，主要问题更像是图口径和实验对照混杂，而不是文本编码器本身完全失败。

## 7. 下一步建议

优先级 1：修正 senior UPU/UTU 建图。

- `UPU`：不要 top-k 截断；按共同餐厅/商品连完整用户对。
- `UTU`：不要 top-k 截断；按同一年同一周连完整用户对。
- `USU`：当前边数已经基本对上师兄论文，可以保留。
- 修正后检查边数，应接近：
  - UPU undirected: 913996 左右
  - UTU undirected: 1040636 左右
  - USU undirected: 222111 左右

优先级 2：做受控对照，不要一次混多个变量。

建议按这个顺序跑：

```text
A. balanced + fixed senior graph + no SKEP + no Graph Reweighting
B. balanced + fixed senior graph + SKEP + no Graph Reweighting
C. balanced + fixed senior graph + SKEP + LogicAE_CB
D. balanced + fixed senior graph + SKEP + Graph Reweighting
```

优先级 3：Graph Reweighting 在 no-LLM 模式先默认关闭。

原因是 full-text mask 会让异常向量相似度过高，`graph_support_score` 饱和。等 LLM mask 或更稀疏的异常 evidence 进入后，再评估它是否真正提高鲁棒性。

## 8. 最终判断

这次 AUC 低的主因不是“师兄图底座不行”，而是当前输出没有真正跑出师兄 UPU/UTU 的完整图。更准确地说：

```text
本次跑到的是：
balanced users + SKEP + top-k UPU/UTU + senior-like USU + graph reweighting

不是：
balanced users + senior full UPU/UTU/USU backbone
```

因此当前 `0.8232` 不能作为最终结论。先修正 UPU/UTU 完整建图，再重跑 senior backbone，才有资格和师兄论文的 AUC 0.8703 进行基线差距分析。
