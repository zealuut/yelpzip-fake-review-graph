# LLM-Masked LogicAE-CB Fake Reviewer Detection 方法介绍

## 更新：图引导异常证据重加权

当前方法新增一个保守的 Graph-guided Abnormal Evidence Reweighting 模块。它的目的不是继续要求 LLM span 达到论文级逐字标注质量，而是把 LLM 输出视为候选异常证据，再用用户协同行为图判断某条 review 证据是否被邻居用户中的相似异常证据支持。

核心形式为：

```text
graph_support_{u,r}
= sum_{v in TopK(N(u))} edge_weight(u,v) * max_j cos(a_{u,r}, a_{v,j})

corrected_score_{u,r}
= 0.7 * evidence_score_{u,r} + 0.3 * graph_support_{u,r}
```

其中 `a_{u,r}` 是用户 `u` 的某条 review 异常向量，`a_{v,j}` 是邻居用户 `v` 的候选异常 review 向量。这个反馈发生在 `a_r -> a_u` 之间，只改变 review 聚合权重，不修改 token mask，不重新调用 LLM，也不多轮重建图。

正式流程变为：

```text
LLM span -> token mask -> abnormal encoder -> review vector a_r
    -> 初始 Top-M 得到 a_u^0
    -> 用 a_u^0 构造固定 UPU / UTU / USU / CB / LogicAE-CB
    -> review-level graph_support_{u,r}
    -> corrected_score 加权聚合得到 a_u'
    -> 固定边 + a_u' 节点特征 -> relation attention
```

这部分的意义是：普通用户偶然出现的夸张表达不会因为 LLM 误标而被强放大；如果一批用户在同餐厅、同时间、相似评分、相似异常表达上互相支持，对应 review 证据才会获得更高权重。

`LogicAE-CB` 建边采用“先过异常相似门槛，再按综合可靠度排序”的策略：

```text
S_logic >= tau_logic
tau_logic = max(0.30, quantile(S_logic_candidates, 0.60))
S_LogicAE_CB = 0.4*S_logic + 0.2*S_time + 0.2*S_rating + 0.2*S_product
```

其中 `S_LogicAE_CB` 只负责在已通过 `S_logic` 门槛的候选边中排序，不再额外设置综合边阈值。这样可以保证 `LogicAE-CB` 的核心语义仍然是异常证据相似，而不是被时间、评分或商品一致性单独拉高。

用户节点同时加入 5 个行为异常指标：`RD`、`EXR`、`MRO`、`AD`、`ATR`，并取平均得到 `behavior_anomaly_score`。这 5 个指标分别对应评分偏离、极端评分比例、单日最大评论数、账号活跃周期和评论间隔密集程度，用于补充用户级行为异常信号。

在 LLM cache 尚未完成时，可以运行 no-LLM fake extractor baseline。该基线不使用 LLM span，也不使用 LLM 数值特征，而是令异常编码器在整条评论 token 上工作，相当于测试“旧虚假提取器 + 用户行为图 + Graph Reweighting”本身的能力。这个结果不会替代最终 LLM-mask 主实验，但可以作为后续判断 LLM 是否带来增益的下界。

## 1. 研究定位

本方法面向 YelpZip 场景下的虚假评论者识别任务，目标不是单独判断某一条评论是否虚假，而是识别具有异常评论行为与协同行为模式的用户。整体思路是在已有用户行为多关系图的基础上，引入评论文本中的异常表达证据，使用户节点不仅包含行为统计特征，也包含由文本异常模式聚合而来的语义异常向量。

原始文本模型在直接做 review-level 分类时效果不理想，主要原因是虚假评论文本具有较强伪装性，单条评论的显性语义不足以稳定区分真假。因此，本方法不再把旧文本模型作为最终分类器，而是将其改造为“单条评论异常模式编码器”。最终的 fake reviewer 判断由用户级图模型完成。

## 2. 总体架构

整体流程可以概括为：

```text
YelpZip review data
    ↓
用户/餐厅活跃度过滤与矛盾标签用户剔除
    ↓
LLM 抽取单条评论异常片段与异常模式
    ↓
LLM span -> token mask
    ↓
旧文本架构改造为 LLM-Masked Logic Encoder
    ↓
生成 review-level abnormal vector a_r
    ↓
按用户聚合 Top-M 可疑评论，得到 user abnormal vector a_u
    ↓
构造 Base 行为关系图：UPU / UTU / USU
    ↓
构造 TextSim / CB / LogicAE-CB 增强关系
    ↓
relation attention 用户级图分类
    ↓
fake reviewer prediction
```

其中，核心新增贡献是：

```text
LLM abnormal evidence
    + old logic architecture
    + user-level abnormal vector
    + LogicAE-CB relation
```

也就是说，本方法不是替代行为图，而是在行为图底座上增加一类由 LLM 引导、旧结构编码的文本异常关系。

## 3. 数据预处理

数据预处理遵循虚假评论者识别中的用户级任务口径。首先删除非活跃评论者与非活跃餐厅，默认阈值为评论数少于 3 的用户和餐厅均被剔除。随后剔除标签存在矛盾的用户，即同一用户同时出现 fake 与 real 评论标签的情况。矛盾用户删除后，再次进行活跃度过滤，避免由于节点删除导致新的低活跃用户或餐厅残留。

预处理后的数据以 review 为基本记录，但训练和评估严格按照 user_id 划分 train、validation 和 test，避免同一用户的评论同时出现在训练集和测试集造成信息泄漏。

用户标签采用用户级定义：

```text
如果用户至少有一条 fake review，则 user_label = 1
否则 user_label = 0
```

该标签只用于监督训练与评估，不作为模型输入特征。

## 4. LLM 异常模式抽取

LLM 不直接判断评论真假，也不输出 fake/real。它只负责从单条评论中抽取可能的异常表达模式，包括：

```text
generic_promotion
generic_attack
lack_of_detail
template_like
exaggeration
overly_absolute
sentiment_rating_mismatch
inconsistent_claim
none
```

每条评论输入给 LLM 的信息仅包含：

```text
review_text
rating
```

LLM 输出结构化 JSON，包括异常类型、异常解释、证据片段、置信度以及若干数值评分：

```json
{
  "abnormal_patterns": [
    {
      "pattern_type": "template_like",
      "description": "The sentence sounds formulaic and lacks concrete details.",
      "evidence_span": "highly recommend this place",
      "confidence": 0.82
    }
  ],
  "sentiment": "positive",
  "specificity_score": 0.25,
  "template_score": 0.70,
  "exaggeration_score": 0.60,
  "experience_detail_score": 0.20,
  "claim_summary": "The review gives broad praise with limited concrete experience."
}
```

这里的关键约束是：`evidence_span` 必须来自原评论中的连续原文片段。这样后续才能将 LLM 标注对齐到 tokenizer 的 token 位置。

## 5. LLM Span 到 Token Mask

为了将 LLM 的异常证据接入旧文本结构，本方法不采用 proposition mask，而是采用 token mask。

原因是旧实验中 proposition 级切分容易受到句法解析、子句边界和命题划分误差影响。如果 LLM 输出的异常片段无法稳定映射到 proposition，会造成训练信号不稳定。相比之下，token mask 更直接，只需要将 LLM 给出的原文片段对齐到 tokenizer 的 offset mapping。

对齐流程为：

```text
LLM evidence_span
    ↓
在原 review_text 中查找字符区间
    ↓
根据 tokenizer offset_mapping 找到重叠 token
    ↓
将这些 token 的 mask 值设为 LLM confidence
```

如果多个 span 覆盖同一个 token，则取最大 confidence。如果 span 找不到，则记录到诊断文件，不让程序崩溃。正式实验会统计 span 匹配比例，用于判断 LLM 标注质量。

因此，本方法是：

```text
LLM span -> token mask
```

不是：

```text
LLM span -> proposition mask
```

## 6. 旧文本结构的改造

旧结构原本用于 review-level 分类，核心包含：

```text
RoBERTa / SKEP Encoder
Logic Tower
Cross-Attention Fusion
Dynamic Gated Fusion
Final Classifier
```

本方法保留其“逻辑编码 + 交叉注意力融合”的主体思想，但改变其任务角色。旧结构不再直接承担最终真假判断，而是被改造成 review-level abnormal pattern encoder。

新的输入包括：

```text
input_ids
attention_mask
abnormal_token_mask
LLM numeric features
review_label
```

模型首先通过 RoBERTa 或 SKEP 得到 token hidden states：

```text
H = Encoder(input_ids, attention_mask)
```

然后利用 LLM token mask 生成异常区域表示：

```text
H_abn = H * abnormal_token_mask
```

`H_abn` 输入 BiLSTM Logic Tower，得到异常逻辑查询向量 `q_r`。随后以 `q_r` 为 Query，原始 token states `H` 为 Key 和 Value 做 cross-attention，使模型能够从异常片段回看完整评论上下文，得到上下文证据向量 `c_r`。

最终拼接：

```text
[CLS_r, q_r, c_r, LLM numeric features, optional secondary prior]
```

经过 bottleneck MLP 得到单条评论的异常模式向量：

```text
a_r
```

辅助分类头输出：

```text
p_fake_review
```

其中 `p_fake_review` 只用于辅助训练和选择用户的高可疑评论，真正进入用户图的核心表示是 `a_r`。

## 7. 用户异常向量聚合

一个用户通常有多条评论。并不是所有评论都同等重要，虚假评论者的异常性往往集中体现在少数高可疑评论上。因此，本方法对每个用户选择 Top-M 条最可疑评论进行聚合。

评论可疑分数为：

```text
evidence_score = 0.5 * p_fake_review
               + 0.2 * template_score
               + 0.2 * exaggeration_score
               + 0.1 * (1 - specificity_score)
```

默认取：

```text
top_m = 3
```

用户异常向量定义为：

```text
a_u = mean(top_m a_r)
```

这样得到的 `a_u` 表示该用户在文本异常模式上的整体倾向。

## 8. 用户节点特征

用户节点特征由两部分组成：

```text
行为统计特征
用户异常向量 a_u
```

行为统计特征包括评论数量、平均评分、评分标准差、评分熵、评分偏离、极端评分比例、爆发比例、活跃天数、账号活跃跨度、平均评论间隔、评论时间滞后、平均评论长度等。

这些特征的设计依据是：虚假评论者可以模仿单条评论文本，但更难长期稳定地模仿真实用户的评分习惯、时间节律、活跃跨度和目标餐厅选择。

节点初始表示可以写作：

```text
x_u = concat(behavior_features_u, a_u)
```

## 9. 多关系用户图

本方法以用户为节点构建多关系图。基础行为关系包括：

```text
UPU: 两个用户评论过同一餐厅或商品
UTU: 两个用户在相同时间桶内发表评论
USU: 两个用户具有相似或突出的爆发式评论行为
```

这些关系对应虚假评论者常见的协同行为：

```text
共同攻击或推广同一对象
集中时间段内活动
任务驱动下的爆发式发布
```

在此基础上，本方法增加三类增强关系：

```text
TextSim: 用户评论文本整体语义相似
CB: 文本相似 + 时间/评分/商品一致性
LogicAE-CB: LLM-LogicAE 异常向量相似 + 时间/评分/商品一致性
```

其中最关键的是 LogicAE-CB。

## 10. LogicAE-CB 关系

LogicAE-CB 用于刻画两个用户是否具有相似的异常表达模式，并且在行为上也存在一致性。

首先计算用户异常向量相似度：

```text
S_logic = cosine(a_u_i, a_u_j)
```

然后计算行为一致性：

```text
S_time    = 是否共享时间桶
S_rating  = 平均评分是否接近或评分倾向一致
S_product = 是否评论过相同商品或餐厅
```

最终关系权重为：

```text
S_LogicAE_CB = 0.4 * S_logic
             + 0.2 * S_time
             + 0.2 * S_rating
             + 0.2 * S_product
```

该关系的含义是：如果两个用户不仅行为上接近，而且其异常文本模式也相似，那么他们更可能处在同类虚假评论策略或协同行为中。

## 11. 用户级关系注意力分类

最终分类阶段不是直接对单条评论做判断，而是在用户图上进行节点分类。

对每种关系，先聚合邻居用户特征：

```text
h_u^r = weighted_mean({x_v | v in N_r(u)})
```

其中 `r` 表示关系类型，如 UPU、UTU、USU、TextSim、CB、LogicAE-CB。

随后将用户自身特征和各关系聚合特征组成多个 feature blocks：

```text
[x_u, h_u^UPU, h_u^UTU, h_u^USU, h_u^LogicAE-CB, ...]
```

relation attention 学习不同关系对当前用户的贡献权重：

```text
α_r = softmax(score(h_u^r))
```

最终用户表示为：

```text
z_u = concat(x_u, Σ α_r h_u^r)
```

分类器输出：

```text
p_fake_user = classifier(z_u)
```

这种设计保留了多关系行为图的核心思想，同时避免引入过重的异构图强化学习或复杂 GNN，使最终实验更稳、更容易在单卡服务器上运行。

## 12. 实验对比口径

本方法的关键实验不是证明 LLM 能直接判断真假，而是验证 LLM-LogicAE 生成的异常关系是否能增强用户图检测。

默认比较：

```text
MLP_no_graph
Base = UPU + UTU + USU
Base_TextSim = Base + TextSim
Base_CB = Base + CB
Base_LogicAE_CB = Base + LogicAE-CB
Full = Base + TextSim + CB + LogicAE-CB
```

核心观察是：

```text
Base_LogicAE_CB 是否优于 Base、Base_TextSim、Base_CB
```

如果 LogicAE-CB 提升明显，说明 LLM 引导的异常文本模式能够补充传统行为关系。如果提升不明显，也可以通过边质量统计判断是 LLM span 抽取问题、mask 对齐问题、异常向量质量问题，还是图关系本身贡献有限。

## 13. 与相关工作的关系

与纯文本分类方法相比，本方法不依赖单条评论的真假判断，而是将文本异常信息上升到用户级和关系级。

与传统行为图方法相比，本方法不仅使用共评论、共时间和爆发行为，还引入了由 LLM 和旧逻辑结构编码得到的异常表达相似性。

与复杂异构图或强化学习图模型相比，本方法没有直接引入过重的新图网络，而是采用轻量 relation attention 做关系融合，降低一次性实验的工程风险。

因此，本方法的定位是：

```text
行为图是主干
LLM 异常证据是增强信号
旧文本结构是异常证据编码器
LogicAE-CB 是连接文本异常与用户协同行为的桥
```

## 14. 方法优势与限制

主要优势：

```text
1. 保留用户级 fake reviewer detection 口径，不退化为 review classification
2. 利用 LLM 提供细粒度异常证据，但不让 LLM 直接做最终判断
3. 复用旧文本模型的 logic tower 与 cross-attention，降低结构推倒重来的风险
4. 将评论级异常向量聚合到用户级，适配 YelpZip 用户图任务
5. 通过 LogicAE-CB 将文本异常模式转化为用户关系边
6. 与 Base/TextSim/CB 可形成清晰消融对比
```

主要限制：

```text
1. LLM span 抽取质量会影响 token mask 和异常向量质量
2. 本地 T5 补齐标注虽然省钱，但可能弱于全量大模型标注
3. LogicAE-CB 的增益依赖于异常向量是否能捕捉稳定的虚假表达模式
4. relation attention 是轻量图模型，不等同于完整异构 GNN
5. 当前方法重点验证增益方向，不追求复现复杂图模型的最高绝对指标
```

## 15. 一句话总结

本方法是在用户行为多关系图的基础上，引入 LLM 抽取的异常文本证据，并通过旧逻辑文本结构将其编码为用户级异常向量，再构造 LogicAE-CB 关系边，使虚假评论者识别同时利用个体行为、文本异常和群体协同结构。
