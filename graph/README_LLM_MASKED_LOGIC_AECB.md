# LLM-Masked LogicAE-CB

## New: Senior-Protocol Balanced Backbone

为了解决当前结果和师兄论文口径不一致的问题，新增 `SENIOR_PROTOCOL=1` 运行模式。该模式默认执行：

- 用户级 1:1 平衡采样，默认请求 `6742` 个用户；如果预处理后某一类用户不足，会自动使用可达到的最大平衡规模。
- 用户划分改为师兄论文口径：`train / val / test = 0.64 / 0.16 / 0.20`。
- 基础行为图改为师兄定义：`UPU` 为共同餐厅，`UTU` 为同一年同一周，`USU` 为 burst ratio 前 `10%` 用户完全连接。
- 输出 `edges/edge_build_config.json` 会记录 `graph_mode`、实际边数和平衡采样信息，方便和论文基线对齐。

推荐先跑无 LLM 的强底座：

```bash
export SENIOR_PROTOCOL=1
export FAKE_EXTRACTOR_ONLY=1
export RUN_LLM_CACHE=0
bash graph/run_all.sh
```

默认输出目录：

```text
graph/outputs/yelpzip_senior_fake_extractor_only
```

如果后续切回 LLM mask，`SENIOR_PROTOCOL=1` 会默认使用独立缓存：

```text
graph/outputs/llm_cache/yelpzip_senior_llm_abnormal_patterns.jsonl
```

如果只想使用师兄口径的平衡数据和原始三类边，保持默认 `GRAPH_MODE=senior` 即可。若要尝试我们前面讨论的 TNSGD 风格时间可靠度增强，可以显式打开：

```bash
export GRAPH_MODE=senior_enhanced
```

注意：`senior` 模式为了模型聚合会保存双向有向边，因此 CSV 行数通常约等于论文无向边数的 2 倍；配置文件里会额外写入 `senior_undirected_pair_estimates` 供对照。

## New: Graph-guided Abnormal Evidence Reweighting

当前版本默认启用一个保守的图引导异常证据重加权模块，目标是处理 LLM span 语义质量中等、`exaggeration/lack_of_detail` 容易泛化的问题。

它不修改 token mask，不让 LLM 重新生成，也不重建多轮动态图。流程是：

```text
review abnormal vector a_r
    -> 初始 evidence_score
    -> 初始 Top-M 聚合得到 a_u^0
    -> 用 a_u^0 构造固定 UPU / UTU / USU / CB / LogicAE-CB 图
    -> 对每条 review 计算 graph_support_{u,r}
    -> corrected_score_{u,r} = 0.7 * evidence_score_{u,r} + 0.3 * graph_support_{u,r}
    -> 用 corrected_score 重新加权聚合得到 a_u'
    -> 固定边 + a_u' 节点特征进入 relation attention
```

其中 review 级图支持分数定义为：

```text
graph_support_{u,r}
= sum_{v in TopK(N(u))} edge_weight(u,v) * max_j cos(a_{u,r}, a_{v,j})
```

这意味着：某条异常评论如果能在邻居用户的异常评论集合中找到相似证据，就上调；如果只是 LLM 对普通夸张词的孤立误标，就不会被图结构强支持。

新增输出包括：

- `graph/outputs/.../edges/GraphSupport_edges.csv`
- `graph/outputs/.../logic_vectors/review_graph_reweight_scores.csv`
- `graph/outputs/.../logic_vectors/user_abnormal_vectors_initial.npy`
- `graph/outputs/.../logic_vectors/user_abnormal_vectors_graph_reweighted.npy`
- `graph/outputs/.../metrics/model_results_initial.csv`
- `graph/outputs/.../metrics/model_results_graph_reweighted.csv`

默认一键脚本会同时跑初始版和图重加权版，并合并到 `metrics/model_results.csv`。如需关闭：

```bash
export DISABLE_GRAPH_REWEIGHTING=1
bash graph/run_all.sh
```

可调参数：

```bash
export GRAPH_REWEIGHT_ALPHA=0.7
export GRAPH_SUPPORT_TOP_K=20
export GRAPH_SUPPORT_NEIGHBOR_REVIEW_CAP=20
```

## No-LLM Fake Extractor Baseline

如果 LLM cache 还没生成完，可以先跑一个不依赖 LLM 的基础实验。这个模式不会读取或生成 LLM mask，而是把整条评论 token 作为候选异常区域，让旧的 RoBERTa/LogicTower/Cross-Attention 提取器自己学习 review-level 异常向量 `a_r`。

推荐命令：

```bash
export FAKE_EXTRACTOR_ONLY=1
export RUN_LLM_CACHE=0
bash graph/run_all.sh
```

这个模式默认输出到：

```text
graph/outputs/yelpzip_fake_extractor_only
```

它默认不启用 SKEP 双塔，也默认不跑 legacy baselines，目的是先快速得到“无 LLM mask，仅虚假提取器 + 行为图”的基础结果。若要强行打开 SKEP：

```bash
export SECONDARY_MODEL_NAME_OR_PATH="pretrain_model/skep-base"
```

也可以直接用底层参数：

```bash
python -m graph.run_final_experiment \
  --mask_source full_text \
  --output_dir graph/outputs/yelpzip_fake_extractor_only
```

`LogicAE-CB` 当前只对异常向量相似度 `S_logic` 设最低门槛，综合分 `S_LogicAE_CB` 只用于排序，不再额外设置 `tau_edge`。默认使用分位数阈值：

```text
tau_logic = max(0.30, quantile(S_logic_candidates, 0.60))
```

可调参数：

```bash
export LOGIC_THRESHOLD_MODE=quantile   # quantile / fixed / none
export LOGIC_THRESHOLD_QUANTILE=0.60
export LOGIC_THRESHOLD_VALUE=0.30
```

用户节点现在额外包含 5 个规范化行为异常指标，并取平均得到 `behavior_anomaly_score`：

```text
RD  = rating_deviation_avg / 4
EXR = extreme_rating_ratio
MRO = percentile(max_daily_reviews)
AD  = 1 - percentile(user_tenure_days)
ATR = 1 - percentile(avg_review_gap_days)
behavior_anomaly_score = mean(RD, EXR, MRO, AD, ATR)
```

## 1. 目标

这版实验把旧 `SyntaxAwareSubSentence` 主线从“最终 review 分类器”改成“单条评论异常模式编码器”，再把评论级异常向量聚合到用户级，并在 YelpZip 上构造用户行为图，验证：

- `Base = UPU + UTU + USU`
- `Base + TextSim`
- `Base + CB`
- `Base + LogicAE-CB`
- `Full = Base + TextSim + CB + LogicAE-CB`

实验口径不是复现师兄论文的完整多关系 GAT，而是在师兄“用户行为多关系图”这条可信主线上加料：

- 行为图底座仍然以 `UPU / UTU / USU` 为核心。
- 我们新增的贡献是 `LLM span -> token mask -> old logic encoder -> user abnormal vector -> LogicAE-CB edge`。
- 最终比较重点是 `Base + LogicAE-CB` 是否优于 `Base`、`Base + TextSim`、`Base + CB`，而不是单纯追求复刻师兄表格中的绝对数值。

同时保留旧项目 review-level 文字分类基线的自动化入口：

- `SyntaxAwareSubSentence`
- `roberta`

## 2. 新目录

- `C:\baidunetdiskdownload\副本\graph`
- 入口脚本：
  - `C:\baidunetdiskdownload\副本\graph\run_final_experiment.py`
  - `C:\baidunetdiskdownload\副本\graph\run_final_experiment.sh`
- Prompt 模板：
  - `C:\baidunetdiskdownload\副本\graph\prompts\llm_abnormal_pattern_extraction.txt`

## 3. 数据格式

### 3.1 数据源优先级

1. `--data_path` 显式指定
2. `graph data/YelpZip_reviews_correct.csv.gz.part.*` 自动合卷并解压
3. `graph data/dataset/yelpzip.csv`

### 3.2 统一后的 review schema

代码会标准化为：

```text
review_node_id
user_id
product_id
rating
review_label
review_date
review_text
split
```

当前 YelpZip 默认支持以下字段映射：

- `user_id`
- `prod_id -> product_id`
- `text -> review_text`
- `label=-1/1` 时默认解释为 `-1=fake, 1=real`
- `tag=fake/real` 会优先用于 label 语义校准

## 4. 用户切分

严格按 `user_id` 划分 `train / val / test`。

在切分前，当前默认预处理是：

1. 删除评论数少于 `3` 的用户
2. 删除评论数少于 `3` 的餐厅/商品
3. 删除标签矛盾用户：
   - 同一 `user_id` 下面同时出现 fake review 和 real review
4. 删除矛盾用户后，再重新做一轮活跃用户/活跃餐厅过滤，直到稳定

对应参数：

- `--min_user_reviews`
- `--min_product_reviews`

- `user_label = 1` 当且仅当该用户至少有一条 fake review
- `user_label = 0` 否则

切分产物：

- `graph/outputs/.../prepared_data/reviews_canonical.csv`
- `graph/outputs/.../prepared_data/users_canonical.csv`
- `graph/outputs/.../prepared_data/user_splits.csv`
- `graph/outputs/.../prepared_data/legacy_textcls_data/{train,dev,test}.tsv`

## 5. LLM JSON 格式

正式实验必须提供 JSONL，每行一条 review，最少要有：

```json
{
  "review_node_id": 123,
  "abnormal_patterns": [
    {
      "pattern_type": "generic_promotion",
      "description": "Broad praise without details.",
      "evidence_span": "best place ever",
      "confidence": 0.82
    }
  ],
  "sentiment": "positive",
  "specificity_score": 0.21,
  "template_score": 0.68,
  "exaggeration_score": 0.74,
  "experience_detail_score": 0.18,
  "claim_summary": "..."
}
```

默认建议路径：

`graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl`

如果没有 JSONL：

- 正式实验会直接报错
- 只有 `--debug_use_empty_mask` 才允许空 mask，用于 smoke test

## 6. abnormal mask 生成

实现位置：

- `graph/llm_utils.py`

对齐流程：

1. 用 tokenizer 的 `offset_mapping`
2. 先做不区分大小写的 exact match
3. 再做 whitespace-normalized exact match
4. 找不到则记录到 `unmatched_spans.csv`

输出：

- `graph/outputs/.../llm_mask/abnormal_token_masks.npy`
- `graph/outputs/.../llm_mask/llm_review_features.csv`
- `graph/outputs/.../llm_mask/mask_alignment_stats.csv`

## 7. old architecture 如何改成 abnormal encoder

实现位置：

- `graph/review_models.py`

新模块名：

- `LLMMaskedLogicEncoder`

结构是：

1. 主文本 backbone 编码 token hidden states
2. `H_abn = H * abnormal_mask`
3. `H_abn` 进入 BiLSTM logic tower
4. logic query 通过 cross-attention 回看原始 token states
5. 拼接 `[CLS, q_r, c_r, LLM numeric features, optional secondary prior]`
6. 过 bottleneck MLP 输出 `a_r`
7. `a_r` 经过辅助头输出 `p_fake_review`

这里保留了旧主线“logic tower + cross-attention”的核心思路，但不再依赖旧的 proposition pooling 当主逻辑输入，而是改成 LLM abnormal token mask。

## 8. 用户向量如何聚合

每条 review 会得到：

- `a_r`
- `p_fake_review`
- `specificity_score`
- `template_score`
- `exaggeration_score`
- `experience_detail_score`

默认 `top_m=3`，排序分数：

```text
evidence_score = 0.5 * p_fake_review
               + 0.2 * template_score
               + 0.2 * exaggeration_score
               + 0.1 * (1 - specificity_score)
```

然后：

```text
a_u = mean(top_m a_r)
```

输出：

- `graph/outputs/.../logic_vectors/review_abnormal_vectors.npy`
- `graph/outputs/.../logic_vectors/user_abnormal_vectors.npy`
- `graph/outputs/.../logic_vectors/review_abnormal_scores.csv`

## 9. LogicAE-CB 边如何构造

实现位置：

- `graph/graph_pipeline.py`

分数：

```text
S_LogicAE_CB = 0.4 * S_logic
             + 0.2 * S_time
             + 0.2 * S_rating
             + 0.2 * S_product
```

其中：

- `S_logic` 来自用户异常向量 cosine 相似度
- `S_time` 是时间桶重叠
- `S_rating` 是评分差异或评分倾向一致
- `S_product` 是共享商品

同时实现了：

- `UPU`
- `UTU`
- `USU`
- `TextSim`
- `CB`
- `LogicAE_CB`

输出：

- `graph/outputs/.../edges/*.csv`

## 10. 图模型

正式实验默认采用轻量 relation attention 用户分类器。它不是复现师兄完整 MRGAT，而是基于当前已构造的多类边，先对每种关系做邻居特征聚合，再学习“当前用户更应该相信哪类关系”。

实现位置：

- `graph/relation_model.py`

做法：

1. 先构造用户自特征：
   - `total_reviews`
   - `avg_rating`
   - `rating_std`
   - `rating_entropy`
   - `rating_deviation_avg`
   - `rating_deviation_std`
   - `positive_ratio`
   - `negative_ratio`
   - `extreme_rating_ratio`
   - `max_daily_reviews`
   - `burst_ratio`
   - `active_days`
   - `user_tenure_days`
   - `avg_review_gap_days`
   - `std_review_gap_days`
   - `avg_review_time_lag_days`
   - `std_review_time_lag_days`
   - `avg_review_length`
   - `user_abnormal_vector`
2. 对每种边关系做 weighted neighbor mean aggregation
3. 将 self block 与各 relation block 输入 `relation_attn`
4. `relation_attn` 学习关系级注意力权重并输出 `user_label`

仍保留 `LogisticRegression` 和 `MLPClassifier` 作为降级/对照入口：

```bash
--relation_model logreg
--relation_model mlp
```

正式推荐：

```bash
--relation_model relation_attn
```

默认会输出：

- `MLP_no_graph`
- `Base`
- `Base_TextSim`
- `Base_CB`
- `Base_LogicAE_CB`
- `Full`

## 11. 如何运行

### 11.0 一键完整运行

推荐正式服务器直接运行：

```bash
export OPENAI_API_KEY="你的 API key"
export LLM_MODEL="gpt-4o-mini"

bash graph/run_all.sh
```

`run_all.sh` 会自动完成：

1. 读取/预处理 YelpZip
2. 断点生成或补齐 `graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl`
3. 使用该 LLM cache 跑最终图实验
4. 输出 `model_results.csv`、`edge_stats.csv`、`run_summary.json`

如果使用本地 vLLM / OpenAI-compatible 接口：

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="你的模型名"

bash graph/run_all.sh
```

如果使用硅基流动 `Pro/deepseek-ai/DeepSeek-V3.2`：

```bash
export LLM_API_KEY="你的硅基流动 API key"
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_MODEL="Pro/deepseek-ai/DeepSeek-V3.2"
export LLM_ENABLE_THINKING="false"
export LLM_NO_RESPONSE_FORMAT=1
export LLM_WORKERS=2

bash graph/run_all.sh
```

如果预算有限，推荐“大模型少量种子标注 + 本地 T5 补齐”的省钱模式：

```bash
export LLM_API_KEY="你的硅基流动 API key"
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_MODEL="Pro/deepseek-ai/DeepSeek-V3.2"
export LLM_ENABLE_THINKING="false"
export LLM_WORKERS=2

export USE_LOCAL_ANNOTATOR=1
export LLM_SEED_SIZE=5000
export LOCAL_ANNOTATOR_BASE_MODEL="google/flan-t5-base"
export LOCAL_ANNOTATOR_BATCH_SIZE=4
export LOCAL_ANNOTATOR_EPOCHS=3
export PRIMARY_MODEL_NAME_OR_PATH="roberta-base"
export SECONDARY_MODEL_NAME_OR_PATH="pretrain_model/skep-base"

bash graph/run_all.sh
```

如果服务器不能直连 HuggingFace，先设置镜像站：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_CACHE="$PWD/.hf_cache/transformers"
```

也可以提前下载模型到服务器本地目录，然后完全使用本地路径：

```bash
export PRIMARY_MODEL_NAME_OR_PATH="/path/to/models/roberta-base"
export SECONDARY_MODEL_NAME_OR_PATH="/path/to/models/skep-base"
export LOCAL_ANNOTATOR_BASE_MODEL="/path/to/models/flan-t5-base"
```

下载示例：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download roberta-base --local-dir /path/to/models/roberta-base
huggingface-cli download google/flan-t5-base --local-dir /path/to/models/flan-t5-base
```

`SKEP` 如果无法从 HuggingFace 找到，请把你项目已有的 `pretrain_model/skep-base` 目录同步到服务器，并设置 `SECONDARY_MODEL_NAME_OR_PATH` 指向该目录。

该模式会先生成：

```text
graph/outputs/llm_cache/yelpzip_llm_seed.jsonl
```

然后本地微调 seq2seq 标注器，补齐：

```text
graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl
```

本地标注器会校验 `evidence_span` 是否能在原评论中找到；找不到的 span 会被丢弃，避免污染 token mask。若显存不足，可把 `LOCAL_ANNOTATOR_BASE_MODEL` 改成 `google/flan-t5-small`。

如果本地 T5 全量生成太慢，可以改用 7B API 全量短标注，这是当前更推荐的省时方案：

```bash
export LLM_API_KEY="你的硅基流动 API key"
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LLM_COMPACT_PROMPT=1
export LLM_MAX_TOKENS=160
export LLM_WORKERS=8
export USE_LOCAL_ANNOTATOR=0

bash graph/run_all.sh
```

该模式不训练本地 T5，而是让 7B API 直接补齐全量 JSONL。`compact prompt` 只输出最多 2 个异常 span 和四个数值分数，不输出长解释，速度和费用都更可控。`Qwen/Qwen2.5-7B-Instruct` 支持 JSON mode，通常不需要设置 `LLM_NO_RESPONSE_FORMAT=1`。

调试接口时可以只生成 20 条。设置 `LLM_LIMIT` 时，`run_all.sh` 会在生成 cache 后自动停止，不会进入正式训练，避免误用不完整 cache：

```bash
LLM_LIMIT=20 LLM_WORKERS=1 bash graph/run_all.sh
```

如果 LLM cache 已经完整存在，只想直接跑最终实验：

```bash
RUN_LLM_CACHE=0 bash graph/run_all.sh
```

正式实验会检查 LLM cache 是否覆盖当前预处理后的全部 `review_node_id`。如果缺失，会写出 `missing_llm_cache_ids.txt` 并中止。

### 11.1 分步：先生成 LLM cache

正式实验前必须先生成：

```text
graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl
```

服务器上使用 OpenAI-compatible 接口时：

```bash
export OPENAI_API_KEY="你的 API key"
export LLM_MODEL="gpt-4o-mini"

bash graph/generate_llm_cache.sh
```

如果你使用本地 vLLM / 兼容接口：

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="你的模型名"

bash graph/generate_llm_cache.sh
```

这个脚本会断点续跑：如果 JSONL 里已经有某个 `review_node_id`，再次运行会自动跳过。正式生成不要加 `--limit`；调试接口时可以先跑：

```bash
bash graph/generate_llm_cache.sh --limit 20 --workers 1
```

### 11.2 分步：正式实验

Linux 服务器推荐：

```bash
bash graph/run_final_experiment.sh
```

或者显式写完整命令：

```bash
python -m graph.run_final_experiment \
  --graph_data_dir "C:/baidunetdiskdownload/副本/graph data" \
  --output_dir "C:/baidunetdiskdownload/副本/graph/outputs/yelpzip_final" \
  --llm_jsonl_path "C:/baidunetdiskdownload/副本/graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl" \
  --primary_model_name_or_path roberta-base \
  --secondary_model_name_or_path pretrain_model/skep-base \
  --time_bucket week \
  --relation_model relation_attn \
  --run_legacy_baselines
```

如果你的服务器项目路径不同，只要改路径参数即可。

### 11.3 smoke test

如果服务器上有 `torch/transformers`，可以直接跑完整 smoke：

```bash
python -m graph.run_final_experiment --smoke_test
```

这会自动：

- 切到 `mock` review encoder
- 启用 `--debug_use_empty_mask`
- 关闭 legacy baselines
- 只抽一小批用户做贯通测试

注意：

- smoke test 不是正式实验
- 不能拿它的结果写进论文或主表

如果只是本地做“无训练依赖”的数据/构图贯通检查，可以跑：

```bash
python -m graph.smoke_test_data_only
```

这条命令不依赖 `torch/transformers`，只验证：

- 数据合卷与统一 schema
- 用户切分
- empty-mask 路线
- 边构造
- relation aggregation baseline 输出

## 12. 输出文件

核心结果：

- `graph/outputs/.../metrics/model_results.csv`
- `graph/outputs/.../metrics/edge_stats.csv`
- `graph/outputs/.../run_summary.json`

如果启用旧基线：

- `graph/outputs/.../legacy_baselines/legacy_baseline_results.csv`

## 13. 是否继续做这版的判断标准

这版值不值得继续，建议主要看：

1. `Base_LogicAE_CB` 是否稳定优于 `Base_CB` 和 `Base_TextSim`
2. `LogicAE_CB` 的 `fake_fake_ratio` 是否明显高于 `TextSim / CB`
3. `Full` 是否比 `Base` 至少有清晰增益，而不是随机小波动
4. AUC 和 AP 是否同步改善，而不是只偶然抬一个阈值指标

## 14. 当前实现的约束

1. 本地工作区没有 `pretrain_model`，所以正式训练要在服务器上提供可用的模型目录或缓存。
2. `roberta` 旧基线仍依赖项目旧目录结构，最好在服务器上沿用原来的 tokenizer/model 放置方式。
3. `LLMMaskedLogicEncoder` 第一版优先追求稳定可跑，没有额外加 attention-mask 对齐损失。
4. 图模型目前是 relation aggregation baseline，还没有接更重的 GNN。
