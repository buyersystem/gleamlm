# GleamLM 训练优化参考 —— 从 40M 到 80M/126M

> 基于 V4 40M 实测（Epoch 1 PPL=14.95）和 SmolLM 论文的优化方案，逐级复用。

---

## 1. WSD 三段式学习率调度

**来源**：SmolLM2（Allal et al., 2025）| **适用**：40M / 80M / 126M

### 对比

| | Cosine（当前） | WSD（建议） |
|------|:---:|:---:|
| 阶段 | Warmup → Decay | Warmup → **Stable** → Decay |
| Stable 占比 | 0 | 80% 总步数 |
| Decay 占比 | 90% | 15% |
| 效果 | LR 全程下降 | 长时间高位 LR，充分学习 |

### 理由

40M 参数少、容量有限，Cosine 过早下降 LR 导致后期学习率过低、token 利用率不足。WSD 让模型 80% 时间在峰值 LR 学习，最后 15% 快速退火到最低。

### 实现（gleamlm_train.py）

```python
def get_lr_wsd(step, total_steps, warmup_ratio=0.02, stable_ratio=0.80, min_lr_ratio=0.05):
    """WSD: Warmup → Stable → Decay"""
    warmup_steps = int(total_steps * warmup_ratio)
    stable_steps = int(total_steps * stable_ratio)
    decay_steps = total_steps - warmup_steps - stable_steps

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < warmup_steps + stable_steps:
        return 1.0
    else:
        progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
```

### 预期

40M 同等 1.2B tokens 训练，PPL 预计额外降 **1.0 ~ 1.5**。

> 备选方案：STAG-LR（单参数高斯调度器，比 WSD 更简洁），见 `docs/STAG-LR-单参数学习率调度器设计.md`。

---

## 2. 预训练关闭 Dropout

**来源**：LLaMA 3 / Qwen3 / SmolLM | **适用**：40M / 80M / 126M

### 规则

| 阶段 | Dropout | 理由 |
|------|:---:|------|
| 预训练 | **0.0** | 全量参数学习、容量不浪费 |
| SFT | 0.1 | 防止小数据过拟合 |
| DPO | 0.0 | 参考模型冻结、无需正则 |

### 实现

```python
# gleamlm_train.py 创建模型时
dropout=0.0  # 预训练不丢弃

# gleamlm_sft.py（已默认 0.1）
parser.add_argument("--dropout", type=float, default=0.1)
```

### 预期

40M 关闭 dropout 等效恢复 ~10% 有效容量，PPL 降 **0.3 ~ 0.5**。

---

## 3. 学习率峰值

### 建议值

| 参数量 | LR 峰值 | 依据 |
|------|:---:|------|
| 40M | **4e-4** | 当前 3e-4 偏保守，小模型梯度方差大但不至发散 |
| 80M | **5e-4** | 接近 SmolLM-135M 的 5e-4 |
| 126M | **5e-4** | 直接对标 SmolLM-135M |

> 40M 不建议 5e-4：参数少、每步梯度方向噪音大，过高 LR 可能引发 loss 尖峰。先 4e-4 跑一轮，无尖峰再提。

### 预期

3e-4 → 4e-4，PPL 降 **0.2 ~ 0.3**。

---

## 4. 有效 Batch 放大

### 建议值

| 参数量 | batch_size | accumulate_grad | 有效 batch | 显存需求 |
|------|:---:|:---:|:---:|------|
| 40M | 8 | **16** | 128 | ~5 GB |
| 80M | 4 | **16** | 64 | 需测试 |
| 126M | 2 | **16** | 32 | 需测试 |

### 理由

40M 当前有效 batch=64，显存只用了 5GB/12GB（42%）。翻倍到 128 后显存约 9GB，仍在安全区。更大 batch → 更平滑梯度 → loss 毛刺减少。

### 预期

不直接降 PPL，但收敛更稳、loss 曲线更干净。

---

## 5. 词表规模

### 建议

| 参数量 | 词表大小 | Embed 参数 | 理由 |
|------|:---:|:---:|------|
| 40M | **12K** | 6.1M (15%) | 当前最优，省参数给 Transformer |
| 80M | **16K ~ 24K** | 8~12M | 适度扩容，中文 token 效率提升 |
| 126M | **24K ~ 32K** | 12~16M | Embed 占比可控，匹配 SmolLM |

### 不要做的事

40M 直接上 32K：Embed 从 6.1M 膨胀到 16.4M，增加 10M 参数全部堆在查表而非推理深度。这些参数放到 Transformer 层数上收益大得多。

---

## 6. 数据配比原则

基于 V4 实测总结：

| 原则 | 说明 |
|------|------|
| Wiki + 百科 ≤ 45% | 知识源高度重叠，过多导致 token 冗余 |
| News ≥ 30% | 语言多样性最高，改善生成流畅度 |
| QA ≤ 20% | 质量方差大，不可验证内容比例不宜过高 |
| 字符加权换算 | 行均字符差 6x（新闻 vs 维基），必须按字符量配比 |

### 40M 验证配比（30/12/43/15）

```
源      行均字符  目标字符% →  行数配比
wiki      123    30.0%       52.8%
baike     145    12.0%       17.9%
news      752    43.0%       12.4%
qa        192    15.0%       16.9%
```

### 80M/126M 建议

各源数据量充足时，Wiki+百科可适当提升到 35+15=50%（知识深度），News 不低于 30%，QA 保持 15-20%。具体根据实际可用数据量调整。

---

## 7. 清洗规则清单

### 通用规则（所有源）

| 规则 | 参数 | 说明 |
|------|------|------|
| 长度过滤 | min=10, max=2000 | 过滤过短/过长条目 |
| 中文占比 | ≥ 0.15 | 滤除纯英文/数字行 |
| 简繁转换 | `--convert_zh` | zhconv 自动转简体 |
| HTML/URL | 正则 | 清空标签和链接 |

### 源特定规则

| 源 | 额外规则 | 说明 |
|------|------|------|
| Wiki | `--filter_wiki_junk` | 美国人口普查模板、坐标条目、非建制地区 |
| News | `--filter_ads` | QQ/微信/电话/扫码/促销/加盟等 12 条正则 |
| QA | `--min_answer_len=20` | 去短答；去 URL；Q-level MD5 去重 |

### 去重策略

| 工具 | 模式 | 适用源 |
|------|------|------|
| `dedup_text.py` | `exact`（MD5 全文） | Wiki、百科 |
| `dedup_text.py` | `prefix`（前 100 字符） | News（标题去重） |
| `filter_qa.py` | Q-hash | QA（同一问题去重） |

### QA 多格式解析

`filter_qa.py` 支持 5 种格式回退：

```
问题：{q} 回答：{a}       # 原始格式
Q: {q} A: {a}              # Q/A 格式
问：{q} 答：{a}            # 问答格式
{"question": "...", ...}   # JSON 片段
{q}\t{a}                   # Tab 分隔
```

---

## 8. Flash Attention 内置加速

**来源**：PyTorch 2.0+ | **适用**：40M / 80M / 126M

### 对比

| | 当前手写 GQA | Flash Attention |
|------|:---:|:---:|
| batch=16, seq=1024 | OOM（注意力矩阵 2.1GB） | 正常运行 |
| 代码改动 | — | 1 行替换 7 行 |
| Windows 兼容 | ✅ | ✅（PyTorch 2.0+） |
| 速度 | 基准 | +10-15% |

### 实现（models/gleamlm_model.py）

```python
# 当前（手写 matmul + softmax，7 行）
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
if mask is not None:
    scores = scores + mask
attn_weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
attn_weights = self.dropout(attn_weights)
output = torch.matmul(attn_weights, V)

# 替换为（1 行，PyTorch 自动选最优后端）
output = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=mask,
    dropout_p=self.dropout if self.training else 0.0
)
```

PyTorch 2.0+ 自动选择 Flash Attention / Memory Efficient / Math fallback，无需显存中存储完整注意力矩阵。dropout 仅训练时生效。

---

## 9. SFT 数据量规划

**来源**：MiniMind 对比分析 + V3 SFT 实测 | **适用**：40M / 80M / 126M

### 数据量与效果

| 数据量 | 效果 | 说明 |
|:---:|------|------|
| 995 条（V3 SFT） | 管线验证 ✅ | 模型从续写变问答，但输出碎片化 |
| 1-5 万条（建议） | 可用对话质量 | 覆盖通用问答、百科、创作场景 |
| 90 万条（MiniMind） | 多轮对话、工具调用、安全对齐 | 工业级对话能力 |

### 40M 建议

当前 V4 Epoch 1 PPL=14.95，远超 V3 的 34.93。SFT 起点显著提升，应配合数据量升级到 **1 万条** 以上。使用 `scripts/generate_sft_data.py` + DeepSeek API 蒸馏批量生成。

### 学习率配套

```python
# 当前 SFT（995 条，31 步）：lr=5e-7 太保守
# 建议（1 万条，~300 步）：lr=1e-5 ~ 5e-5
```

步数越少，学习率需越高才能产生有效参数更新。

---

## 10. DPO rejected 生成策略

**来源**：V3 DPO 实测 | **适用**：40M / 80M / 126M

### 问题

当前 DPO 数据：chosen = DeepSeek 高质量回答，rejected = **预训练模型**（PPL=35）的噪声输出：

```
chosen:   "人工智能是研究如何让计算机模拟人类智能行为的学科..."
rejected: "有没有必要在一个人面前说过我很不开心的时候我发现自己..."
          ↑ 完全无关的噪声，不需要偏好学习就能区分
```

DPO 学到的只是"不要输出纯噪声"，而 SFT 已教会这个。chosen vs rejected 差距极端，偏好信号无效。

### 改进

**用 SFT 模型生成 rejected**，而非预训练模型：

| rejected 来源 | 质量 | 与 chosen 差距 | DPO 效果 |
|------|:---:|:---:|:---:|
| 预训练模型（当前） | 噪声 | 极端 | 学习"别输出垃圾"（SFT 已覆盖） |
| SFT 模型（建议） | 合理但较差 | 微妙 | 学习"哪个回答更好"（真正偏好） |

### 实现

```bash
# 用 SFT 模型对同一组 instruction 生成 rejected
python scripts/generate_rejected.py \
    --model checkpoints/sft/sft_best.pt \
    --data ./data/sft_data_clean.jsonl \
    --output ./data/dpo_rejected.jsonl

# 构造 DPO 对：chosen=蒸馏高质量 / rejected=SFT 模型生成
```

---

## 附录 A：V3 vs V4 实测对比

| | V3 | V4 |
|------|:---:|:---:|
| 架构 | 8层 × 512dim × MHA | **12层 × 512dim × GQA + QK-Norm** |
| 词表 | SentencePiece 32K | **BBPE 12K** |
| 参数 | ~39M | ~40.8M |
| Transformer | ~22M | **~34M** (+55%) |
| 数据 | 4源 40/23/22/15 | 4源 30/12/43/15 |
| Epoch 7 PPL | 34.93 | — |
| Epoch 0 PPL | — | **16.55** |
| Epoch 1 PPL | — | **14.95** |

> V4 第 1 个 epoch 已超越 V3 第 7 个 epoch 达 20 个 PPL 点。

## 附录 B：预期改进叠加

以 V4 40M Epoch 5 预估 PPL ~12.5 为基线：

| 改进 | 预期间 PPL 降幅 |
|------|:---:|
| WSD 三段式调度 | -1.0 ~ -1.5 |
| 预训练 dropout=0 | -0.3 ~ -0.5 |
| LR 3e-4 → 4e-4 | -0.2 ~ -0.3 |
| Batch 64 → 128 | 平滑但不降 PPL |
| Flash Attention | +10-15% 速度，省显存（不直接降 PPL） |
| SFT 数据 995→1万条 | 对话质量质变，不反映在预训练 PPL |
| **叠加后预期间 PPL** | **~10 ~ 11** |
