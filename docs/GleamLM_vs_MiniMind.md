# GleamLM-Nano V4 vs MiniMind 对比分析

> 对比版本：GleamLM-Nano V4（~40.7M）vs MiniMind-3（64M）/ MiniMind2-Small（26M）
> 项目来源：[github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

## 1. 项目定位

| 维度 | GleamLM-Nano V4 | MiniMind |
|---|---|---|
| 定位 | **教学项目**，聚焦核心架构原理 | **全链路复现工程**，覆盖完整流程 |
| 范围 | Pretrain + SFT + DPO + FP16量化 | Pretrain + SFT + LoRA + DPO + PPO/GRPO/CISPO + Agentic RL + Tool Use + 蒸馏 + 视觉/语音多模态 |
| 代码量 | ~2500 行 | 整个仓库数万行 |
| 外部依赖 | 纯 PyTorch，零 HuggingFace，自研 BBPE 分词器 | 继承 HuggingFace（PreTrainedModel、GenerationMixin） |
| 目标用户 | 理解模型架构底层原理 | 快速搭建可用对话模型，兼顾学习和实用 |

---

## 2. 模型配置对比

| 参数 | GleamLM-Nano V4 | MiniMind-3 | MiniMind2-Small |
|---|---|---|---|
| 参数量 | **~40.7M** | **64M** | **26M** |
| d_model | **512** | **768** | **512** |
| num_layers | **12** | 8 | 8 |
| Q heads / KV heads | **8 / 4（GQA）** | **8 / 4（GQA）** | **8 / 2（GQA）** |
| vocab_size | **12,001**（自研 BBPE） | **6400** | **6400** |
| max_seq_len | **1024** | **32768** | **32768** |
| rope_theta | 10000 | **1e6** | **1e6** |
| intermediate_size | **1365**（int(8/3×512)） | **~2412**（ceil(768×π/64)×64） | **~1600**（ceil(512×π/64)×64） |
| FFN 激活函数 | SwiGLU | SwiGLU | SwiGLU |
| 归一化 | RMSNorm | RMSNorm | RMSNorm |
| 位置编码 | RoPE | RoPE + **YaRN 外推** | RoPE + YaRN 外推 |
| QK-Norm | ✅ | ✅ | ✅ |
| Flash Attention | ✅ `F.scaled_dot_product_attention` | ✅ | ✅ |
| MoE | ❌ | ✅（4 experts / top-1） | ❌ |
| 权重绑定 | ✅ | ✅ | ✅ |

---

## 3. 架构细节差异

### 3.1 QK-Norm（已追平 ✅）

GleamLM V4 已在注意力计算前对 Q 和 K 额外做 RMSNorm，与 MiniMind / LLaMA 3 一致：

```python
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

xq, xk = self.q_norm(xq), self.k_norm(xk)
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

**效果**：Q 和 K 的数值范围被归一化到稳定区间后，注意力分数 `Q@K^T` 的方差更可控，训练更稳定，尤其对长序列友好。

### 3.2 层数策略差异：Deep-Narrow vs 标准

这是当前最大架构差异：

| | GleamLM-Nano V4 | MiniMind-3 | MiniMind2-Small |
|---|---|---|---|
| num_layers | **12**（Deep-Narrow） | 8 | 8 |
| d_model | 512 | 768 | 512 |
| Embed 参数 | 6.1M（15%） | 4.9M（8%） | 3.3M（13%） |
| Transformer 参数 | **34.6M（85%）** | ~59M（92%） | ~22.7M（87%） |

GleamLM V4 验证了 **Deep-Narrow（12×512）** 架构在小模型上的有效性：同参数量下，12 层窄网络比 8 层宽网络能更高效利用参数（V3 8×512 PPL=34.93 → V4 12×512 PPL=13.65）。MiniMind-3 走的是 8×768 加宽路线。

### 3.3 FFN 中间维度公式差异

- **GleamLM**：`d_ff = int(8/3 × d_model)` = 1365（标准 SwiGLU 公式，保证参数量与 4× ReLU FFN 一致）
- **MiniMind**：`intermediate_size = ceil(hidden_size × π / 64) × 64` ≈ 2412（d_model=768 时）

MiniMind 的公式本质上在 **d_ff / d_model ≈ π** 附近取值，同时对齐 64 的倍数以利用 Tensor Cores。对于 d_model=512，这个公式给出 ~1600，比标准公式的 1365 大，参数量更多，表达能力更强但也更贵。

### 3.4 RoPE 外推能力

- **GleamLM**：max_seq_len=1024，RoPE 支持天然外推到约 2048，80M 版本将升级到 2048 + RoPE θ=500K
- **MiniMind**：max_position_embeddings=32768，支持 **YaRN** 外推

### 3.5 词表大小差异（V4 已大幅改善）

| | GleamLM-Nano V3（旧） | GleamLM-Nano V4 | MiniMind-3 | MiniMind2-Small |
|---|---|---|---|---|
| vocab_size | 32000 | **12,001** | 6400 | 6400 |
| embedding 参数量 | 16.4M（42%） | **6.1M（15%）** | 4.9M（8%） | 3.3M（13%） |
| 分词器 | SentencePiece BPE | **自研 BBPE**（纯 Python） | 自定义 BPE | 自定义 BPE |

V4 自研 BBPE 12K 词表的核心改进：embedding 占比从 42% 降至 15%，释放 10.3M 参数给 Transformer（22.6M→34.6M）。BBPE 以字节为基元，起步仅 256 个符号，12K 词表中可用合并 slot 比 16K BPE 还多。同时原生支持 `<|im_start|>`、`<|im_end|>` ChatML 特殊 token。

---

## 4. 训练流程对比

| 阶段 | GleamLM-Nano V4 | MiniMind |
|---|---|---|
| 预训练 | ✅ 从零实现（AMP + DDP + Cosine + 断点续训） | ✅ 从零实现 |
| SFT 指令微调 | ✅ ChatML + loss mask，10000 条 | ✅ 覆盖多轮对话 |
| DPO 偏好对齐 | ✅ 150 对，policy + frozen reference | ✅ 从零实现 |
| FP16 量化 | ✅ 体积减半（178.9MB→89.5MB） | ✅ |
| LoRA 微调 | ❌ | ✅ 从零实现，无 peft 依赖 |
| PPO / GRPO / CISPO | ❌ | ✅ 从零实现 |
| Agentic RL | ❌ | ✅ 支持工具调用的多轮强化学习 |
| 模型蒸馏 | ❌ | ✅ 白盒蒸馏（R1 推理蒸馏） |
| 多模态 | ❌ | ✅ MiniMind-V（视觉）、MiniMind-O（全模态） |
| 分布式训练 | ✅ DDP | ✅ DDP + DeepSpeed |
| 训练可视化 | ✅ TensorBoard | ✅ wandb / swanlab |
| 断点续训 | ✅ 保存 optimizer/scheduler/scaler 全量状态 | ✅ |

---

## 5. 数据处理对比

| 方面 | GleamLM-Nano V4 | MiniMind |
|---|---|---|
| 数据格式 | `.txt` 纯文本 | `.jsonl`（JSON Lines） |
| 分词 | **自研 BBPE 12K**（纯 Python，零依赖），预分词缓存 | 自定义 BPE + ByteLevel，预分词 |
| 词表 | 12,001（Embed 占比 15%） | 6400（极简，embedding 层小） |
| 缓存 | `.npy` 二进制缓存（mmap 磁盘映射，~1MB 内存） | 无独立缓存，直读 JSONL |
| 窗口切分 | 滑动窗口（stride=75%） | 按 max_length 截断/拼接 |
| 数据配比 | 四源字符加权（wiki 30%/baike 12%/news 43%/qa 15%） | 全阶段开源 |
| ChatML token | ✅ `<\|im_start\|>` / `<\|im_end\|>` 原生单 token | ✅ 内置 |
| 开源数据 | ❌ 需自行准备 | ✅ 全阶段开源（预训练、SFT、DPO、RLHF） |

---

## 6. 推理对比

| 方面 | GleamLM-Nano V4 | MiniMind |
|---|---|---|
| 采样参数 | temperature + top-k + top-p + repetition penalty | 相同 + do_sample 控制 |
| KV Cache | ✅ 手动实现 | ✅ 内置 |
| 流式输出 | ✅ 每 4 token yield | ✅ token 级 streamer |
| 交互模式 | ✅ 命令行 | ✅ Web demo + OpenAI API 兼容 |
| Flash Attention | ✅ `F.scaled_dot_product_attention` | ✅ |
| HuggingFace 兼容 | ❌ | ✅（可用 vLLM、ollama、llama.cpp 推理） |
| 外推能力 | 有限（~2×） | ✅ YaRN（32K） |
| SFT 推理 | ✅ ChatML 格式 + `<\|im_end\|>` 自动截断 | ✅ |
| DPO 推理 | ✅ | ✅ |

---

## 7. 各自优势总结

### GleamLM-Nano V4 的优势

| 优势 | 原因 |
|---|---|
| **代码量极小** | ~2500 行，模型+训练+推理全在一个仓库 |
| **零外部抽象** | 不继承 PreTrainedModel，所有逻辑透明 |
| **自研 BBPE 分词器** | 607 行纯 Python，零依赖，原生 ChatML |
| **Deep-Narrow 验证** | 12×512 比 8×512 同参下 PPL 暴降 21 点（34.93→13.65） |
| **全链路打通** | 预训练→SFT→DPO→量化→推理完整闭环 |
| **训练循环完全手写** | DDP、AMP、Cosine LR、Checkpoint 全部自己写 |
| **中文文档完善** | 14 章课程文档 + 52 条排坑记录 |
| **硬件友好** | 单卡 12GB 可完整训练，Windows/Linux 双平台 |
| **V4 预训练质量** | PPL 13.65，40M 级别 SOTA |

### MiniMind 的优势

| 优势 | 原因 |
|---|---|
| **全链路覆盖最广** | 从 Pretrain 到 SFT 到 RLHF 到 Agentic RL，一站式 |
| **YaRN 外推** | 4096 → 32768 推理扩展 |
| **MoE 支持** | Dense + MoE 双版本 |
| **生态系统兼容** | 可转 Transformers / vLLM / ollama / llama.cpp |
| **完整开源数据** | 全阶段数据预训练到 RLHF |
| **多模态** | 支持视觉（V）和全模态（O） |
| **SFT 数据量** | 90 万+条，对话质量显著更优 |

---

## 8. 各维度评级

| 维度 | GleamLM-Nano V4 | MiniMind | 推荐场景 |
|---|---|---|---|
| 教学纯净度 | ★★★★★ | ★★★ | 想逐行理解核心原理 |
| 功能完整度 | ★★★ | ★★★★★ | 想做真正可用的对话模型 |
| 代码可读性 | ★★★★★ | ★★★★ | 想一口气读完、理解透彻 |
| 预训练质量（同参数量级） | ★★★★★ | ★★★★ | 追求小模型 PPL 极致 |
| 部署便利性 | ★★★ | ★★★★★ | 想对接 vLLM / ollama 等工具 |
| 长文本能力 | ★★（1K） | ★★★★★（32K+YaRN） | 需要处理长文档 |
| 对话质量 | ★★★（10K SFT） | ★★★★★（90万+ SFT） | 需要多轮对话能力 |
| 训练成本 | 低（40M，单卡 12GB） | 低（64M，约 3 元/2 小时） | 快速验证想法 |
| 扩展潜力 | ★★★★（已有 80M/126M/0.6B 路线） | ★★★★★（已有全链路） | 想深入研究或做毕业设计 |

**核心差异**：GleamLM V4 的核心亮点是 **Deep-Narrow 架构验证**（12×512 比 8×512 PPL 暴降 21 点）和 **自研 BBPE 分词器**（Embed 占比 15%）。MiniMind 的核心优势是全链路覆盖（含 RLHF、Agentic RL、多模态）和海量 SFT 数据带来的对话能力。两者架构已追平（QK-Norm、Flash Attention），差异主要在生态和规模。

---

## 9. 训练效果对比

### 9.1 预训练 PPL

| 模型 | 参数量 | 词表 | 预训练数据 | Epoch | Val PPL |
|------|:---:|:---:|------|:---:|:---:|
| **GleamLM-Nano V4** | 40.7M | BBPE 12K | 四源混合 1.2B tokens | 5 | **13.65** |
| GleamLM-Nano V3（旧） | 39M | BPE 32K | 四源混合 1.2B tokens | 8 | 34.93 |
| MiniMind-3 | 64M | BPE 6.4K | 多源 | — | 未公开 |
| MiniMind2-Small | 26M | BPE 6.4K | 多源 | — | 未公开 |

> V4 第 1 个 epoch（PPL=14.95）已超越 V3 第 7 个 epoch（PPL=34.93）。Deep-Narrow + BBPE + QK-Norm 三项叠加带来质的飞跃。

### 9.2 SFT 效果对比

| 维度 | GleamLM-Nano V4 | MiniMind |
|------|-----------|----------|
| SFT 数据量 | 10,000 条 | 90 万+条 |
| 优化步数 | ~300 步 | ~数千步 |
| 覆盖场景 | 通用问答+百科+创作 | 多轮对话+工具调用+代码+数学+安全 |
| 数据来源 | DeepSeek API 蒸馏（单 turn） | 全阶段开源数据集（多 turn） |
| ChatML 格式 | ✅ 原生单 token | ✅ 内置 |
| SFT 学习率 | 5e-6（3 epoch） | 5e-7（多 epoch） |
| SFT 效果 | 成功从续写→对话，format 达标 | 多轮对话连贯，功能完整 |

### 9.3 V3→V4 参数效率的革命性提升

V3 时代的核心问题——大词表吞噬 Transformer 参数——已在 V4 彻底解决：

| 指标 | GleamLM V3（旧） | GleamLM V4 | MiniMind-2-Small |
|------|:---:|:---:|:---:|
| 总参数量 | 39M | 40.7M | 26M |
| vocab_size | 32,000 | 12,001 | 6,400 |
| Embedding 参数量 | 16.4M（42%） | **6.1M（15%）** | 3.3M（13%） |
| Transformer 参数 | 22.6M（58%） | **34.6M（85%）** | 22.7M（87%） |
| num_layers | 8 | **12** | 8 |
| QK-Norm | ❌ | ✅ | ✅ |
| Flash Attention | ❌ | ✅ | ✅ |
| Val PPL | 34.93 | **13.65** | 未公开 |

> **V4 比 V3 的 Transformer 参数多 53%，PPL 低 21 点。** V4 的 40.7M 在 Transformer 参数（34.6M）上远超 MiniMind-2-Small 的 22.7M，预训练质量已不可同日而语。

---

## 10. V4 已追平/超越的方面

| 方面 | V3 状态（旧） | V4 状态 | MiniMind |
|------|:---:|:---:|:---:|
| QK-Norm | ❌ | ✅ 追平 | ✅ |
| Flash Attention | ❌ 手写 | ✅ 追平 | ✅ |
| SFT 全链路 | ❌ | ✅ 追平 | ✅ |
| DPO 偏好对齐 | ❌ | ✅ 追平 | ✅ |
| ChatML token 原生支持 | ❌ 被拆成 3 子词 | ✅ 追平 | ✅ |
| FP16 量化 | ❌ | ✅ 追平 | ✅ |
| 断点续训（全量状态） | ❌ | ✅ 追平 | ✅ |
| 词表效率（Embed 占比） | 42%（差） | **15%（优）** | 8-13% |
| 预训练 PPL（同参数级） | 34.93（差） | **13.65（优）** | 未公开 |
| Transformer 层数 | 8（持平） | **12（超越）** | 8 |
| 自研分词器 | SentencePiece 依赖 | **BBPE 纯 Python（超越）** | 自定义 BPE |

---

## 11. 仍然存在的差距

| 方面 | GleamLM V4 | MiniMind |
|------|-----------|----------|
| YaRN 长文本外推 | ❌（80M 规划中） | ✅ |
| RLHF（PPO/GRPO/CISPO） | ❌（远期规划） | ✅ |
| SFT 数据量 | 10,000 条 | 90 万+条 |
| HuggingFace 生态兼容 | ❌ | ✅ |
| 多模态 | ❌ | ✅ |
| Web Demo | ❌ | ✅ |

---

*最后更新：2026-06-28（V4 全链路完成更新）*
