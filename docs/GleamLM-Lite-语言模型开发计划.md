# GleamLM-Lite-Pro 语言模型开发计划

## 一、项目定位

**烁珑 GleamLM** 是一套**面向端侧设备与教学科研的开源语言模型平台**，采用当前主流 LLM 技术栈（SwiGLU / GQA / RoPE / RMSNorm）。

**双轨定位**：

| 轨道 | 方向 | 核心优势 |
|------|------|---------|
| **端侧部署** | 游戏 NPC 引擎、嵌入式语音助手、本地知识库 | 40M 极致轻量 + INT4 量化仅 ~20MB |
| **教学科研** | 从零手写、每行可讲、消融实验平台 | 全链路透明 + 排坑记录 + 单卡 12GB 可复现 |

帮助学习者从经典 Transformer 平稳过渡到现代大模型架构，同时为端侧 AI 部署提供可商用的小模型基线。

**平台策略**：

| 阶段 | 操作系统 | GPU | 定位 | 面向人群 |
|------|----------|-----|------|---------|
| GleamLM-Nano（40M V4） | **Windows + Linux 双平台** | 单卡 12GB | **教学入门** — 从零手写、每行可讲 | 学生 / 自学者 |
| GleamLM-Lite（87M） | **Windows + Linux 双平台** | 单卡 12GB 可训 / 4×48GB 加速 | **快速消融平台** — 单卡日级迭代 | 研究生 / 研究者 |
| GleamLM-Pro（126M） | **Linux only** | 单卡 12GB 可训 / 4×48GB 加速 | **Lite 旗舰** — 对标 SmolLM2-135M | 研究者 / 技术报告 |
| GleamLM-0.6B | **Linux only** | 多卡集群 | **工业级验证** — 架构创新验证平台 | 研究团队 |

**当前版本：GleamLM-Nano V4（40M 参数，12 层 + QK-Norm + BBPE 12K）**

---

## 二、核心技术栈（完全现代 SOTA）

| 组件 | 实现 |
|------|------|
| 主干架构 | Pre-Norm Decoder-only |
| 位置编码 | RoPE 旋转位置编码（实数运算版，快 2-3 倍） |
| 归一化 | RMSNorm（替代 LayerNorm） |
| 注意力 | GQA 分组查询注意力（8Q / 4KV 头） |
| 激活函数 | SwiGLU 门控前馈网络 |
| 训练 | BF16 AMP + DDP + Warmup + Cosine |
| 推理 | KV Cache + 流式生成 + Temperature/TopK/TopP/RepetitionPenalty |

---

## 三、GleamLM-Nano V4 参数规格（40M，Deep-Narrow）

| 参数 | 值 | 说明 |
|------|-----|------|
| 上下文窗口 | 1024 | RoPE 支持外推 |
| 词表大小 | 12,002 | 自研 BBPE（纯 Python，零依赖） |
| 网络层数 | 12 | Deep-Narrow 架构 |
| 模型维度 | 512 | 消费级显卡友好 |
| QK-Norm | ✅ | LLaMA3 标准 |
| 总注意力头 | 8 | 标准配置 |
| KV 注意力头 | 4 | GQA 轻量化 |
| SwiGLU 中间维度 | 1365 | 约 2/3 × 4 × d_model |
| Dropout | 0.1 | 小模型正则化 |
| Embed 参数 | 6.1M | vocab=12003 × 512 |
| Transformer 参数 | 34.6M | 12 层 |
| 参数量 | **约 40.7M**（weight tying） | 不绑定时约 47M |

---

## 三-B、GleamLM-Pro 参数规格（设计草案）

> **注意**：`configs/pro.yaml` 为占位配置，参数与本草案不一致（pro.yaml: 16L×1024d, vocab=24K, d_ff=4096, max_seq=4096）。以下为设计目标的 126M 规格，最终参数待 Lite 训练完成后统一确定。

> 18 层 × 768 维，对标 SmolLM2-135M / GPT-2 Small(124M)，
> 架构更现代（QK-Norm + SwiGLU + GQA + BBPE）。

| 参数 | 值 | 说明 |
|------|-----|------|
| 上下文窗口 | 2048 | RoPE 支持外推至 4096 |
| 词表大小 | 12,002 | 复用 Nano 的 BBPE 词表 |
| 网络层数 | 18 | 深度优先 |
| 模型维度 | 768 | GPT-2 Small 同维度 |
| QK-Norm | ✅ | LLaMA3 标准 |
| 总注意力头 | 12 | head_dim=64 |
| KV 注意力头 | 6 | GQA 2:1 分组 |
| SwiGLU 中间维度 | 2048 | 约 2/3 × 4 × d_model |
| Dropout | 0.1 | 小模型正则化 |
| Embed 参数 | 9.2M | vocab=12002 × 768 |
| Transformer 参数 | 116.8M | 18 层 |
| 参数量 | **约 126M**（weight tying） | 不绑定时约 136M |
| Chinchilla 数据量 | ~2.5B tokens | 最优训练量 |
| 预估 PPL | ~22-25 | 取决于数据配比质量 |

**定位**：80M 上快速消融筛选有效改进 → 126M 上验证并产出正式技术报告。
126M 是 Lite 线的旗舰模型，直接对标业界同参数级 SOTA。

---

## 四、项目目录结构

### 历史：V4 单体仓库（已废弃）

> 以下为 Nano V4 初期的单体仓库结构，已于 V4 完成后按 [仓库拆分规划](#仓库拆分规划) 重构为当前的多层结构。**当前实际目录结构详见 [仓库拆分规划](#仓库拆分规划)。**

```
GleamLM/
├── LICENSE                          # Apache 2.0
├── README.md                        # 项目主文档
│
├── models/
│   ├── __init__.py                  # load_model_for_inference()
│   ├── xfind_model.py               # RoPE/RMSNorm/GQA/SwiGLU/Decoder
│   └── xfind_config.py              # 全局配置 + 命令行参数 + 显存指南
│
├── xfind_train.py                   # 训练脚本（AMP + DDP + 梯度累积 + 断点续训）
├── xfind_infer.py                   # 推理脚本（KV Cache + 交互式对话）
├── xfind_dataset.py                 # 数据集（memmap 磁盘映射，~1MB 内存）
├── xfind_quantize.py                # FP16 量化导出
├── xfind_sft.py                     # SFT 指令微调（ChatML 格式 + loss mask）
├── xfind_dpo.py                     # DPO 偏好对齐（策略模型 + 冻结参考）
│
├── tokenizer/
│   ├── sp_tokenizer.py              # V3 旧版分词器（SentencePiece 32K，已由 BBPE 取代）
│   └── checkpoints/                 # bpe_32k.model / .vocab
│
├── inference/
│   ├── sampler.py                   # 采样策略
│   └── streamer.py                  # KV Cache 流式生成器
│
├── evaluation/
│   ├── perplexity.py                # PPL 评估
│   └── generate_samples.py          # 生成样例评测
│
├── tools/
│   ├── download_v3_data.py          # V3 多源数据下载
│   ├── clean_text.py                # 文本清洗/去重/简繁转换
│   ├── build_dataset.py             # 数据格式化为训练样本
│   ├── eval_ppl.py                  # 命令行 PPL 评估
│   └── quick_run.py                 # 冒烟测试/小规模/全量训练
│
├── data/
│   ├── raw/                         # 原始语料
│   └── splits/                      # 训练/验证/测试集
│
├── checkpoints/
│   ├── best_model.pt                # 最优模型
│   ├── checkpoint_epoch_*.pt        # 断点续训
│   └── runs/                        # TensorBoard 日志
│
├── docs/
│   ├── GleamLM-语言模型开发计划.md
│   └── 排坑记录.md
│
└── images/Nano/                      # 训练曲线图
```

---

## 五、开发阶段（全部已完成）

### 阶段一：架构实现 ✅

- [x] RoPE 旋转位置编码（实数优化版）
- [x] RMSNorm 归一化
- [x] GQA 分组注意力（8Q / 4KV）
- [x] SwiGLU 激活结构
- [x] Pre-Norm 完整 Decoder Block
- [x] 权重初始化（参考 Llama 标准）

**产出**：可正常前向/反向的现代 39M 模型

### 阶段二：工程训练体系 ✅

- [x] BF16 混合精度 AMP 训练
- [x] DDP 多卡分布式训练
- [x] Warmup（1%）+ CosineAnnealing
- [x] 梯度累积（batch_size=8, accum=8, 有效 batch=64）
- [x] TensorBoard 监控 + checkpoint 机制
- [x] 断点续训（保存 optimizer/scheduler/scaler 状态）
- [x] 训练速度：4.3 it/s，4h/epoch

**产出**：稳定、高速、可复现的工业级训练链路

### 阶段三：数据体系 ✅

- [x] 中文维基百科数据（modelscope zhwiki）
- [x] 文本清洗/去重/过滤流水线
- [x] BPE 分词器训练（32K 词表，SentencePiece）
- [x] memmap 磁盘映射（内存 7.3GB → 1MB）
- [x] `<|endoftext|>` 文档分隔符（V2）
- [x] 训练/验证/测试集切分

**产出**：~4.56 亿 tokens 中文预训练数据集

### 阶段四：推理与评估 ✅

- [x] KV Cache 推理加速
- [x] Temperature / TopK / TopP / RepetitionPenalty 采样
- [x] 流式文本生成
- [x] 模型性能评测（PPL + 生成样例）
- [x] FP16 量化导出
- [x] 完整项目文档

**产出**：可演示、可部署、可开源的完整 LLM

---

## 六、版本历程（已完成）

### V1（已完成）

| Epoch | Val Loss | PPL | 降幅 |
|-------|----------|-----|------|
| 1 | 4.07 | 47.94 | — |
| 2 | 3.90 | 45.60 | -2.34 |
| 3 | 3.85 | 45.11 | -0.49 |
| 4 | 3.68 | 42.95 | -2.16 |
| 5 | 3.64 | **38.19** | **-4.76** |

配置：label_smoothing=0, stride=512, weight tying, epochs=5

### V2（已完成）

- label_smoothing=0.1, stride=768, epochs=8, 无 weight tying
- PPL=45.01（epoch 4/8）

修复清单：
- [x] label_smoothing=0.1（打破 softmax 尖锐分布，缓解生成循环）
- [x] stride 512→768（减少滑动窗口重叠，降低模板强化）
- [x] 重建训练数据（`<|endoftext|>` 分隔符已生效）

### V3（已完成）✅

> V3 不动架构、不动参数，只改数据——目的是验证数据策略本身有效，避免"数据也改了模型也改了，不知道哪个起作用"的归因混乱。验证后的数据配方原样复用到 80M/126M 和 0.6B。

**设计原则**：

- 数据量紧贴 Chinchilla 最优线（39M → 0.78B），适度超出到 1.2B，给模型更多覆盖面的同时保持单卡可复现
- 每类数据有明确的训练目的——加维基的打底知识、加新闻的句式多样性、加百科的知识密度、加问答的对话模式
- 整个训练流程在 **单张 4070 Ti** 上一个周末之内可完成复现

**数据配比**（0.46B 纯维基 → 1.2B 多源混合）：

| 数据源 | token 估算 | 占比 | 解决什么 |
|--------|-----------|------|------|
| 中文维基 | 0.46B | 38% | 保留，基础知识基底 |
| 中文新闻 | 0.35B | 29% | 现代句式、叙事流畅性 |
| 百度百科 | 0.25B | 21% | 结构化知识、实体密度 |
| 社区问答 | 0.14B | 12% | 对话模式、问题-回答结构 |
| **总计** | **1.2B** | 100% | — |

> Chinchilla 最优 ≈ 0.78B tokens / 39M 参数。1.2B 为 ~31×，适度超出，单卡可复现。

**训练结果**（8 epoch，~3.5 天，4070 Ti 12GB）：

| Epoch | Train Loss | Val Loss | Val PPL | PPL 降幅 |
|-------|-----------|----------|---------|---------|
| 0 | 5.2603 | 3.8242 | 45.80 | — |
| 1 | 4.7488 | 3.7038 | 40.60 | -5.20 |
| 2 | 4.6784 | 3.6490 | 38.44 | -2.16 |
| 3 | 4.6405 | 3.6143 | 37.12 | -1.32 |
| 4 | 4.6147 | 3.5887 | 36.19 | -0.93 |
| 5 | 4.5957 | 3.5702 | 35.55 | -0.64 |
| 6 | 4.5824 | 3.5585 | 35.13 | -0.42 |
| 7 | 4.5746 | 3.5532 | **34.93** | -0.20 |

> 8 epoch 全程无过拟合，val_loss 持续下降。PPL 34.93 为 39M 在 1.2B tokens 上的容量天花板。

修复清单：
- [x] 繁体中文统一转简体（zhconv）
- [x] 下载并清洗新闻语料（0.35B tokens）
- [x] 下载并清洗百度百科（0.25B tokens）
- [x] 下载并清洗社区问答（0.14B tokens）
- [x] 合并 4 源数据，重建训练数据集
- [x] 8 epoch 完整训练 + 生成质量评估
- [x] V1 vs V3 对比（PPL 38.19 → 34.93）
- [x] 生成样例收录（用于 README 展示）
- [x] `<|endoftext|>` 推理截断（排坑 #32）

### V4（已完成）✅ — BBPE + 深窄架构

> V4 不动数据配方（沿用多源混合），集中解决 39M 阶段暴露的 **3 个致命缺陷**。

**V3→V4 三大改动**：

| # | 缺陷（V3） | 修复（V4） | 排坑参考 |
|---|-----------|-----------|---------|
| 1 | `<\|im_start\|>` 等 ChatML token 不存在于词表，被拆成 3 个子词碎片 | BBPE 12K 词表，原生注册 `<\|im_start\|>` `<\|im_end\|>` `<\|endoftext\|>` 为单 token | #32 |
| 2 | 42% 参数（16.4M）浪费在 Embedding"看门"，Transformer 仅 22.6M | 词表 32K→12K，Embed 6.1M→省下 10.3M 加入 Transformer | #33 |
| 3 | 8 层深度对长距离语义建模能力有限，生成文本 token 循环重复 | 8→12 层（深窄架构），Transformer 22.6M→34.6M（+53%） | #17 |

**训练结果**（5 epoch，4源 30/12/43/15，label_smoothing=0.1，bs=4×accum=16）：

| Epoch | Train Loss | Val PPL |
|-------|-----------|---------|
| 0 | — | 16.55 |
| 1 | — | **14.95** |
| 5 | — | ~13.65 |

> Epoch 1 已超越 V3 epoch 7（PPL 34.93），V4 效果是 V3 的 3 倍以上。

**架构参数**：

| 参数 | V3 | V4 |
|------|-----|-----|
| 分词器 | SentencePiece 32K | **BBPE 12K（纯 Python，零依赖）** |
| num_layers | 8 | **12** |
| d_model | 512 | 512 |
| Q heads / KV heads | 8 / 4 | 8 / 4 |
| d_ff | 1365 | 1365 |
| QK-Norm | ❌ | ✅ |
| Embed 参数量 | 16.4M (42%) | **6.1M (15%)** |
| Transformer 参数量 | 22.6M (58%) | **34.6M (85%)** |
| **总参数** | ~39M | **~40.7M** |

> **设计原则**：总参数量不变，把"看门"的 10M 参数转移到"思考"层。Transformer 大脑扩大 53%。

**已完成清单**：

- [x] BBPE Tokenizer 12K（纯 Python，~175 行）
- [x] 3 个特殊 token 注册（`<|im_start|>`/`<|im_end|>`/`<|endoftext|>`）
- [x] QK-Norm（LLaMA3 标准）
- [x] 12 层深窄架构（d_model=512, num_layers=12）
- [x] 4 源数据 30/12/43/15 字符加权配比
- [x] 全量数据 BBPE 12K 重新分词
- [x] 5 epoch 预训练（PPL 13.65）
- [x] label_smoothing=0.1, stride=768, weight_tying=True

---

### V2 归因分析

V2 同时改变三个变量（label_smoothing↑、stride↑、weight_tying↓），导致 PPL 上升的**归因不唯一**：

| 变量 | 方向 | 对 PPL 的理论影响 |
|------|------|------------------|
| label_smoothing 0→0.1 | 提高 | 训练 loss 系统性偏高，PPL ↑ |
| stride 512→768 | 提高 | 评估重叠减少，PPL 更真实，↑ |
| weight_tying on→off | 去掉 | 正则化消失，小模型易过拟合，PPL ↑ |

> **教训**：80M 阶段每个对照实验只改变一个变量，保持归因清晰。

### V4 架构验证结论

- Pre-Norm + RMSNorm + RoPE + GQA + SwiGLU + **QK-Norm** 在小模型上稳定可训
- BF16 AMP + DDP + 梯度累积工程链路成熟
- **12 层深窄**（512d）比 8 层宽窄能更好利用同等参数量（PPL 34.93 → 13.65）
- **BBPE 12K** 在 40M 模型上的 Embed 占比仅 15%（vs 32K BPE 的 41%），参数效率大幅提升
- **Deep-Narrow（12×512）** 已验证完成，87M 保持 12 层并聚焦 FFN 扩容
- 纯文本数据缺少代码，限制逻辑推理能力（80M 加入代码数据）

---

## 七、仓库拆分规划

> 80M 阶段 **不拆多包**，保持 mono-repo 体内建子目录。目标：教学版与实验版共享核心库，训练脚本和配置各自独立，互不污染。

### 拆分后目录结构（到文件，2026-06-28 提纯后同步）

```
GleamLM/                              # mono-repo
├── LICENSE
├── README.md
├── .gitignore
├── .vscode/settings.json
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                    # gleamlm pip 包配置
│
├── gleamlm/                           # 共享核心库（项目根可直接 import）
│   ├── __init__.py                    # load_model_for_inference()
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py                   # GleamLMModel (+use_flash_attn)
│   │   └── config.py                  # ModelConfig + 绝对路径常量
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── tokenizer.py               # BBPE Tokenizer（纯 Python 零依赖）
│   │   └── checkpoints/bbpe_12k/       # V4 词表文件
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── dataset.py                 # LMDataset + collate_fn
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── sampler.py                 # Temperature / TopK / TopP
│   │   └── streamer.py                # KV Cache 流式生成
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py
│       └── logging.py
│
├── gleamlm-nano/                     # 40M 教学版（import gleamlm.*）
│   ├── train.py
│   ├── infer.py
│   ├── sft.py
│   ├── dpo.py
│   ├── quantize.py
│   ├── quick_test_sft_dpo.py
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   └── generate_samples.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_model.py
│   │   ├── test_tokenizer.py
│   │   └── test_dataset.py
│   ├── data/
│   │   ├── raw/                      # wiki/baike/news/qa *.txt (已被全局 data/ 替代)
│   │   └── splits/                   # (已被 data/nano_data/ 替代)
│   └── checkpoints/                  # 模型权重 + TensorBoard runs
│
├── gleamlm-lite/                     # 87M 实验版（import gleamlm.*）
│   ├── train.py                      # Cosine LR + FlashAttn + Z-Loss
│   ├── infer.py                      # 87M 推理（2048 context）
│   ├── test_train.py                 # 轻量训练冒烟测试
│   ├── make_small_data.py            # 小数据集生成工具
│   ├── sft.py                        # SFT 指令微调
│   ├── dpo.py                        # DPO 偏好对齐
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   └── generate_samples.py
│   ├── data/
│   │   └── splits/                   # 轻量测试数据
│   └── checkpoints/                  # 87M 训练产出
│
├── gleamlm-pro/                      # 126M / 0.6B 远期占位
│   └── .gitkeep
│
├── scripts/                          # nano / lite 共享辅助脚本
│   ├── eval_ppl.py                   # 命令行 PPL 评估
│   ├── eval_knowledge.py             # 知识评估
│   ├── eval_layer_dropout.py         # 层 dropout 测试
│   ├── quick_run.py                  # 冒烟/小数据测试
│   ├── check_ckpt.py                 # Checkpoint 诊断
│   ├── verify_both.py                # 40M+87M 双模型验证
│   ├── verify_lite.py                # 87M 单独验证
│   ├── verify_paths.py               # 路径解析验证
│   ├── _check_fanti.py               # 繁简校验（内部）
│   ├── _check_stats.py               # 数据统计（内部）
│   └── _read_tb.py                   # TensorBoard 读取（内部）
│
├── data_tools/                       # 数据处理管线（共享）
│   ├── download_data.py              # 多源数据下载指引
│   ├── clean_text.py                 # 文本清洗 + 简繁转换
│   ├── dedup_text.py                 # MD5 exact/prefix 去重
│   ├── filter_qa.py                  # QA 专项过滤
│   ├── prepare_data.py               # 一键管线（清洗→去重→混合→切分）
│   ├── build_dataset.py              # 流式多源混合 + split
│   ├── extract_parquet.py            # Parquet → txt 转换
│   ├── generate_sft_data.py          # DeepSeek API 蒸馏 SFT（全量）
│   ├── generate_sft_data_full.py     # 全量 10000 条 SFT
│   ├── gen_sft.py                    # 精简版 SFT 生成
│   ├── generate_rejected.py          # 基模型生成 DPO rejected
│   ├── clean_sft_data.py             # SFT 数据清洗
│   ├── make_small_nano.py            # Nano 小数据集生成
│   └── make_small_data.py            # Lite 小数据集生成
│
├── data/                             # 全局共享数据
│   ├── .gitkeep
│   ├── raw/                           # 原始语料（wiki/baike/news/qa）
│   │   └── .gitkeep
│   ├── nano_data/                     # Nano 训练/验证/测试 + .npy 缓存
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   ├── lite_data/                     # Lite 训练/验证/测试 + .npy 缓存
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   ├── chinese-fineweb-edu/           # HuggingFace 下载的 edu 语料（Parquet）
│   ├── sft_data.jsonl                 # SFT 训练数据（10000条）
│   ├── sft_data_clean.jsonl           # 清洗后 SFT 数据
│   ├── dpo_data.jsonl                 # DPO 训练对
│   └── 563w_baidubaike.json.7z        # 百度百科原始压缩包
│
├── images/                           # 训练曲线
│   └── Nano/                         # Nano 版训练曲线 SVG
│
├── docs/                             # 全局文档
│   ├── GleamLM-Lite-语言模型开发计划.md
│   ├── GleamLM-训练优化参考.md
│   ├── GleamLM-Pro-开发计划.md
│   ├── GleamLM-0.6B-开发计划.md
│   ├── GleamLM_vs_MiniMind.md
│   ├── GleamLM_README.md
│   ├── GleamLM宣传.md
│   ├── 排坑记录.md
│   ├── 测试报告.md                   # 40M 评估报告
│   ├── 课程文档.md
│   ├── 第6章 大语言模型.md
│   └── 预告.txt
│
├── assets/
│   └── GleamLM.png
```

### 与原 7 包计划的差异

| 原计划 | 处理 |
|------|------|
| `gleamlm-cli/`（13 个 CLI 工具） | ❌ 删除。功能零实现，先做工具再打包 |
| `gleamlm-viz/`（注意力热力图） | ❌ 删除。功能未实现 |
| `gleamlm-convert/`（ONNX 导出） | ❌ 删除。远期需求 |
| `gleamlm-pro/`（0.6B 详细结构） | ❌ 简化为 `.gitkeep` 占位 |
| 总包 `xfind`（一键装 7 包） | ❌ 删除。无实际需求 |
| `sentencepiece` 依赖 | ❌ V4 已废弃，不写进 setup.py |
| `gleamlm-core/` | ✅ **已执行。提纯到根 `gleamlm/` 目录**，所有脚本统一 `from gleamlm...` |

### 拆分原则

| 原则 | 说明 |
|------|------|
| **80M 不拆** | 80M 实验无需代码重构，只新建 `gleamlm-lite/` 添加训练脚本和 configs |
| **先跑通再提纯** | 等 80M 训练完成后，从 nano 提取模型/分词/数据集到 `gleamlm-core/` |
| **教学不降级** | nano 保持单文件训练脚本，注释密度不变，新手 10 分钟能跑通 |
| **实验不加锁** | lite 可以上 FlashAttn / WSD / Z-Loss，不受 nano 的保守配置约束 |
| **数据可复用** | nano 验证过的 V4 数据配比和 SFT/DPO 参数，lite 直接复用 |

### 拆分时机

```
现在 → 80M 训练完成前：不拆包，保持当前 mono-repo
80M 完成后：提取 gleamlm-core/（模型+分词器+数据集）
126M 启动时：lite 可根据需要独立依赖 core
0.6B 真启动时：再考虑 pro 独立的目录结构
```

### gleamlm-core 模块规划（80M 完成后再做）

> core 只放**与模型规模无关的通用能力**：模型定义、分词器、数据集、推理采样、checkpoint 工具。耦合训练框架或应用逻辑的不进 core。

| 进 core | 不进 core |
|---------|----------|
| `models/model.py` + `config.py` | FSDP / 梯度检查点 / 并行策略 |
| `tokenizer/tokenizer.py`（BBPE） | API 服务 |
| `dataset/dataset.py` | 数据处理管道（`data_tools/`） |
| `inference/sampler.py` + `streamer.py` | SFT / DPO 训练脚本 |
| `utils/checkpoint.py` | 量化 / 蒸馏脚本 |

> 不引入 `sentencepiece` 依赖 —— V4 已完全迁移到 BBPE（纯 Python 零依赖）。

---

## 八、GleamLM-Lite（87M）开发计划

**目标**：在 V4 40M 验证基线上，将模型扩大至 ~87M。核心任务是解决 40M 事实知识溃泛（测试报告证实：SFT 模型 50 题正确率仅 4%，实体一致性 0%）。

> **基线对齐**：以下所有实验以 GleamLM-Nano **V4**（12 层 × 512d × BBPE 12K × QK-Norm，PPL 13.65）为对照基线。40M 的 12 层已被测试证实为中文生成的最低存活阈值，87M 保持 12 层并全力扩 FFN。

### 前置资产：V4 全链路已验证通过

87M 不是从零开始。以下组件已通过 V4（PPL 13.65）验证，87M 零改动复用：

| 组件 | 复用方式 |
|------|------|
| 训练管线（AMP/DDP/断点续训/梯度累积） | 改 config 即用 |
| 数据管线（download→clean→dedup→char-weight→split） | 全量复用 |
| **BBPE Tokenizer 12K（纯 Python，零依赖）** | **不改，直接复用** |
| 评估管线（PPL/生成样例/TensorBoard） | 不改 |
| SFT / DPO 脚本（ChatML + loss mask） | 改 max_seq_len 即用 |
| QK-Norm（LLaMA3 标准） | 不改 |
| 排坑记录（52 条） | 不会再踩 |
| 测试报告（见 `测试报告.md`） | 12 层不可少、知识只在 FFN |

87M 新增改动：`d_model=768`、`max_seq_len=2048`、`d_ff=2048`、训练策略优化。**词表保持 12K**——测试证实参数应投给 FFN 而非 Embedding。

### 参数规格

#### 主实验配置（12L × 768d × d_ff=2048, ~87.1M）

| 参数 | V4 Nano (40M) | Lite 87M | 变化 |
|------|:---:|:---:|------|
| d_model | 512 | **768** | ↑ 50% |
| num_layers | 12 | 12 | **不变**（测试证实 12 层是存活阈值） |
| num_heads | 8 | **12** | head_dim=64 保持不变 |
| num_kv_heads | 4 | **6** | GQA 2:1 |
| d_ff (SwiGLU) | 1365 | **2048** | 标准公式 8/3×768 |
| vocab_size (BBPE) | 12,002 | **12,002** | **不变，复用 V4** |
| max_seq_len | 1024 | **2048** | 2× |
| RoPE θ | 10,000 | **10,000** | 不变（待验证是否需要提升） |
| weight_tying | True | True | 保留 |
| dropout（预训练） | 0.1 | **0.0** | 恢复容量 |
| QK-Norm | ✅ | ✅ | 不动 |
| **总参数** | ~40.7M | **~87.1M** | 2.1× |
| Embed 参数 | 6.1M (15%) | 9.2M (11%) | 占比下降 |
| FFN 参数 | 16.8M (41%) | **56.6M (65%)** | **3.4×** |

> 设计原则：**测试证实 12 层是中文生成的硬阈值，且事实知识 100% 存于 FFN**。因此方案是：保持 12 层，d_model 扩到 768，d_ff 按标准 SwiGLU 公式 8/3×768→2048（天然对齐 32×head_dim=32×64），词表不再扩容。FFN 容量达 40M 的 3.4 倍——这是测试给的硬约束下的数学唯一解。

#### 消融实验设计

在 12×768 主配置下设置三项消融实验：

| 实验 | 变量 | 配置 | 对比问题 |
|------|------|------|------|
| **M0** | （主实验） | 12×768, d_ff=2048, BBPE 12K | 与 V4 对比：宽化 FFN 的知识改善 |
| **M1** | LR 调度 | Cosine vs **WSD** | WSD 相比当前 Cosine 默认的收益 |
| **M2** | 代码数据 | 5 源含代码 vs 4 源 | 代码数据的推理能力增益 |

> **已移除 M1 词表消融**：测试证实 FFN 是知识唯一载体，增词表=抢 FFN 预算，与核心目标冲突。12K 词表已通过 V4 验证，直接复用。

### 训练配置

#### 核心训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | betas=(0.9, 0.95), eps=1e-8 |
| 学习率峰值 | **4e-4** | 87M 梯度方差小于 40M，安全上提 |
| LR 调度 | **Cosine**（CosineAnnealing + Warmup） | 标准余弦退火。WSD 为后续消融选项 |
| Warmup | 2% | 约 1500 步 |
| 精度 | BF16 AMP | |
| Attention | **`F.scaled_dot_product_attention`** | Flash Attention 自动后端 |
| 有效 batch | 64 | bs=4 × accum=16（12GB 安全水位 85%） |
| 梯度裁剪 | 1.0 | 保留 |
| Weight Decay | 0.01 | 保留 |
| Label Smoothing | **0.1** | V4 验证有效 |
| Weight Tying | True | 保留 |
| Dropout（预训练） | **0.0** | 等效恢复 ~10% 容量 |
| Z-Loss 正则化 | **1e-4** | 防 logit 爆炸 |

#### 与 40M V4 训练策略的差异

| 策略 | 40M V4 | 87M Lite | 原因 |
|------|--------|----------|------|
| d_model | 512 | 768 | 加宽 |
| d_ff | 1365 | 2048 | 标准公式 8/3×768，3.4× 知识容量 |
| head_dim | 64 | 64 | 不变 |
| vocab_size | 12K BBPE | 12K BBPE | 不变，复用 V4 |
| max_seq_len | 1024 | 2048 | 2× 上下文 |
| RoPE θ | 10,000 | 10,000 | 不变 |
| 学习率峰值 | 3e-4 | 4e-4 | 更大模型梯度方差小 |
| LR 调度 | Cosine | **Cosine** | 标准余弦退火，WSD 为后续消融选项 |
| warmup_ratio | 1% | 2% | 更大模型需更稳启动 |
| dropout（预训练） | 0.1 | **0.0** | 恢复容量 |
| Flash Attention | 手写 | **`F.scaled_dot_product_attention`** | 省显存 +10% 速 |
| 有效 batch | bs=4×accum=16 (64) | bs=4×accum=16 (64) | 不变 |
| Z-Loss | 无 | **1e-4** | 训练稳定性 |
| epoch 策略 | 固定 5 | **2 epoch 起步 + 按需扩展** | 灵活决策 |
| 训练阶段 | 单阶段全源 | **单阶段全源**（两阶段训练为后续优化选项） | 当前首次训练保持简单 |

### 数据策略

#### 数据配比（5源 + edu — 2026-06-30 已实施）

> **废弃**原 4 源 V4 配比（wiki 30% / baike 12% / news 43% / qa 15%）。
> 引入 Chinese FineWeb Edu 数据集（HuggingFace `chinese-fineweb-edu`，教育级质量过滤网页文本），解决 87M 模型数据量不足问题（Chinchilla 最优 ~1.74B tokens，原有仅 ~1.09B tokens）。

| 数据源 | token 估算 | 字符配比% | 文件大小 | 训练目的 |
|--------|-----------|:---:|------|------|
| **Chinese FineWeb Edu** | **~1.5B** | **35%** | 5.8 GB (清洗后) | **主力教育语料**，高质量多样化中文文本 |
| 中文新闻 | ~870M | 20% | — | 现代句式、叙事流畅性 |
| 中文维基 | ~870M | 20% | — | 基础知识基底，与百科各 20% 平衡 |
| 百度百科 | ~650M | 15% | — | 结构化知识，Nano 验证过的高效知识源 |
| 社区问答 | ~435M | 10% | — | 对话模式、问题-回答结构 |
| **总计** | **~4.3B** | **100%** | **train.txt 13.85 GB** | Chinchilla 最优 1.74B，当前 2.5× 超出 |

> **数据目录**：`data/lite_data/`（Lite 专用，Nano 原 `data/nano_data/` 不动）
> 
> **处理管线**：
> 1. `data_tools/extract_parquet.py` — Parquet → txt（pyarrow 流式逐文件）
> 2. `data_tools/clean_text.py` — 轻清洗（长度过滤 30-5000，其他过滤全关）
> 3. `data_tools/build_dataset.py` — 五源字符加权配比混合 → train/valid/test 切分
>
> **配比理由**：
> - Edu 50%：主力数据，教育级质量，数据量充足
> - News 20%：保留句式多样性（行均 752 字，效率高）
> - Wiki+Baike 20%：从 63% 压缩到 20%，大幅降低重叠冗余（排坑 #40）
> - QA 10%：保留对话能力

> Edu 数据来源：[opencsg/chinese-fineweb-edu](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu)，IndustryCorpus 子集 40 文件 ~37 GB，当前取前 5 文件 ~4.6 GB 测试配比。后续可扩展至全量。

#### 数据量规划（已更新）

| 参考标准 | 数据量 | 说明 |
|---------|--------|------|
| Chinchilla 最优 | ~1.74B tokens | 87M × 20 |
| 推荐训练量 | **3-5B tokens** | 多 epoch，WSD 的 Stable 阶段反复学习 |

#### 训练耗时

> 注意：80M 用 bs=4（非 8）。参照 40M V4 实测（bs=4, seq=1024, ~15h/epoch），80M 的 2× 宽度 + 2× seq_len 计算量约 4×，以下基于此校准。

| GPU | 速度 | 说明 |
|-----|------|------|
| 1×4070 Ti (12GB) | ~1.5-2 it/s | bs=4, seq=2048, 768d |
| 4×48GB (Linux DDP) | ~6-8 it/s | 4 卡加速（Linear 估算） |

| 数据量 | 1×4070 Ti 每 epoch | 1×4070 Ti 4 epochs |
|--------|-------------------|-------------------|
| 1.5B | ~50-60h | ~8-10 天 |
| 3B | ~100-120h | ~17-20 天 |

> **策略**：以 4 epochs 为第一轮观察点。单卡 12GB 可训但耗时长（参照 40M V4 的 ~15h/epoch），建议 **4×48GB Linux DDP 加速**（单卡 ~10-15h/epoch）。

#### 数据清洗管线（L1-L3，V4 已实现）

```
L1: 编码统一（UTF-8）+ 繁体转简体（zhconv）
L2: 规则清洗（min_len/max_len/中文占比/广告过滤/Wiki垃圾条目）
L3: 去重（exact MD5 / prefix / Q-hash，按数据源选择策略）
L4: 跨文档 MinHash 去重（80M 新增，防 train/valid 泄漏）
L5: 字符加权行数配比换算（prepare_data.py）
```

> L1-L3 和 L5 已在 V4 实现。L4 跨文档 MinHash 去重为 87M 新增项。

### 词表策略

**路线**：**直接复用 V4 BBPE 12K，不扩容。**

| 决策 | 依据 |
|------|------|
| 保留 12K | V4 已验证词表效率：Embed 占比从 41%（32K BPE）→ 15%（12K BBPE） |
| 不扩 16K | 测试证实 FFN 是知识唯一载体。扩词表 = 从 FFN 抢 3.1M 参数给 Embedding"看门"，与核心目标冲突 |
| 收益 | nano 和 lite 词表统一，SFT/DPO 数据可直接跨版本复用，零迁移成本。无需训练 16K tokenizer、无需重建 .npy |

#### 远期：领域种子词表（Vocabulary Seeding）

> **当前不实施**，待 Lite 训练完成、评估专业领域表现后再决定。

**思路**：不扩大 BPE 合并 slot 数，而是直接在词表中追加预定义的领域专有名词为独立 token。

```
词表结构（vocab_size = 12001 + 1000 = 13001）：

ID 范围        内容                 来源
0-255          256 字节基座          固定
256-1255       1000 个领域种子       手动指定（医学/法律/CS 高频词）
1256-12999     BPE 合并区            统计频率竞争（11744 个 slot，与原来相同）
13000-13008    9 个特殊 token        末尾追加
```

**种子词来源**：从 edu 数据跑 TF-IDF 提取 top-1000，或按医学/法律/CS 三领域人工整理。

| 领域 | 示例种子词 |
|------|-----------|
| 计算机科学 | 机器学习、神经网络、embedding、transformer、反向传播 |
| 医学 | 高血压、糖尿病、抗生素、免疫系统、病理学 |
| 法律 | 知识产权、民事诉讼、行政处罚、司法解释、合同纠纷 |

**实现改动**：`BBPETokenizer.train_from_files()` 增加 `seed_tokens` 参数，分配种子 ID 后 BPE 合并自动跳过已占 slot。Embedding 增长 1000 × 768 = 0.77M 参数（87M 的 0.9%，可忽略）。

**预期收益**：领域专有名词成单 token，不被 BPE 碎片化，语义保真度更高。尤其对知识密集型的 edu 数据效果显著。

### 断点与决策流程

> **当前默认使用 Cosine LR。** WSD 为后续消融选项。

```
Epoch 1-2: 观察 loss 下降速度
  ↓ loss 下降正常
Epoch 2: 评估 PPL + 生成样例
  ↓ PPL < 20 且生成连贯
→ 保存最佳模型，可选提前停止
  ↓ PPL ≥ 20 或生成质量不佳
→ 扩展更多 epoch 或启动 WSD 消融实验
```

### 评估方案

#### 核心指标

| 指标 | 工具 | 目标 |
|------|------|------|
| PPL (perplexity) | `evaluation/perplexity.py` | **< 12**（V4 基线 13.65，80M 叠加优化后目标） |
| Distinct-1/2 | `evaluation/generate_samples.py` | 评估生成多样性 |
| 人工可读性评分 | 生成样例 + 人工判断 | 300+ 字连贯段落，减少事实幻觉 |

#### 对比评估矩阵

| 对比维度 | 实验组 | 对比基线 | 核心问题 |
|---------|-------|---------|------|
| FFN 扩容 | 87M M0 (12L, d_ff=2048) | V4 (12L, d_ff=1365) | FFN 3.4× 能否解决事实知识溃泛？ |
| 调度器效应 | 87M M0 (Cosine) | 87M M1 (WSD) | Cosine vs WSD 的差异 |
| 代码数据效应 | 87M M2 (+code) | 87M M0 (4源) | 代码数据对推理的提升 |
| 规模效应 | 87M M0 | V4 40M | Scaling Law 验证 |
| 两阶段训练 | Stage2 纯知识数据 | 全源单阶段 | 知识数据集中训练的效果 |

### 任务清单与优先级

#### P0 — V4 40M 收官（预训练已 ✅，SFT 已跑通）

> **状态**：V4 40M 预训练已完成（PPL 13.65），SFT/DPO 全链路已验证通过。测试报告已出具（事实准确率 4%，12 层不可少）。

**SFT 数据配比**（DeepSeek API 蒸馏生成，10,000 条，标准 ChatML 格式）：

| 类别 | 占比 | 条数 | 内容范围 |
|------|:---:|------|----------|
| A 类 · 通用问答 | 40% | 4000 | 烹饪、家务、健康、学习、科技、旅行 |
| B 类 · 知识回答 | 30% | 3000 | 历史、地理、科学、文化（模板扩展） |
| C 类 · 创作与闲聊 | 30% | 3000 | 描写、感悟、聊天、观点 |

| # | 任务 | 说明 | 状态 |
|---|------|------|:--:|
| 1 | SFT 指令微调 | 3 epochs, lr=5e-6, batch=8×accum=4 | ✅ |
| 2 | SFT 效果评估 | loss 3.33→2.2，模型成功从续写→对话 | ✅ |
| 3 | DPO 偏好对齐 | 150 对, β=0.1, lr=1e-7, rejected 由 SFT 模型生成 | ✅ |
| 4 | 知识测试评估 | 50 题填空准确率 4%，证实 40M 知识含量趋近于零 | ✅ |
| 5 | 层 dropout 测试 | 证实 12 层是中文生成硬阈值，8L 不可行 | ✅ |

> SFT/DPO 经验验证：lr=5e-6 适合小数据量。DPO 用 SFT 模型生成 rejected 比用预训练模型效果更好（排坑 #36）。87M 直接复用全部配方，词表统一无需迁移。

#### P1 — 87M 阻塞项（全部为代码改动，无需数据重建）

| # | 任务 | 说明 |
|---|------|------|
| 6 | 代码数据收集与清洗（M2 消融实验用） | 0.20B tokens 代码语料准备 |

#### P2 — 87M 核心实验

| # | 任务 | 实验 | 产出 |
|---|------|------|------|
| 7 | 主实验训练（12L×768d, d_ff=2048, 12K, Cosine LR） | M0 | PPL + 生成质量基线 |
| 8 | LR 调度消融（WSD 替代 Cosine） | M1 | Cosine vs WSD 的定量对比 |
| 9 | 2 epoch 中间评估与决策 | M0 | 确定是否扩展训练或启动 M2 |

#### P3 — 优化实验

| # | 任务 | 实验 | 触发条件 |
|---|------|------|---------|
| 10 | 代码数据消融（5 源混合） | M2 | M0 PPL 达预期后验证代码增益 |
| 11 | 两阶段训练（Stage2 纯知识数据） | M0 | 知识准确率不足时启动 |
| 12 | 5B tokens 长训练 | M0 | 4 epoch 后 loss 仍在快速下降 |

#### P_post — 87M 训练完成后，代码健康度重构（待 87M 主实验完成后执行）

> 以下任务全部为纯代码层面重构，不动模型架构和训练配方。目的：消除 nano/lite 之间 ~2000+ 行的复制粘贴代码，提升可维护性。
>
> **标记说明**：✅ 已完成 | 📋 待执行

##### 已完成（2026-07-05）

| # | 任务 | 说明 | 状态 |
|---|------|------|:--:|
| R5 | **WSD scheduler 复活** | 注释死的 WSD 代码正式实现到 `gleamlm/utils/torch_utils.py`（与 `get_lr_cosine` 并列），删除 `train.py`/`test_train.py`/`smoke_test.py` 中的重复注释块 | ✅ |
| R6 | **unused import 清理** | 删除 6 处未使用的 `import sys`/`import os`（nano/lite 的 sft.py、dpo.py、tests/test_model.py） | ✅ |
| R8 | **JSON 解析错误处理** | `SFTDataset.__init__` 和 `DPODataset.__init__` 的 `json.loads(line)` 加 try/except，Warning + 跳过损坏行而非崩溃 | ✅ |

##### 小改动 — 可立即执行（零风险）

| # | 任务 | 说明 | 改动量 |
|---|------|------|:--:|
| R14 | **修复 pin_memory + num_workers=0 无效组合** | 6 个训练脚本（train.py×2, sft.py×2, dpo.py×2）的 DataLoader 均设置了 `pin_memory=True, num_workers=0` — `pin_memory` 在 0 worker 时是无效的。改为 `pin_memory=False` 或启用 `num_workers=2` | 极小 |
| R15 | **修复 dpo.py evaluate_dpo 未定义变量** | `gleamlm-nano/dpo.py` 中 `epochs==0` 时 `avg_loss` 未初始化但被 `torch.save()` 引用，加 `avg_loss = 0.0` 初始值 | 极小 |
| R16 | **sys.argv 全局修改去副作用** | `gleamlm-nano/train.py`（line 154-156）中 `sys.argv = [...]` 直接修改全局变量，改为用 argparse `set_defaults` 或解析前插入 | 小 |
| R17 | **GradScaler 添加 CPU 回退** | 训练脚本中 `GradScaler('cuda')` 在 CPU 环境下会失败，加 `if torch.cuda.is_available()` 条件检查 | 极小 |

##### 小改动 — 需少量测试

| # | 任务 | 说明 | 改动量 |
|---|------|------|:--:|
| R4 | **infer.py 统一** | `load_model` + `generate` + `interactive` + `main` 提取为 `gleamlm/inference/cli.py`，nano/lite infer.py 只留模型路径默认值 | 小 |
| R7 | **统一路径管理** | 创建 `gleamlm/utils/paths.py`，统一 `_CHECKPOINT_DIR` / `DEFAULT_CHECKPOINT_DIR` / `_SCRIPT_DIR` 等分散在 10+ 文件中的路径常量 | 小 |
| R10 | **checkpoint 架构校验** | 加载 checkpoint 前加 `assert_same_architecture()` 校验，避免 `strict=True` 加载不匹配模型时崩溃（6 处：sft.py×2, train.py×2, dpo.py×2） | 小 |
| R18 | **generate_samples 行为一致性验证** | lite 版已修复 text cover bug；确认 nano 版 `generated.append(chunk)` + `''.join()` 行为与修复后 lite 版一致（当前 nano 版本正确，仅需 double-check） | 验证 |

##### 中等改动 — 87M 训练完成后分批执行

| # | 任务 | 说明 | 改动量 |
|---|------|------|:--:|
| R9 | **训练脚本加类型注解** | `sft.py` / `dpo.py` / `train.py` 的 `train_one_epoch`、`evaluate` 、 `main` 等函数增加完整类型注解（参考 `gleamlm/inference/streamer.py` 风格） | 中等 |
| R1 | **SFT 基类抽提** | `SFTDataset` + `train_one_epoch` + `evaluate_sft` + `generate_response_sft` 提取到 `gleamlm/training/sft_trainer.py`<br>接口：`SFTTrainer(model, tokenizer, data_path, max_seq_len, lr, epochs, batch_size, save_dir, device) -> .train() / .evaluate()`，nano/lite 的 sft.py 变为 ~20 行配置包装 | 中等 |
| R2 | **DPO 基类抽提** | `DPODataset` + `dpad_collate` + `compute_log_probs` + `dpo_loss` + `get_reference_logps` + `train_one_epoch` + `evaluate_dpo` 提取到 `gleamlm/training/dpo_trainer.py`<br>接口：`DPOTrainer(model, ref_model, tokenizer, data_path, max_seq_len, lr, epochs, batch_size, beta, save_dir, device) -> .train() / .evaluate()` | 中等 |
| R3 | **train.py 统一** | `evaluate()`、 `save_checkpoint()`、 `load_checkpoint()`、 `get_optimizer_and_scheduler()` 提取为 `gleamlm/training/base_trainer.py`，nano/lite train.py 共享 | 中等 |
| R11 | **合并 SFT 数据生成脚本** | `data_tools/gen_sft.py`（250+ 硬编码 QA 对，模板变体生成）和 `data_tools/generate_sft_data.py`（DeepSeek API 蒸馏）功能重叠，合并为一个工具，减少 300+ 行重复逻辑 | 中等 |
| R12 | **打 git tag** | 给当前版本打 `v0.1.0` 标签，固化 `setuptools_scm` 版本号 | 极小 |

#### 重构后目录结构（预期）

```
gleamlm/
├── training/                  # 新增：共享训练模块
│   ├── __init__.py
│   ├── base_trainer.py        # optimizer/scheduler/checkpoint 通用逻辑
│   ├── sft_trainer.py         # SFTDataset + SFT 训练/评估
│   └── dpo_trainer.py         # DPODataset + DPO 训练/评估
├── inference/
│   ├── __init__.py
│   ├── sampler.py
│   ├── streamer.py
│   ├── generate.py
│   └── cli.py                 # 新增：统一的 CLI 推理入口
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── torch_utils.py
│   └── paths.py                # 新增：统一路径常量
│
gleamlm-nano/                   # 精简为配置 + 入口脚本
│   ├── train.py                # ~30 行配置 → 调 base_trainer
│   ├── sft.py                  # ~20 行配置 → 调 sft_trainer
│   ├── dpo.py                  # ~20 行配置 → 调 dpo_trainer
│   └── infer.py                # ~15 行配置 → 调 cli
│
gleamlm-lite/                   # 同上结构
│   ├── train.py
│   ├── sft.py
│   ├── dpo.py
│   └── infer.py
```

#### 重构原则

| 原则 | 说明 |
|------|------|
| **只提纯不动逻辑** | 复制到共享模块时不做行为改动，先提纯再优化 |
| **nano/lite 行为不变** | 重构后 `python gleamlm-nano/sft.py` 和之前行为完全一致 |
| **先跑通再提纯** | 等 87M 所有训练实验完成后再做，避免引入重构风险影响训练进度 |
| **测试先行** | 重构前补全测试覆盖，重构后跑全量测试确认无回归 |

#### P_doc — 文档补全

| # | 任务 | 说明 | 状态 |
|---|------|------|:--:|
| 13 | **Nano → Lite 架构演进文档** | 为什么 FFN 扩 3.4×、为什么加 Z-Loss、WSD vs Cosine 为什么选 WSD、Flash Attention 收益 | 📋 待写 |
| 14 | **项目架构设计说明** | 教学/工程双层设计思路，核心库抽离原则，数据隔离设计 | 📋 待写 |
| 15 | **Jupyter 走读 notebook** | 逐行讲解 GleamLMModel 前向过程，适合课堂教学 | 📋 待写 |
| 16 | **Lite 训练日志** | 补充 Lite 实际训练结果（PPL 曲线、生成样例、与 Nano 对比） | 📋 训练后补充 |

### 预期效果

- PPL 目标：**< 10**（V4 40M 已 13.65，FFN 3.4× 扩容应显著突破）
- FFN 3.4× 扩容预期 PPL 降 2.0-3.0；WSD + dropout=0 + Z-Loss 叠加再降 1.5-2.0
- 事实准确率目标：从 40M 的 4% → **≥30%**（FFN 3.4× 的物理容量提升 + 两阶段训练）
- 生成质量：中文流畅，减少事实幻觉
- 显存安全：bs=4 时 ~8-9 GB，单张 12GB 显卡可跑（水位 ~75%）

### 显存预估

| 配置 | 显存占用 (bs=4) | 单卡 12GB |
|------|:---:|:---:|
| M0 主实验 12L×768d, d_ff=2048, 12K | ~8-9 GB | ✅ 安全（排坑 #45：水位 85% = 10.2GB） |
| M2 代码数据（同 M0 模型） | ~8-9 GB | ✅ |

### 长期迭代路线

| 版本 | 参数量 | 定位 | 硬件 | 状态 |
|------|--------|------|------|------|
| GleamLM-Nano V4 | ~40.7M | Deep-Narrow (12×512) 教学基线 | 单卡 12GB | ✅ 已完成 |
| GleamLM-Lite | ~87.1M | 12L×768d, d_ff=2048, FFN 3.4× | 单卡 12GB | 🔨 训练中 |
| GleamLM-Lite-126M | ~126M | Widen+Deep (18×768) 旗舰 | 单卡 12GB / 4×48GB | 📋 规划中 |
| GleamLM-Pro-0.6B | ~0.6B | 工业级验证 | 多卡集群 | 📋 寻求合作 |

---

## 九、项目核心亮点

1. **架构先进**：SwiGLU / GQA / RoPE / RMSNorm，完全对标 Llama/Qwen
2. **工程成熟**：AMP、DDP、梯度累积、断点续训全套工业配置
3. **硬件友好**：单卡 12GB 即可完整训练，4h/epoch
4. **教学价值极高**：从零手写、每行可讲、代码即教材
5. **可无限扩展**：原生支持模型并行，40M → 87M → 126M → 0.6B
6. **完整开源**：数据流水线、训练脚本、推理代码、评估工具、排坑记录全部公开

---

## 十、许可证

**代码**：Apache License 2.0，可自由用于教学、研究、商业应用。

---

## 十一、gleamlm-cli 工具生态（远期）

> 以下为远期规划，待 80M 训练完成、gleamlm-core 稳定后再启动。当前零实现。

13 个规划中的命令行工具（模型诊断 / 速度基准 / 输出质量测评 / 注意力可视化 / Checkpoint 管理 / 分词器分析 / 显存剖析 / 多模型对比 / 模型格式互转 / 云 API 基准 / 本地云端混合对比 / 多供应商评测 / API 供应商信息）。

> 工具待功能实现后再打包，不在 80M 阶段占用时间。

---

## 十三、可选规划：39M 预训练质量专项提升

> 本节内容为**可选执行**的预训练质量提升方案，独立于 80M 核心路线。目标是在现有 39M 架构上通过架构微调和训练配方优化，在不增加参数量前提下降低 PPL、提升生成质量。验证有效的改进可直接复用到 80M。

### 设计原则

- **消融可控**：每次只改一个变量，保持归因清晰（吸取 V2 教训）
- **39M 优先验证**：所有改进先在 39M 上做消融实验，确认有效后再移植到 80M
- **不改参数量**：不增加 d_model / num_layers / vocab_size，只改架构细节和训练配方

### 改进方向

| # | 改进项 | 改动量 | 预期收益 | 验证方式 |
|---|--------|--------|---------|---------|
| 1 | QK-Norm（Q/K 加 RMSNorm） | ~10 行 | 注意力分布更稳定，PPL ↓ 0.5-1.0 | 39M 消融 |
| 2 | Embedding Scale（`x * sqrt(d_model)`） | 1 行 | 前向信号方差稳定，PPL 小幅下降 | 39M 消融 |
| 3 | `F.scaled_dot_product_attention` | ~20 行 | 省显存、数值更稳、利用 HW 加速 | 39M 消融 |
| 4 | 权重初始化调优（LLaMA 风格） | ~5 行 | 早期训练更稳定 | 39M 消融 |
| 5 | LR / Warmup 网格搜索 | 脚本调整 | 找到最优学习率 | 39M 消融 |
| 6 | 交叉验证数据去重（MinHash） | 中等 | 防训练/验证泄漏，PPL 更真实 | 验证集重算 |
| 7 | 训练 Loss 曲线诊断工具 | 脚本新增 | 快速定位训练异常 | 工具本身 |
| 8 | **Z-Loss 正则化** | ~5 行 | 训练稳定性 ↑，PPL ↓ 0.5-1.0 | 39M 消融 |
| 9 | **Checkpoint EMA** | ~10 行 | 最终模型质量 ↑，零训练成本 | 39M 消融 |
| 10 | **数据 Packing + 跨文档 Mask** | 中等 | GPU 利用率 ↑ 20-40%，等效加速 | 数据集重构建 |
| 11 | **多阶段训练数据配比** | 策略级 | 小模型效果显著提升（SmolLM2 验证） | 80M 直接采用 |
| 12 | **Per-Head Gated Attention** | ~15 行 | 自动抑制噪声头，表达效率 ↑ | 39M 消融 |
| 13 | **WSD 学习率调度** | ~20 行 | 训练可随时延长，灵活度 ↑ | 39M 消融 |

> 8-13 为新增项，参考来源：IMU-1（2026.01）、SmolLM2（2025.02）、Nemotron-Flash（2025.11）。

### 1. QK-Norm

**原理**：在 Q 和 K 投影之后、RoPE 之前添加 RMSNorm，将 Q 和 K 的 L2 范数归一化。防止注意力分数随训练增长而爆炸，使 softmax 分布更平滑。

**实现位置**：`models/xfind_model.py` 的 `GroupedQueryAttention.forward()` 中，在 `apply_rotary_emb` 之前对 Q 和 K 做 RMSNorm，head_dim 维度归一化。

**参考模型**：LLaMA 3 (8B/70B)、MiniMind-V、Gemma 2、Mistral Nemo 均使用 QK-Norm。

**预期代价**：每层增加 2×head_dim 个可学习参数（~128 参数/层，可忽略）

**消融设计**：

```
实验 A：基线（无 QK-Norm）→ 记录 PPL
实验 B：+ QK-Norm                → 同数据同配置训练
控制变量：数据(V3)、训练(8 epoch)、LR(3e-4)、warmup(1%)
```

### 2. Embedding Scale

**原理**：将 token embedding 输出乘以 `sqrt(d_model)`，使 embedding 的方差从 ~1 变为 ~d_model，与后续残差流中的信号方差对齐。Llama 系列的标准做法。

**实现位置**：`xfind_model.py` 的 `forward()` 中，`x = self.token_embed(input_ids)` 后加 `x = x * math.sqrt(self.d_model)`。

### 3. `F.scaled_dot_product_attention`

**原理**：PyTorch 2.0+ 原生的融合注意力函数，在 RTX 4070 Ti 上自动使用 cuDNN 的 Flash Attention / Memory-Efficient Attention 后端，无需安装 `flash-attn` 包。比手写实现更快、更省显存、数值更稳定。

**实现位置**：将 `GroupedQueryAttention.forward()` 中手动计算 `scores → mask → softmax → V` 替换为 `F.scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p, is_causal)`。

**注意**：需要调整 mask 传递方式，`F.scaled_dot_product_attention` 原生支持 `is_causal=True` 参数。

### 4. 权重初始化调优

**原理**：当前初始化对所有 Linear 层使用 `fan_in^(-0.5)` 的正态分布。LLaMA 风格推荐：
- Embedding: `std=0.02` 或 `d_model^(-0.5)`
- 各层 Linear（包括 attn 和 ffn 的 W_q/W_k/W_v/W_o/W_gate/W_up/W_down）: 保持 `fan_in^(-0.5)`，但 W_o 和 W_down 乘以 `1/sqrt(2*num_layers)` 的缩放因子（GPT-2 风格的残差初始化），抑制深层累积的方差增长

### 5. LR / Warmup 网格搜索

在确认 QK-Norm 有效后（稳定性有保障），尝试更高学习率：

| LR | warmup | 预期 |
|----|--------|------|
| 3e-4（基线） | 1% | 当前配置 |
| 4e-4 | 2% | 更高峰值，可能加速收敛 |
| 5e-4 | 2% | 更高风险，需 QK-Norm 兜底 |
| 5e-4 | 5% | 更长 warmup 对冲高 LR 风险 |

每个配置训练 4 epoch，评估 PPL 下降曲线。

### 6. 交叉验证数据去重

**问题**：V3 多源数据在切分训练/验证集时，未做跨文档去重。同一主题的相似文本可能同时出现在训练集和验证集中，导致验证 PPL 偏低（虚假乐观）。

**方案**：MinHash + LSH 跨文档去重，在数据切分前对全文做指纹去重。实现参考 `datasketch` 库或 handcraft MinHash。

### 7. 训练 Loss 曲线诊断工具

在训练脚本中增加自动诊断，检测以下异常并报警：

- **Loss 不下降**：前 10% 步数内 loss 未下降 10% → 建议调整 LR / warmup
- **Loss 突然跳变**：单步 loss 相比前 100 步滑动均值升高 > 20% → 可能是数据异常或 NaN 前兆
- **梯度范数异常**：梯度范数连续 10 步 > clip_threshold × 3 → 建议增大 clip 或降低 LR

**实现位置**：`xfind_train.py` 的 logging 回调中增加诊断逻辑，输出到 TensorBoard + 控制台警告。

### 8. Z-Loss 正则化（IMU-1 / PaLM 验证）

**原理**：对 logits 的 log-sum-exp 加一个极小惩罚项，防止 softmax 分布过于尖锐（即模型对某个 token 过度自信）。尖锐的 softmax 会导致梯度消失和训练不稳定，小模型对此尤其敏感。

**实现位置**：`xfind_train.py` 的 train_epoch 中，在 CE loss 后追加：

```python
# z_loss 系数通常取 1e-4，PaLM/IMU-1 均在此量级
logits = res  # 模型原始输出
z_loss = 1e-4 * torch.logsumexp(logits, dim=-1).pow(2).mean()
loss = ce_loss + z_loss  # 替代原来的 loss = ce_loss
```

**参考来源**：PaLM（2022）首次提出，IMU-1（2026）在小模型上验证有效。Gemma 2 也使用该技术。

**预期收益**：训练更稳定，PPL 通常降低 0.5-1.0。实现仅需 5 行代码，零参数量增长。

### 9. Checkpoint EMA（IMU-1 验证）

**原理**：训练结束后对最后若干个 checkpoint 做指数移动平均（EMA），零训练成本换最终模型质量提升。小模型数据量少、步数少，参数噪声影响相对大，EMA 的平滑效果更显著。

**实现方式**（二选一）：

```python
# 方案 A：训练中维护 EMA（在线）
ema_model = copy.deepcopy(model)
for step, batch in enumerate(train_loader):
    # ... 正常训练 ...
    # 每步更新 EMA
    ema_decay = 0.999  # 或 0.9999
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * p.data

# 方案 B：训练后对 checkpoint 做 EMA（离线，更简单）
# 遍历 checkpoints/ 目录，对最后 N 个 ckpt 做加权平均
```

**参考来源**：IMU-1（2026）消融实验证实 EMA 在小模型上提升显著（尤其 benchmark 得分）。

**预期收益**：零训练成本提升模型质量，benchmark 得分通常提高 1-3%。建议 80M 阶段直接采用。

### 10. 数据 Packing + 跨文档注意力 Mask（SmolLM2 / 行业标配）

**问题**：当前每个序列固定 1024 token，短文档（wiki 约 200-500 token/篇）浪费大量剩余位置，GPU 在短序列上做无用填充。

**原理**：将多个短文档拼接成一个完整 1024 token 序列，但用 attention mask 阻止跨文档注意力：

```
打包后的序列：[doc1: 300tok] [doc2: 500tok] [doc3: 224tok] = 1024 tok

注意力 Mask（简化为分块矩阵）：
       doc1  doc2  doc3
doc1 ┌─────┬─────┬─────┐
doc2 │  ✗  │─────│─────│  ← 每个文档只能注意自己和之前的 token（causal）
doc3 │  ✗  │  ✗  │─────│    但不能跨文档注意（doc3 不能看 doc1/doc2）
     └─────┴─────┴─────┘

关键：需要在打包边界处断开因果链，即 doc2[0] 不能看 doc1 的任何 token
```

**实现位置**：`xfind_dataset.py` 的 `LMDataset.__getitem__()` 中改为打包逻辑；`collate_fn` 中生成对应的 4D attention mask。

**参考来源**：SmolLM2、Llama 3、Qwen 2.5 等所有 SOTA 模型均使用 packing。

**预期收益**：GPU 有效计算利用率提升 20-40%，等效训练加速。对短文档占比高的小数据集（如 wiki）效果尤其显著。

### 11. 多阶段训练数据配比（SmolLM2 核心创新）

**原理**：不在整个训练过程中使用固定数据比例，而是分阶段动态调整。SmolLM2 证实这对小模型至关重要——小模型容量有限，需要在正确的时间接触到合适的数据类型。

**SmolLM2 的三阶段配比参考**：

```
阶段 1（0-40% tokens）：打基础
  web:90%  code:5%  math:5%
  目的：建立语言基础、常识知识

阶段 2（40-70% tokens）：注入专项能力
  web:60%  code:25% math:15%
  目的：强化代码和数学推理

阶段 3（70-100% tokens）：冲刺推理
  web:40%  code:30% math:30%
  目的：最大化 benchmark 得分
```

**XFIND 80M 适配方案**（基于 V3 4 源 + 代码）：

```
阶段 1（0-40%）：打语言基础
  wiki:45%  news:30%  baike:15%  qa:8%  code:2%

阶段 2（40-75%）：注入推理
  wiki:30%  news:20%  baike:15%  qa:10%  code:25%

阶段 3（75-100%）：冲刺质量
  wiki:20%  news:15%  baike:15%  qa:15%  code:35%
```

**实现方式**：在训练脚本中按 global_step 判断当前阶段，切换 DataLoader 或数据集索引。

**参考来源**：SmolLM2（2025.02）论文核心结论——多阶段配比是小模型超越大得多的密集模型的秘诀。

### 12. Per-Head Gated Attention（IMU-1）

**原理**：每个注意力头输出后乘一个可学习的标量 gate（通过 tanh 映射到 [-1, 1]），让模型自动学会抑制噪声头、增强有效头。类似于门控机制的极简版，几乎无参数代价。

**实现位置**：`models/xfind_model.py` 的 `GroupedQueryAttention.__init__()` 中加 `self.gate = nn.Parameter(torch.ones(num_heads, 1, 1))`，`forward()` 中 `attn_out = attn_out * self.gate.tanh()`。

```python
# __init__
self.gate = nn.Parameter(torch.ones(num_heads, 1, 1))

# forward, 在 W_o 投影之前
attn_output = attn_output * self.gate.tanh()
attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
return self.W_o(attn_output)
```

**代价**：num_heads 个参数（8 个 / 层，可忽略）。**收益**：自动控制注意力头的重要性分配，小模型因容量有限受益更明显。

**参考来源**：IMU-1（2026.01）消融实验证实有效，Shazeer（2020）Talking-Heads Attention 的前身。

### 13. WSD（Warmup-Stable-Decay）学习率调度（IMU-1 / DeepSeek 验证）

**原理**：替代 Cosine Annealing 的三段式调度：

```
Warmup  (1-2%步):  0 → peak_lr      线性增长
Stable  (80-90%步):  peak_lr         保持恒定
Decay   (剩余步):    peak_lr → 0    快速衰减（Cosine 或 Linear）
```

**与 Cosine 的核心差异**：

| 方面 | Cosine | WSD |
|------|--------|-----|
| 训练中延长 | 困难（需重新计算 schedule） | 容易（Stable 段不改，延长 Decay） |
| 中间阶段 LR | 持续下降 | 保持恒定 |
| Decay 时机 | 从头到尾平滑衰减 | 可灵活决定何时开始衰减 |
| 额外探索 | 无 | Decay 时机本身是一个可调参数 |

**实现位置**：`xfind_train.py` 的 `get_lr_cosine()` 替换为 `get_lr_wsd()`：

```python
def get_lr_wsd(step, total_steps, warmup_ratio=0.02, stable_ratio=0.85, min_lr_ratio=0.0):
    """WSD 调度：Warmup → Stable → Decay"""
    warmup_steps = int(total_steps * warmup_ratio)
    stable_steps = int(total_steps * stable_ratio)
    decay_steps = total_steps - stable_steps

    if step < warmup_steps:
        return step / max(1, warmup_steps)  # 线性 warmup
    elif step < stable_steps:
        return 1.0  # Stable 段恒定
    else:
        progress = (step - stable_steps) / max(1, decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
```

**WSD 的另一大优势**：训练中途发现 PPL 还在降，想多训几轮？Cosine 必须重新从头算 schedule，WSD 直接拉长 stable_ratio 即可。

**参考来源**：IMU-1（2026）、DeepSeek-V2/V3、MiniCPM 均使用 WSD 或其变体。

### 执行优先级

```
第一优先（加一行代码即可，不需重训）
├── 8. Z-Loss            ← 5行，不改架构，训前加一句
└── 9. Checkpoint EMA    ← 10行，训后执行，免费提升

第二优先（改架构 + 39M 消融，~3 天）
├── 1. QK-Norm
├── 2. Embedding Scale
├── 3. F.scaled_dot_product_attention
└── 12. Per-Head Gated Attention

第三优先（训练配方 + 消融调优，~2 天）
├── 4. 权重初始化调优
├── 5. LR / Warmup 网格搜索
└── 13. WSD 学习率调度

第四优先（数据侧，~3 天）
├── 6. 交叉验证数据去重
├── 10. 数据 Packing + 跨文档 Mask
└── 7. Loss 曲线诊断工具

第五优先（80M 专项，SFT 完成后启动）
├── 11. 多阶段训练数据配比  ← 80M 直接采用
└── 将验证有效的改进移植到 80M 配置
```

### 与 80M 核心路线的关系

```
80M 核心路线（P0-P2）         可选规划（质量专项）
───────────────               ──────────────
仓库拆分（gleam-core）         QK-Norm 消融
80M 默认配置训练               Embedding Scale
80M 深度对比实验               F.scaled_dot_product_attention
词表扩展实验                   LR/Warmup 调优
                              ───── 融合 ────→
                              验证有效的改进移植到 80M 架构
                              80M 以"更好的架构 + 更好的配方"重新训练
```

两者可**并行推进**：仓库拆分和 80M 参数配置规划不依赖架构改进，可先行实施。架构改进在 39M 上验证后，80M 直接用改进后的代码和配方重新训练，获得叠加收益。

---

## 十四、端侧部署方向规划（Game NPC / On-Device SLM）

> GleamLM-Nano 40M 的极致轻量特性天然适配端侧设备部署。本节规划"通用 SFT → RAG 引擎 → 端侧量化 → 游戏 Demo"的完整落地路径。

### 端侧定位

| 维度 | 云端 LLM | GleamLM 端侧 |
|------|---------|-------------|
| 延迟 | 200-500ms | **<10ms** |
| 离线 | 不可用 | **完全离线** |
| 隐私 | 对话上传服务器 | **全在本地** |
| 成本 | 按 token 计费 | **零边际成本** |
| 模型大小 (INT4) | 数 GB | **~20MB** |
| 并发 | 全球排队 | **独占设备** |

> 当前行业热点：Apple Intelligence、高通 AI Engine、联发科 NeuroPilot 全在押注 On-Device SLM。

### 游戏 NPC 场景需求

| 场景 | 现有方案 | 小模型优势 |
|------|---------|-----------|
| NPC 对话 | 固定脚本 3 句话来回说 | 动态回应玩家行为，不破绽 |
| 任务提示 | 硬编码文字 | 根据玩家进度生成自然语言引导 |
| 关卡叙事 | 预写文本 | 根据玩家选择实时生成分支叙事 |

### 技术路径

```
预训练 best_model.pt
  ↓
通用 SFT（几千条 QA，短回复 + 口语化 + 角色扮演）
  ↓
RAG 引擎（检索增强生成 — 游戏剧情文本向量化 → 按场景匹配）
  ↓
INT4/INT8 量化导出（~20MB）
  ↓
ONNX 导出 → Unity/Unreal 引擎集成
  ↓
游戏 NPC Demo
```

### 通用 SFT 数据特点（端侧 NPC 专用）

与通用 ChatBot SFT 的差异：

| 维度 | 通用 ChatBot SFT | 端侧 NPC SFT |
|------|-----------------|-------------|
| 回复长度 | 200-500 token | **20-80 token**（移动端不等长回复） |
| 语气 | 正式、客观 | **口语化**（"哎，你终于来了！"） |
| 内容来源 | 模型参数记忆 | **RAG 上下文**（基于场景信息回） |
| 对话感 | 弱 | **强**（追问、反问、语气词、省略句） |
| 角色一致性 | 不要求 | **高**（同一人格不跳戏） |

**SFT 数据类型配比**：

| 类型 | 占比 | 示例 |
|------|------|------|
| 通用短对话 | 40% | 打招呼、告别、简单问答 |
| 角色扮演 | 30% | 给定身份和性格 → 按角色回应 |
| 上下文理解 | 20% | 给一段场景描述 → 基于场景回应玩家 |
| 安全边界 | 10% | 拒绝不当请求 → "这可不关我的事" |

**数据生成 system prompt 示例**（用 GPT-4/Claude/DeepSeek API 批量生成）：

```
你是一个游戏 NPC 角色。你的回复必须：
- 简短，2-4句话以内
- 口语化，像真人对话
- 始终在角色设定的框架内回应
- 不要解释、不要说教、不要写成文章

当前角色：[角色名]，性格：[性格]，所在场景：[场景]
玩家说：[玩家输入]
你的回复：
```

### RAG 引擎设计

**优势**：模型不需要记住游戏剧情，只需基于检索到的上下文生成。换一个游戏只需替换 rag_doc.txt，无需重新训练。

```
游戏剧情文本 → 分块（200-500 字/块）→ 向量化 → 向量数据库
                                                    ↓
玩家输入 → 向量检索（Top-3 相关块）→ 拼接 prompt → 模型生成 → NPC 回复
```

**Prompt 模板**：
```
你是《[游戏名]》中的 [角色名]，性格[性格描述]。
以下是当前场景的相关信息：
---
[检索到的剧情文本块]
---
玩家：[输入]
你的回复：
```

### 端侧部署路线图

| 阶段 | 产出 | 难度 |
|------|------|------|
| 1. 通用 SFT（短对话 + 角色扮演） | 学会短回复、口语化 | 低 |
| 2. RAG 引擎 | 基于上下文的动态 NPC 对话 | 低 |
| 3. INT4 量化 + 推理优化 | 模型 < 20MB | 低（已有 gleamlm_quantize.py） |
| 4. ONNX 导出 | 脱离 PyTorch，C++/Unity 可加载 | 中 |
| 5. Unity/Unreal Demo | 可玩的游戏 NPC 演示 | 中 |
| 6. Android/iOS Demo | 移动端跑通 | 中 |

### 预期效果

- 模型大小：INT4 量化 ~20MB（小于一张手游贴图）
- 推理速度：移动端 > 100 token/s
- 对话质量：约 3-5 轮连贯 NPC 角色对话
- 可控性：RAG 约束下准确回应游戏世界观，不漂移

### 开源价值

通用 SFT + RAG 引擎的组合证明 GleamLM 是一个**游戏 NPC 引擎**而非"某个游戏的模型"。换游戏只需换 rag_doc.txt，模型本身和 SFT 权重可复用。这比做一次性的游戏专项微调更有开源传播力。

### 多方向 SFT 策略（一个基座，多个变体）

预训练基座只训一次，不同方向靠 SFT 数据切换——几千条 QA 一个小时即可微调出一个变体：

```
预训练 best_model.pt（基座，不动）
    ├── SFT → 教育版（知识问答、学科辅导）
    │       └── DPO → 教育版对齐
    │
    ├── SFT → 游戏版（NPC 对话、角色扮演）
    │       └── DPO → 游戏版对齐
    │
    └── SFT → 通用助手（日常闲聊）
            └── DPO → 通用对齐
```

| 维度 | 说明 |
|------|------|
| 基座训练 | 仅一次，~3 天 |
| 每个变体 SFT | ~1 小时 |
| 每个变体大小 (FP16) | ~80MB |
| 每个变体大小 (INT4) | ~20MB |

### 游戏公司合作切入点

| 目标 | 切入角度 |
|------|---------|
| 中型手游公司 | "NPC 固定脚本 → 动态对话，20MB，不要服务器" |
| 独立游戏工作室 | 开源免费 → 商业定制需求 |
| 引擎公司（Unity中国/Unreal） | ONNX 小模型插件生态 |
| 教育游戏公司 | "语文课文 NPC 能跟学生讨论文章" |

**差异化卖点**：

| 游戏公司的痛点 | GleamLM 解法 |
|------|------|
| 云端 API 有延迟+费用 | 本地推理，零成本零延迟 |
| 玩家数据不能上传 | 完全离线，隐私无忧 |
| 手游包体不能大 | INT4 量化 20MB，小于一张贴图 |
| 换个游戏要重新集成 | RAG 换文本就行，模型不动 |
| 技术团队不一定懂 LLM | 纯 PyTorch + ONNX，Unity 直读 |

> 核心话术：**不是在卖模型，是在卖"让任何游戏都能拥有会说话的 NPC，不需要云端、不需要大模型密钥"这个能力。**

---

## 附录：蒸馏 SFT 方案

### 背景

预训练后的模型只会"续写"，不会"回答问题"。SFT（指令微调）用"问题→回答"格式的数据再训一轮，让模型从续写器变为对话助手。本方案直接通过 DeepSeek-V4-Pro API 生成 10000 条高质量中文指令数据，免去本地部署大模型的显存开销。

### 教师模型选择

| 教师 | 优势 | 方式 |
|------|------|------|
| **DeepSeek-V4-Pro** | 中文质量极高，API 调用零本地显存 | Trae IDE 内置代理直接调用 |

**推荐**：DeepSeek-V4-Pro，通过 Trae IDE 的 AI 代理接口直接生成，质量高、速度快。

### 数据设计（10000 条）

三类数据混合，覆盖不同能力维度：

**A 类：通用问答（40%）**
```
问：什么是光合作用？
答：光合作用是植物利用光能，将二氧化碳和水转化为有机物并释放氧气的过程。
```

**B 类：知识回答（30%）**
```
问：清朝有几个皇帝？分别是谁？
答：清朝共 12 位皇帝：努尔哈赤、皇太极、顺治、康熙、雍正、乾隆、嘉庆、道光、咸丰、同治、光绪、宣统。
```

**C 类：创作与闲聊（30%）**
```
问：用一句话形容秋天的落叶。
答：秋风起，落叶如金蝶般在空中旋舞，铺满归家的路。
```

**数据格式**（JSONL）：
```json
{"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，让计算机从数据中学习规律，而无需显式编程。"}
```

### 生成流程

```
Step 1: 编写种子问题（~200 个手动编写的多样问题）
Step 2: 通过 Trae IDE 代理调用 DeepSeek-V4-Pro 批量生成答案（temperature=0.7）
Step 3: 人工抽查前 200 条，剔除低质量（答非所问、敷衍、过短）
Step 4: 从合格样本中抽取格式模板，让 DeepSeek-V4-Pro 继续生成更多变体
Step 5: 总计生成 10000 条
Step 6: 加入 ChatML 格式包装（V4 BBPE 原生支持）
```

### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据量 | 10000 条 | ~55 分钟可训完 |
| max_seq_len | 512 | 对话足够 |
| batch_size | 8 | RTX 4070 Ti 可承载 |
| epochs | **2** | SFT 通常 1-2 epoch 足够（MiniMind 参考） |
| learning_rate | **5e-7** | 预训练 LR 的 1/600（MiniMind 参考：SFT LR = 预训练 LR / 1000） |
| warmup | 50 steps | 快速预热 |
| optimizer | AdamW | 与预训练一致 |
| chat template | ChatML 格式 | — |

> **LR 校准理由**：小模型 SFT 极易灾难性遗忘。MiniMind 的 SFT 用 5e-7（预训练 5e-4 的 1/1000）。我们的预训练 LR=3e-4，因此 SFT 用 5e-7（1/600），收敛更稳。

### ChatML 格式 + Loss Mask 机制

**ChatML 格式**（token 化后见下方 mask）：
```
<|im_start|>system
你是一个有帮助的AI助手。
<|im_end|>
<|im_start|>user
{{instruction}}
<|im_end|>
<|im_start|>assistant
{{output}}
<|im_end|>
```

**Loss Mask 机制**（SFT 与预训练的核心差异，参考 MiniMind）：

SFT 只对 **assistant 回复部分** 计算损失，user 和 system 部分设为 `ignore_index=-100`：

```
原始 token 序列：
   <|im_start|> system ... <|im_end|> <|im_start|> user ... <|im_end|> <|im_start|> assistant ... <|im_end|>
损失标签（labels）：
   -100  -100  -100  ...  -100   -100   -100  ...  -100   实际token_id  ...  实际token_id  -100
   └── system 忽略 ──┘       └── user 忽略 ──┘         └── assistant 计算 loss ──┘
```

```python
# 核心实现：通过 labels 逐 token 控制 loss
def build_sft_labels(input_ids, bos_id, eos_id):
    """仅对 assistant 部分（bos 到 eos 之间）计算 loss"""
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        # 检测每段对话的起止
        if input_ids[i] == bos_id:
            start = i   # bos 位置
            end = start + 1
            while end < len(input_ids) and input_ids[end] != eos_id:
                end += 1
            # 偶数段（0-based: system/user）ignore；奇数段（assistant）计算 loss
            if start > 0:  # 跳过 system 段
                pass  # 由上游按奇偶性标记，assistant 段设 labels[j] = real_token[j]
            i = end + 1 if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

> **原则**：`CrossEntropyLoss(ignore_index=-100)` 自动跳过 label=-100 的 token，不参与梯度更新。该机制也复用于 DPO 的 chosen/rejected 分支。

### 系统消息随机注入（参考 MiniMind）

训练时以 **20% 概率**从预设池中随机选取一条系统消息注入，让模型适应"有/无 system prompt"两种场景：

```python
SYSTEM_PROMPTS = [
    "你是一个有帮助的AI助手。",
    "你是一个友善的中文对话助手，请用简洁清晰的语言回答问题。",
    "你是一个知识渊博的助手，请准确回答问题。",
    "You are a helpful AI assistant.",
]

import random
system_msg = random.choice(SYSTEM_PROMPTS) if random.random() < 0.2 else ""
```

> 成本：一行 `random.random()`，几乎零开销。

### 评估

SFT 后对比前后生成效果：

```
预训练模型输入"你好" → "!是不是这么好看?!!...公众号模板..."
SFT 后输入"你好"     → "你好！有什么可以帮助你的吗？"
```

定性评估 50 个手写问题，统计回答相关率、信息准确率、不重复率。

### 后续：DPO 偏好对齐（可选）

SFT 教会模型"怎么回答"，DPO 教会模型"哪种回答更好"。

**数据格式**（参考 MiniMind，chosen/rejected 对结构一致）：
```json
{"instruction": "...", "chosen": "好的回答", "rejected": "差的回答"}
```

**DPO 训练细节**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据量 | 500-1000 条 | 偏好对 |
| beta | **0.1** | 控制偏离参考模型的程度（MiniMind 标准值） |
| 参考模型 | **冻结的 SFT 模型** | 不参与训练，仅输出 reference log-prob |
| batch 组织 | **半 chosen + 半 rejected** | 交替排列，loss 统一计算 |
| loss mask | **复用 SFT 逻辑** | chosen 和 rejected 各自计算 assistant 部分 loss |
| LR | 5e-7 | 与 SFT 相同 |

**DPO Loss 公式**（MiniMind 实现）：
```
log_ratio = policy_logp(chosen) - policy_logp(rejected)
ref_ratio = ref_logp(chosen) - ref_logp(rejected)
loss = -log_sigmoid(beta * (log_ratio - ref_ratio))
```

仅需 500-1000 条对比数据，可让模型更安全、更礼貌。

### 时间估算

| 阶段 | 耗时 |
|------|------|
| 编写种子问题 + DeepSeek-V4-Pro 生成 10000 条 | 1-2 小时（API 批量调用） |
| 人工抽查清洗 | 1 小时 |
| SFT 训练（2 epochs） | ~55 分钟 |
| 评估对比 | 30 分钟 |
| **总计** | **约 4-5 小时** |

---

*文档：GleamLM 语言模型开发计划 v5.2*
*修订日期：2026-07-05*
*作者：philexohf*
