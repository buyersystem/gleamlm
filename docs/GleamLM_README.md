# GleamLM —— 面向教育和研究的小型语言模型

> **纯 PyTorch 从零实现，零 HuggingFace 依赖，12G 显存跑通预训练 → SFT → DPO 全链路。**

作者另一个项目：[Transformer 中英机器翻译](https://github.com/philexohf/Transformer_zh_en2026)（50+ Stars）

---

## 🚀 30 秒体验（无需训练）

下载预训练模型，直接对话：

```bash
wget https://modelscope.cn/models/philexohf/GleamLM-Nano/files/gleamlm_nano.pt
python gleamlm_infer.py --model gleamlm_nano.pt --prompt "你好"
```

输出示例：
> 你好！有什么我可以帮你的吗？

SFT 对话模型：
```bash
python gleamlm_infer.py --model gleamlm_nano.pt --sft --prompt "你好，请介绍一下你自己。"
```

---

## 项目简介

搞大模型的人应该都有过这种体验——想认真学一下底层，结果发现市面上的开源项目一个比一个重。HuggingFace 一封装，`from_pretrained` 一行代码就加载完了，但你让我说 Attention 里的 Q、K、V 到底怎么算的，RoPE 的频率基怎么预计算的，BBPE 分词训练到底在干什么——答不上来，全是黑盒。

更头疼的是硬件。动不动就要 8 张 A100、24G 起步，手上就一张 4070Ti，12G 显存，想完整跑一遍预训练到对话的全流程？门都没有。

所以干脆自己写了一个——**GleamLM**，全程纯 PyTorch 手写，零 HuggingFace 依赖，不限平台，12G 显存就能从头跑到尾。

**为什么这么做？不是为了炫技。你不亲手写一遍，永远不会知道那些论文里的公式到底怎么变成代码的。**

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **纯手写 Decoder-only** | 12 层 Transformer，RMSNorm / RoPE / GQA / SwiGLU / QK-Norm，全自己写 |
| **自研 BBPE 分词器** | 607 行纯 Python，零依赖，12K 词表，原生支持 ChatML 特殊 token |
| **完整数据管线** | 中文四源混合（维基/百科/新闻/问答），自动清洗、去重、字符加权配比 |
| **全链路训练** | 预训练 → SFT（ChatML + loss mask）→ DPO 偏好对齐，断点续训 |
| **高效推理** | KV Cache + 流式生成 + Temperature / TopK / TopP 采样 |
| **双平台兼容** | Windows / Linux 均可，单卡 12GB 显存 |

### 模型规格（GleamLM-Nano ~40M）

| 参数 | 值 |
|------|-----|
| 参数量 | **~40M**（Embed 6.1M + Transformer 34.6M） |
| 上下文窗口 | 1024（RoPE 支持外推至 2048/4096） |
| 词表大小 | 12,001（自研 BBPE） |
| 网络层数 | 12 |
| 模型维度 | 512 |
| 注意力 | GQA（8 Q-heads / 4 KV-heads）+ QK-Norm |
| 前馈网络 | SwiGLU（中间维度 1365） |
| 训练精度 | BF16/FP16 AMP |
| 分布式 | DDP（`torchrun` 一行启动） |

---

## 快速开始

### 环境

- Python 3.10+
- PyTorch 2.5+ with CUDA 12.4
- RTX 4070 Ti 12GB（或同等显存）

```bash
pip install -r requirements.txt
```

### 1. 数据准备（一键管线）

```bash
# 下载原始数据（仅首次）
pip install py7zr kagglehub
python tools/download_data.py

# 一键：清洗 → 去重 → QA过滤 → 字符加权配比 → 混合切分
python tools/prepare_data.py --input data/raw --output data/splits
```

### 2. 预训练

```bash
python gleamlm_train.py --data_dir ./data/splits --epochs 5

# 断点续训
python gleamlm_train.py --data_dir ./data/splits --load_checkpoint checkpoints/checkpoint_epoch_3.pt

# 监控
tensorboard --logdir ./checkpoints/runs
```

| 关键参数 | 默认值 | 说明 |
|----------|--------|------|
| `--epochs` | 5 | 训练轮数 |
| `--batch_size` | 4 | Micro-batch（显存安全） |
| `--accumulate_grad` | 16 | 梯度累积（有效 batch=64） |
| `--lr` | 3e-4 | 峰值学习率 |
| `--label_smoothing` | 0.1 | 标签平滑 |

优化器：AdamW（β=0.9,0.95，wd=0.01），BF16 AMP，Cosine Warmup + Decay。

单卡 RTX 4070 Ti 12GB，每 epoch ~15 小时，5 epoch 约 75 小时。

预训练基座模型已在魔搭上线：[GleamLM-Nano · 模型库](https://www.modelscope.cn/models/philexohf/GleamLM-Nano)

### 3. 推理

```bash
# 单次生成
python gleamlm_infer.py --model checkpoints/best_model.pt --prompt "人工智能是"

# 交互模式
python gleamlm_infer.py --model checkpoints/best_model.pt

# 调整采样
python gleamlm_infer.py --model checkpoints/best_model.pt --temperature 0.8 --top_k 50 --top_p 0.9

# SFT 模型推理（ChatML 格式 + 自动截断）
python gleamlm_infer.py --model checkpoints/sft/sft_best.pt --sft --prompt "你好，请介绍一下你自己。"

# DPO 模型推理
python gleamlm_infer.py --model checkpoints/dpo/dpo_best.pt --sft --prompt "什么是机器学习？"
```

### 4. SFT 指令微调

```bash
python gleamlm_sft.py --data_path ./data/sft_data.jsonl --model_path ./checkpoints/best_model.pt
```

### 5. DPO 偏好对齐

```bash
python gleamlm_dpo.py --data_path ./data/dpo_data.jsonl --model_path ./checkpoints/sft/sft_best.pt
```

### 6. 量化导出

FP32 → FP16，体积减半（178.9 MB → 89.5 MB），推理精度基本无损。

```bash
python gleamlm_quantize.py --input checkpoints/best_model.pt --output checkpoints/model_fp16.pt
```

### 7. 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 训练结果

### 预训练（GleamLM-Nano，~40M）

训练配置：`batch_size=4, accumulate_grad=16`（等效 64），`label_smoothing=0.1`，`stride=768`，Cosine Warmup + Decay，12GB 显存持续 ~92% 满载。

| Epoch | Train Loss | Val Loss | PPL | 备注 |
|-------|-----------|----------|-----|------|
| 0 | 3.2960 | 2.8064 | 16.55 | 语法收敛，生成通顺但内容空洞 |
| 1 | 2.8764 | 2.7045 | 14.95 | 首句沾边，后续漂移 |
| 2 | 2.8053 | 2.6568 | 14.25 | 高频事实固化中 |
| 3 | 2.7655 | 2.6255 | 13.81 | 边际收益递减，改善持续 |
| 4 | 2.7440 | 2.6136 | **13.65** | 训练完成，全程无过拟合 |

**最佳结果**：`val_loss=2.6136`，`val_ppl=13.65`，模型保存至 `./checkpoints`。

### 预训练生成样例

| 输入 | 输出（节选） |
|------|------|
| `中国有五千年的` | 历史，是中华人民共和国的一部分。（首词正确预测"历史"） |
| `机器学习是人工智能的` | 一个重要方面。（精准命中常见搭配） |
| `读书的好处是` | 每个人都会有自己的兴趣爱好和想法... |
| `世界上最高的山峰是` | 位于中国西藏自治区拉萨市南部的一座山峰...（地理关联正确） |

> 模型对高频搭配和常见知识有一定记忆，但 40M 参数在长尾知识上仍会发散。后续通过 SFT + DPO 对齐改善。

### SFT + DPO 对齐

SFT 数据：DeepSeek-V4-Pro API 蒸馏生成 10000 条高质量中文指令（通用问答 40% / 知识回答 30% / 创作闲聊 30%），ChatML 格式，loss mask 只训回答部分。

| 阶段 | 训练数据 | 学习率 | 耗时 | 效果 |
|------|---------|--------|------|------|
| SFT | 10000 条 | 5e-6 | ~55 分钟 | 从"续写"转为"直接回答" |
| DPO | 150 对 chosen/rejected | 1e-7 | ~2 分钟 | 纠正方向性错误，减少幻觉 |

**DPO 前后对比**（`--sft --temperature 0.7`）：

| Prompt | SFT 后 | DPO 后 | 改善 |
|--------|--------|--------|:---:|
| 请用一句话描述北京的秋天 | 北京是世界上最大的热带气旋生物多样性保护区 | 落叶遍野、金黄如雪、红得让人心旷神怡 | ✅ |
| 什么是光合作用 | 天然的氧化物，分子量约2000万个太阳质量 | 生物体生长发育和光照时间变化 | ✅ |
| 你好，请介绍一下你自己 | 如果你是个人，建议先学会分析别人的优劣 | 练字孩子的成长故事 | 叙事更连贯 |
| 什么是机器学习 | 将信息传递给机器人 | 操作系统/计算机模块分离 | 方向修正 |

> DPO 最显著的效果是纠正方向性错误。但 40M 参数容量不足以支撑精准事实记忆，这是小模型的物理上限。下一阶段转向 GleamLM-Lite（80M）。

---

## 数据集

| 数据源 | 原始 | 清洗后 | 保留率 | 来源 |
|--------|:---:|:---:|:---:|------|
| 中文维基 | 565万 | 545万 | 96.4% | [modelscope](https://www.modelscope.cn/datasets/caoaolong/zhwiki) |
| 百度百科 | 214万 | 213万 | 99.8% | 自行搜索 |
| 新闻 2016 | 202万 | 171万 | 84.5% | 自行搜索 |
| 社区问答 | 403万 | 92万 | 22.8% | [Kaggle](https://www.kaggle.com/datasets/terrychanorg/webtext2019zhjsonwebtext2019zh) |
| **合计** | **1,384万** | **1,021万** | **73.8%** | — |

最终切分：train 6.48 GB（90%）/ valid 0.36 GB（5%）/ test 0.36 GB（5%），合计 7.20 GB，~1.2B 训练字符。

各源按字符占比自动配比（行均字符差异大，新闻 ~752 字/行 vs 维基 ~123 字/行）。

---

## 项目结构

```
GleamLM/
├── gleamlm_train.py           # 预训练（AMP + DDP + Cosine + 断点续训）
├── gleamlm_infer.py           # 推理（KV Cache + 交互式对话）
├── gleamlm_sft.py             # SFT 指令微调（ChatML + loss mask）
├── gleamlm_dpo.py             # DPO 偏好对齐
├── gleamlm_quantize.py        # FP16 量化导出
├── models/
│   ├── gleamlm_model.py       # 模型定义（RMSNorm / RoPE / GQA / SwiGLU）
│   └── gleamlm_config.py      # 全局配置
├── tokenizer/
│   └── bbpe_tokenizer.py      # V4 BBPE 分词器（607行，纯 Python 零依赖）
├── inference/
│   ├── sampler.py             # Temperature / TopK / TopP 采样
│   └── streamer.py            # KV Cache 流式生成
├── tools/
│   ├── prepare_data.py        # 一键数据管线
│   ├── download_data.py       # 多源数据下载
│   └── ...
├── scripts/
│   ├── generate_sft_data.py   # DeepSeek API 蒸馏 SFT 数据
│   └── ...
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_dataset.py
├── data/
│   ├── raw/                   # 原始语料
│   └── splits/                # train/valid/test + .npy 预分词缓存
├── checkpoints/               # 模型检查点 + TensorBoard 日志
└── requirements.txt
```

---

## 版本路线

| 版本 | 参数量 | 定位 | 状态 |
|------|--------|------|------|
| GleamLM-Nano | ~40M | 教学级 / 单卡资源 | ✅ 已完成 |
| GleamLM-Lite | ~80M | 教学级 / 服务器资源 | 🚧 开发中 |
| GleamLM-Pro | ~126M | 科研进阶 / 服务器资源 | 📋 规划中 |
| GleamLM-0.6B | ~0.6B | 工业级验证 / 算力集群 | 🤝 寻求合作 |

---

## 写在最后

这个项目最大的价值不是模型本身，而是**亲手把 LLM 的每一个模块都写了一遍**。

从 RoPE 的旋转矩阵到 GQA 的 KV 缓存，从 BBPE 的 merge 算法到 DPO 的偏好损失——**当你不再依赖黑盒，你才真正理解了大模型。**

如果你也想从零手搓一个 LLM，或者想找一个轻量、无依赖、能完整跑通全链路的学习项目，欢迎 Star ⭐ 关注后续迭代。

有问题直接提 Issue，我会逐条回复。

---

## 许可证

Apache License 2.0
