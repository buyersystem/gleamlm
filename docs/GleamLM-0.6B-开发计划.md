# GleamLM-0.6B 开发计划

## 定位

**中英双语 600M 参数 Decoder-only 语言模型**，架构对齐 Qwen3-0.6B，纯中文维基+英文维基从零训练。Dense 架构，不做 MoE，不做 SFT/RLHF（对齐留到后续版本）。

---

## 架构规格

| 配置项 | XFIND-Nano（当前） | XFIND-Pro | 说明 |
|---|---|---|---|
| `vocab_size` | 32,000 | **64,000** | 中英双语最佳区间 |
| `d_model` | 512 | **1,024** | 2× |
| `num_layers` | 8 | **37** | 28层的 0.6B 词表太大浪费参数，37层只做大中文 |
| `num_heads` | 8 | **16** | Q 头数 |
| `num_kv_heads` | 4 | **8** | GQA 2:1 |
| `d_ff` | 1,365 | **3,072** | SwiGLU 中间维度 |
| `max_seq_len` | 1,024 | **4,096** | 基础窗口，YaRN 外推到 32K |
| `dropout` | 0.1 | **0.1** | |
| RoPE θ | 10,000 | **1,000,000** | Qwen3 同款 |
| QK-Norm | 无 | **有** | Attention 内 Q/K 各加 RMSNorm |
| YaRN | 无 | **有** | 上下文扩展 |
| Weight Tying | 是 | **否** | 600M 级别独立 lm_head 效果更好 |
| 总参数 | 39M | **~597M** | |

### 参数量精算

```
Embedding:             64,000 × 1,024  =   65,536,000
LM Head (独立):        64,000 × 1,024  =   65,536,000

每层 Attention:
  W_q:                 1,024 × 1,024    =    1,048,576
  W_k:                 1,024 × 512      =      524,288
  W_v:                 1,024 × 512      =      524,288
  W_o:                 1,024 × 1,024    =    1,048,576
  Q-Norm:              1,024            =        1,024
  K-Norm:              1,024            =        1,024
  小计:                                  3,149,824

每层 SwiGLU FFN:
  W_gate:              1,024 × 3,072    =    3,145,728
  W_up:                1,024 × 3,072    =    3,145,728
  W_down:              3,072 × 1,024    =    3,145,728
  小计:                                  9,437,184

每层 Norms:
  attn_norm (RMSNorm): 1,024            =        1,024
  ffn_norm (RMSNorm):  1,024            =        1,024
  小计:                                      2,048

每层合计:                               12,593,152
37 层:                                 465,946,624

Final Norm (RMSNorm):  1,024            =        1,024

─────────────────────────────────────────────
总计:                                  597,019,648  (~597M)
```

---

## 硬件方案

| 方案 | 显卡 | 可行性 | 预估显存/卡 | 预估训练时长 |
|---|---|---|---|---|
| **A（推荐）** | 3×L20 48GB DDP | ✅ 可行 | ~20GB（含 gradient checkpointing） | ~5-8 天 |
| B | 8×L20 48GB DDP | ✅ 宽松 | ~12GB | ~3-5 天 |
| C | 1×RTX 4070 Ti 12GB | ❌ 不可行 | — | — |

> 4070 Ti 12GB 不可能的原因：600M 模型 fp16 参数 ≈ 1.2GB，AdamW 优化器状态（momentum + variance 各 fp32）≈ 4.8GB，仅这两项就 6GB。seq_len=4096 的注意力激活矩阵 ≈ 2GB/层，37 层即使 gradient checkpointing 也只保留 1 层激活，加上 embedding 输出序列 4K×1K×fp16 = 8MB/样本，batch_size=1 时激活值 ≈ 2.5GB。总计 ~8.5GB 已接近 12GB 上限，还需要梯度空间、中间缓冲区，没有任何余地。

---

## 与 Qwen3-0.6B 的差异

| 维度 | Qwen3-0.6B | XFIND-Pro | 原因 |
|---|---|---|---|
| 词表 | 151,936 | 64,000 | 去掉 100+ 种语言的 subword，专注中英 |
| 层数 | 28 | 37 | 省下的 90M embedding 参数加到深度上 |
| 训练数据 | 36T tokens（多语言） | ~3B tokens（中英维基） | 算力有限 |
| 训练设备 | 数千 GPU | 3×L20 | 家用级 |

**核心思路**：不比广度（多语言），比深度（同参数量下更多 transformer 层）。在中文任务上，37 层的纯中文 600M 大概率优于 28 层的多语言 600M。

---

## Linux 训练优化方案（对比 Windows 训练的重大提升）

XFIND-Pro 在 Linux 上训练，可以利用 Windows 不具备的多个高性能组件。以下按收益从大到小排列。

### Flash Attention 2/3（最大单项收益）

**Windows 不支持，Linux 独占。** 手写 attention 的显存是 O(seq²)，4096 长度的注意力矩阵 ≈ 2GB/层。37 层意味着训练时要保留 74GB 的注意力激活（即使只保留一层也很大）。Flash Attention 通过 tiling 和 recomputation 把显存降到 O(seq)，同时用 fused kernel 加速 3-5×。

| 维度 | 手写 Attention | Flash Attention 2 |
|---|---|---|
| 显存（注意力部分） | O(seq²)，4096² ≈ 2GB/层 | O(seq)，几乎不存注意力矩阵 |
| 训练速度 | 基准 | 3-5× 更快 |
| micro_batch 提升 | 1 | 可到 2-4 |

```python
# pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)
```

**代码改动**：在 `GroupedQueryAttention.forward()` 中，训练模式下用 `flash_attn_func` 替换手写的 softmax(QK^T)V，推理模式保留手写（兼容 KV Cache）。

---

### torch.compile（JIT 编译加速）

PyTorch 2.0+ 的 `torch.compile` 在 Linux 上通过 Triton 后端做 kernel fusion，效果远超 Windows。

| 效果 | 说明 |
|---|---|
| 训练速度 | 1.3-1.8× 加速 |
| 显存 | 减少 20-30%（多个小 kernel 融合，消掉中间缓冲区） |
| 代价 | 首次编译 2-3 分钟，后续无开销 |

```python
# xfind_train.py 中，模型构建后加一行
model = torch.compile(model, mode="reduce-overhead")
```

---

### Fused AdamW（融合优化器）

PyTorch 2.0+ 的 `adamw(..., fused=True)` 把 AdamW 的 step 操作合并成一个 CUDA kernel。

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95),
    fused=True  # Linux CUDA 12+ 支持
)
```

收益：**~10% 训练加速 + ~1GB 显存节省**。

---

### DeepSpeed ZeRO-2（优化器状态分片）

DDP 模式下每张卡存完整优化器状态（momentum + variance 各 fp32 ≈ 4.8GB）。ZeRO-2 把优化器状态分片到所有 GPU 上，每张卡只存 1/N。

| 方案 | 优化器显存/卡 | 总显存/卡 |
|---|---|---|
| DDP | 4.8GB | ~8GB |
| **DeepSpeed ZeRO-2** | **1.6GB** | **~5GB** |

省下的显存可以直接提升 micro_batch 大小。

```json
// ds_config.json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "bf16": { "enabled": true },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

```bash
deepspeed --num_gpus=3 xfind_train.py --deepspeed ds_config.json
```

---

### NCCL 通信优化

Linux NCCL 性能远超 Windows。环境变量调优：

```bash
# P2P 通信（3×L20 大概率走 PCIe，无 NVLink）
export NCCL_P2P_LEVEL=NVL
# 多线程网络通信
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
# 缓冲区大小
export NCCL_BUFFSIZE=8388608
```

---

### DataLoader 优化

Linux 下多进程 DataLoader 更稳定高效：

```python
dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=4,           # Linux 多进程 fork 快
    pin_memory=True,          # 加速 CPU→GPU 传输
    prefetch_factor=2,        # 每个 worker 预取 2 batch
    persistent_workers=True   # 不重复创建进程
)
```

---

### 系统级优化

```bash
# GPU 持久模式
sudo nvidia-smi -pm 1
# GPU 时钟锁定（L20 base/max: 1215/1410 MHz）
sudo nvidia-smi -ac 1215,1410
# 文件描述符上限
ulimit -n 65536
# CPU 性能模式
sudo cpupower frequency-set -g performance
```

---

### Document Packing（数据打包，GPU 利用率 +30-50%）

短文档（几百字）只占 4096 窗口的一小段，剩余位置 pad 填满，GPU 计算的 30-40% 浪费在 pad 上。Packing 把多个短文档拼到一个 4096 窗口里：

```
无 Packing:  [doc1 | pad | pad | pad]
有 Packing:  [doc1 | <sep> | doc2 | doc3]
```

| 指标 | 无 Packing | Packing |
|---|---|---|
| GPU 有效计算利用率 | ~60-70% | **~95%** |
| 等效训练吞吐 | 基准 | **+30-50%** |
| 模型质量 | 基准 | 持平或略好（更多样化的上下文） |

实现要点：
1. 样本间插入 `<|sep|>` token 作为文档边界
2. 注意力掩码：跨文档不互相注意（`document_mask`），防止语义污染
3. 每个 packed sample 中每个子文档独立计算 loss（忽略 sep token 和子文档 pad）
4. 修改 `xfind_dataset.py` 的 `__getitem__` 和 `collate_fn`

```python
# 概念示例：从 memmap 连续取 tokens，凑满 4096
def pack_documents(all_ids, max_seq_len=4096, sep_token=2):
    samples = []
    i = 0
    while i < len(all_ids):
        sample = []
        while len(sample) < max_seq_len:
            i += 1  # 跳过行尾 EOS，开始新文档
            doc = read_next_doc(all_ids, i)  # 读到下一个行尾
            if len(sample) + len(doc) + 1 <= max_seq_len:
                sample.append(sep_token)
                sample.extend(doc)
            else:
                break
        samples.append(sample[:max_seq_len])
    return samples
```

> 有了 Packing 后，有效 batch 可以进一步增长。实际上 Packing + micro_batch=4 比 micro_batch=8 无 Packing 更高效。

---

### Z-Loss 正则化（防 loss spike，DeepSeek/PaLM 标配）

在交叉熵损失上加一项压制 logit 爆炸的惩罚项：

```python
# 标准 loss
ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

# Z-Loss: 防止 logsumexp 过大导致训练不稳定
log_z = torch.logsumexp(logits, dim=-1)  # [batch, seq]
z_loss = 1e-4 * (log_z ** 2).mean()

total_loss = ce_loss + z_loss
```

**原理**：交叉熵只关心正确 token 的概率，不关心 logit 的绝对值是否有爆炸趋势。Z-Loss 直接惩罚过大的 logsumexp（softmax 的分母），让数值范围保持紧凑。

| 收益 | 说明 |
|---|---|
| 训练稳定性 | 大降 loss spike 概率（37 层深网络的关键保障） |
| 可容忍更大 lr | 不惧怕 logit 爆炸，lr 可以从 3e-4 提到 5e-4 |
| 零显存开销 | 只需多算一个 logsumexp，无额外参数 |

---

### Liger Kernel（零改动 GPU kernel 加速）

LinkedIn 开源的 fused kernel 集合，一行 import 替换 PyTorch 原生实现：

```python
# 在 xfind_train.py 开头加一行
import liger_kernel.transformers.cross_entropy as _  # noqa
# 之后照常使用 F.cross_entropy，底层自动走 fused kernel
```

支持的融合（自动生效）：
- **Fused Linear Cross Entropy**：`logits = lm_head(x); loss = CE(logits, targets)` → 单 kernel 完成
- **RMSNorm**：fused kernel，加快 1.5×
- **SwiGLU**：fused gate + up + silu 乘法
- **RoPE**：fused cos/sin apply

| 收益 | 说明 |
|---|---|
| 训练加速 | **15-30%**（减少 kernel launch 和中间显存读写） |
| 代码改动 | **1 行 import**，零模型代码修改 |
| 兼容性 | 支持 PyTorch 2.0+，Linux x86_64 |

> 与 torch.compile 叠加使用收益更大——torch.compile 做跨 operator 的融合，Liger 是单 operator 的 fusion。

---

### 优化后的训练配置

| 参数 | 优化前（保守 DDP） | 优化后（Full Linux 方案） |
|---|---|---|
| 并行框架 | DDP | DeepSpeed ZeRO-2 |
| Attention | 手写 PyTorch | Flash Attention 2 |
| 编译 | 无 | torch.compile(mode="reduce-overhead") |
| AdamW | 标准 | fused=True |
| Kernel 融合 | 无 | Liger Kernel（Fused CE + RMSNorm + SwiGLU + RoPE） |
| 数据 | 逐文档 | **Document Packing**（利用率 60→95%） |
| 损失函数 | CrossEntropy | CrossEntropy + **Z-Loss** 正则化 |
| micro_batch | 1 | **4**（Flash Attn 省显存） |
| accumulate_grad | 32 | 8（micro_batch 大了） |
| 有效 batch | 32 | **96**（3×4×8=96） |
| num_workers | 0 | 4 |
| 预估吞吐（等效） | ~3 it/s | **等效 12-18 it/s**（真实 8-12 × Packing 1.3-1.5×） |
| 3B tokens 训练时间 | ~11 天 | **~2-3 天** |

---

## 开发阶段

### 阶段一：64K BPE Tokenizer

**目标**：训练覆盖中英双语的 64K 词表 BPE tokenizer。

**数据**：中文维基（1.83GB 清洗后）+ 英文维基（BookCorpus 或 Wikipedia EN 子集），中英比例 1:1 或 2:1。

**工具**：SentencePiece BPE，复用 `tokenizer/xfind_tokenizer.py` 的框架。

**产出**：
- `tokenizer/checkpoints/bpe_64k.model`
- `tokenizer/checkpoints/bpe_64k.vocab`

**预估耗时**：~30 分钟（CPU）。

---

### 阶段二：模型代码改造

**涉及文件**：

| 文件 | 改动 |
|---|---|
| `models/xfind_model.py` | 新增 Flash Attention 2、QK-Norm、YaRN、更新 RoPE θ 默认值 |
| `models/xfind_config.py` | 新增 `XFIND_0_6B` 配置类 |

**新增组件**：

#### Flash Attention 2（Linux 训练核心加速）

在 `GroupedQueryAttention.forward()` 中，训练模式下用 `flash_attn_func` 替换手写的 softmax(QK^T)V，推理模式保留手写（兼容 KV Cache）。

```python
# 训练时：显存 O(seq) 而非 O(seq²)，速度 3-5×
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)

# 推理时：保留手写 attention 以支持 KV Cache 拼接
# (Flash Attention 2 原生支持 KV Cache，见 flash_attn_varlen_func)
```

> 注意：Flash Attention 需要 Q/K/V 为 `[batch, seq, heads, head_dim]` 格式，
> 推理模式下 KV Cache 需要 `flash_attn_varlen_func` 处理变长序列。

#### QK-Norm

在 `GroupedQueryAttention` 中，Q 和 K 投影后各过一个 RMSNorm，再做 RoPE：

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, ...):
        ...
        # QK-Norm: Qwen3 标配，稳定训练
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def forward(self, x, ...):
        Q = self.W_q(x).view(...).transpose(1, 2)
        K = self.W_k(x).view(...).transpose(1, 2)
        V = self.W_v(x).view(...).transpose(1, 2)
        
        # QK-Norm（在 RoPE 之前）
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # RoPE
        Q, K = apply_rotary_emb(Q, K, ...)
        ...
```

#### RoPE θ = 1,000,000

修改 `precompute_freqs_cis` 的默认 theta 值，并在 `GroupedQueryAttention.__init__` 中传入。

#### YaRN 上下文扩展

在 `precompute_freqs_cis` 基础上增加 frequency scaling 逻辑：

```python
def precompute_freqs_cis(dim, max_seq_len, theta=1000000.0, yarn_scale=1.0):
    """YaRN 扩展：通过 scale 参数拉伸 RoPE 频率"""
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # YaRN scaling
    if yarn_scale > 1.0:
        freq = freq / yarn_scale
    ...
```

#### 新配置类

```python
class XFIND_0_6B_Config:
    """XFIND-Pro 默认配置"""
    vocab_size = 64000
    d_model = 1024
    num_layers = 37
    num_heads = 16
    num_kv_heads = 8
    d_ff = 3072
    max_seq_len = 4096
    dropout = 0.1
    tie_weights = False
    # 训练参数
    batch_size = 1          # seq_len=4096 时 micro-batch 需要很小
    accumulate_grad = 32    # 有效 batch = 32（对标 Qwen 小模型）
    lr = 3e-4
    warmup_ratio = 0.01
    weight_decay = 0.01
    label_smoothing = 0.1
    clip_grad = 1.0
```

**验证方式**：用新配置构建 Mini 版本（d_model=512, 8层, vocab=64K）在小数据上跑通前向+反向，确认 QK-Norm 和 YaRN 无 bug。

**预估耗时**：2-3 天（编写 + 测试）。

---

### 阶段三：数据集扩展

**目标**：积累 3-5B tokens 中英双语训练数据。

| 数据源 | 估计 tokens | 语言 | 获取方式 |
|---|---|---|---|
| 中文维基 | ~0.5B | 中文 | 已有 |
| 英文维基百科 | ~3B | 英文 | HuggingFace / 直接下载 dump |
| WikiText-103 | ~0.1B | 英文 | 可直接用 |
| BookCorpus | ~1B | 英文 | HuggingFace |

**数据处理**：复用 `tools/clean_text.py` 和 `tools/build_dataset.py`，调整 `max_len` 适应 4096 上下文。

**预估耗时**：下载 1-2 天，处理 2-3 小时。

---

### 阶段四：训练脚本适配

**涉及文件**：

| 文件 | 改动 |
|---|---|
| `xfind_train.py` | DeepSpeed + torch.compile + fused AdamW + Z-Loss + Liger Kernel |
| `xfind_dataset.py` | Document Packing + stride 适配 seq_len=4096 + 多进程 DataLoader |
| `ds_config.json` | **新增** DeepSpeed ZeRO-2 配置文件 |

**关键改动**：

1. **DeepSpeed ZeRO-2（替代 DDP）**：

```bash
deepspeed --num_gpus=3 xfind_train.py --deepspeed ds_config.json
```

2. **Liger Kernel**（1 行 import）：

```python
import liger_kernel.transformers.cross_entropy as _  # fused CE + RMSNorm + SwiGLU
```

3. **torch.compile + fused AdamW**：

```python
model = torch.compile(model, mode="reduce-overhead")
optimizer = torch.optim.AdamW(model.parameters(), ..., fused=True)
```

4. **Z-Loss 正则化**：

```python
ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
log_z = torch.logsumexp(logits, dim=-1)
z_loss = 1e-4 * (log_z ** 2).mean()
total_loss = ce_loss + z_loss
```

5. **Document Packing**（`xfind_dataset.py`）：
   - 从 memmap 读取 token 流，将多条短文档拼满 4096 窗口
   - 文档间插入 `<|sep|>` token
   - 注意力掩码阻止跨文档 attention
   - stride 设为 4096（Packing 后无重叠需求）

6. **DataLoader 多进程**：

```python
DataLoader(dataset, batch_size=4, num_workers=4,
           pin_memory=True, prefetch_factor=2, persistent_workers=True)
```

**预估耗时**：2-3 天。

---

### 阶段五：训练

**训练环境**：Linux + 3×L20 48GB + CUDA 12.4 + PyTorch 2.5+

**训练配置（Full Linux 优化方案）**：

| 参数 | 值 |
|---|---|
| 显卡 | 3×L20 48GB |
| 并行框架 | DeepSpeed ZeRO-2 |
| Attention | Flash Attention 2 |
| 编译 | torch.compile(mode="reduce-overhead") |
| Kernel | Liger Kernel（Fused CE + RMSNorm + SwiGLU + RoPE） |
| 优化器 | Fused AdamW |
| 损失函数 | CrossEntropy + Z-Loss（1e-4） |
| 数据 | Document Packing（利用率 60→95%） |
| Micro-batch | **4** |
| 梯度累积 | 8 |
| 有效 batch | **96**（3×4×8=96） |
| 峰值学习率 | 3e-4（可尝试 5e-4，Z-Loss 允许） |
| Warmup | 1% |
| 学习率衰减 | Cosine to 0 |
| 精度 | BF16 AMP |
| Gradient Checkpointing | torch.compile 内置 |
| num_workers | 4 |
| 训练 tokens | 3-5B |
| Epochs | 1 |

**启动命令**：

```bash
# 一键启动（含 NCCL 优化环境变量）
export NCCL_P2P_LEVEL=NVL
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
deepspeed --num_gpus=3 xfind_train.py --deepspeed ds_config.json
```

**预估训练时长**：3×L20，**等效 12-18 it/s**（真实 8-12 × Packing × Liger），3B tokens → **~2-3 天**。

> 对比无优化的 DDP 方案（~3 it/s，11 天）：速度 **4-6×**，时间缩短到 **1/4**。

---

### 阶段六：评估

| 评估维度 | 工具 | 对标 |
|---|---|---|
| PPL（困惑度） | `evaluation/perplexity.py` | 中文维基 test set |
| 生成质量 | `evaluation/generate_samples.py` | 50 个 prompt，人工评估 |
| CEVAL（中文知识） | CEVAL benchmark | Qwen3-0.6B |
| MMLU（英文知识） | MMLU benchmark | Qwen3-0.6B |
| 推理速度 | 单次推理耗时 | KV Cache 吞吐量 |

---

## 风险与缓解

| 风险 | 概率 | 缓解措施 |
|---|---|---|
| 64K tokenizer 中文覆盖不足 | 低 | 中英 1:1 训练数据喂 tokenizer |
| 37 层训练不稳定（loss spike） | 中 | QK-Norm + 较小 lr + gradient clipping=0.5 |
| 3B tokens 不够（欠拟合） | 中 | 加入 BookCorpus 等多源数据 |
| Flash Attention 2 安装失败 | 中 | 降级为手写 attention（性能回退但可训练） |
| DeepSpeed ZeRO-2 兼容性问题 | 低 | 降级为 DDP + torchrun |
| L20 显存不够 | 低 | Flash Attention + ZeRO-2 已大幅降低显存 |

---

## 汇总

```
XFIND-Pro 路线：
  Tokenizer (64K) → 模型改造 → 数据扩展 → 训练脚本 → 3×L20 训练 → 评估
                    ↓
          Flash Attention 2 / QK-Norm / RoPE 1M / YaRN
                    ↓
          DeepSpeed ZeRO-2 / torch.compile / fused AdamW

XFIND-Nano (39M) 已打通全部工程链路，0.6B 版本新增：
   1. 模型层：Flash Attention 2 + QK-Norm + RoPE 1M + YaRN
   2. 训练层：DeepSpeed ZeRO-2 + torch.compile + fused AdamW
   3. 数据层：64K 词表 + 中英双语 + Document Packing + seq_len=4096
   4. 损失函数：CrossEntropy + Z-Loss 正则化
   5. Kernel：Liger Kernel（Fused CE + RMSNorm + SwiGLU + RoPE）
   6. 硬件层：3×L20 48GB，预估 2-3 天

 代码改动量：
   - models/xfind_model.py：~60 行（Flash Attention 分支 + QK-Norm + RoPE/YaRN 参数）
   - models/xfind_config.py：~30 行（新配置类）
   - xfind_train.py：~40 行（DeepSpeed + compile + fused AdamW + Z-Loss + Liger import + DataLoader）
   - xfind_dataset.py：~80 行（Document Packing + stride + num_workers）
   - ds_config.json：~15 行（新文件）
 
 **从 Mini 到 0.6B，不是简单的放大——是技术栈的全面升级。**
```
