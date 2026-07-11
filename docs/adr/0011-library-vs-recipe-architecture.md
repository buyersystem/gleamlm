# ADR-0011: 核心库 vs 变体配方 — 架构分工边界

## 背景

ADR-0005 确立了 `gleamlm/` 共享核心库 + 变体目录的单一仓库结构。经过训练模块重构（R1-R3）、预处理管线迁移、量化统一等操作后，暴露出边界模糊的问题：

- `nano/quantize.py` 被精简为 6 行 import 链，用户打开后无法理解量化流程
- `gleamlm/deploy/quantize.py` 承担了实现，但变体目录需要的是**可读配方而非黑盒抽象**

核心库的深层封装与变体目录的教育可见性之间存在张力。两端都需要克制：核心库不过度封装编排逻辑，变体目录不过度精简为 import 桩。

## 决策：三层角色模型

```
┌──────────────────────────────────────────────────────┐
│ gleamlm/         精于 How — 可导入的函数与类          │
│   参数化、变体无关                                     │
│   提供深度模块（小接口，大实现）                        │
│   不包含硬编码的变体默认参数                            │
├──────────────────────────────────────────────────────┤
│ gleamlm-{variant}/  精于 What + When — 完整配方       │
│   自包含的端到端脚本                                   │
│   显式写出变体特定的架构参数和超参                       │
│   编排可见，实施可委托                                 │
├──────────────────────────────────────────────────────┤
│ data_tools/ scripts/ configs/  数据获取 & 评估         │
│   依赖外部 API 或一次性任务                           │
│   不纳入核心库，不纳入变体目录                         │
└──────────────────────────────────────────────────────┘
```

### 核心库 (`gleamlm/`)

**包含**：模型定义、分词器、数据集类、训练循环（函数级）、推理引擎、评估函数、预处理算法（清洗/去重/混合）、量化/导出、配置加载、LR 调度、checkpoint 校验。

**不含**：`if __name__ == "__main__": main()` 模式的编排脚本（`prepare_data.py` 和 `cli.py` 是历史遗留例外，标记为待迁出）。

**判断标准 — 删除测试**：如果从 `gleamlm/` 删除一个函数，变体脚本是否还能通过复制粘贴独立运行？如果不能，该函数必须在 `gleamlm/` 内。

### 变体目录 (`gleamlm-nano/`, `gleamlm-lite/`, 未来的 `gleamlm-pro/`)

**包含**：每个训练阶段的入口脚本（`train.py`, `sft.py`, `dpo.py`, `infer.py`, `quantize.py`）。每个脚本展示该变体的完整端到端流程，包括显式架构参数和超参默认值。

**不含**：可被多个变体复用的逻辑。如果一个函数在 nano 和 lite 中都需要，提纯到核心库。

**配方测试**：
1. 用户将 `nano/sft.py` 复制到自己目录，修改几个参数（架构、学习率、epoch 数）后能直接运行 — **通过**。
2. 复制后发现 import 链条断裂、缺少函数实现、无法理解每个步骤在做什么 — **不通过**。

### 编排可见原则

变体脚本的 `main()` 应该让读者不跳转到任何其他文件就能理解**发生了什么**：

```python
# ✅ 正确 — 步骤可见，实施委托
# 步骤 1: 加载 checkpoint
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

# 步骤 2: 提取架构参数
arch = {"d_model": 512, "num_layers": 12, ...}  # 变体特定，显式写出

# 步骤 3: 构建模型 + 加载权重
model = GleamLMModel(**arch, tie_weights=False)
model.load_state_dict(checkpoint["model_state_dict"])

# 步骤 4: 转为 FP16
model = model.half()

# 步骤 5: 保存
torch.save({"model_state_dict": model.state_dict(), ...}, output)
```

```python
# ❌ 错误 — 黑盒，看不到流程
from gleamlm.deploy.quantize import main

if __name__ == "__main__":
    main()
```

## 分工矩阵

| 关注点 | 在 `gleamlm/` 中 | 在变体目录中 |
|---|---|---|
| 模型架构参数 | — | `d_model=512`, `num_layers=12`, ... |
| 训练超参默认值 | — | `lr=5e-6`, `epochs=3`, `batch_size=8`, ... |
| 路径常量 | `utils/paths.py`（获取函数） | `DEFAULT_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")` |
| 训练循环实现 | `train_one_epoch_sft()`, `train_one_epoch_dpo()` 等 | — |
| 训练编排 | — | `main()`: 解析参数 → 加载模型 → 创建数据集 → 循环 → 保存 |
| 数据集类 | `SFTDataset`, `DPODataset`, `LMDataset` | — |
| 损失函数 | `dpo_loss()`, `compute_log_probs()` | — |
| 推理引擎 | `sample_token()`, `TextStreamer`, `generate_response()` | `load_model()` + `generate()` + `interactive()` 编排上述 |
| 量化实现 | `deploy/quantize.py`: `quantize_to_fp16(input, output)` — 纯函数 | `quantize()` 展示流程：load → extract arch → half() → save |
| 预处理算法 | `preprocessing/`：清洗、去重、混合的纯函数 | — |
| 预处理编排 | `preprocessing/prepare_data.py`（待迁出，see 待清理项） | — |
| 配置加载 | `utils/config.py`：`load_config()`, `load_config_as_args()` | — |

## 反模式

### 核心库中禁止
- 硬编码任何变体的架构参数（`d_model=512` 只能出现在 `nano/` 中）
- `if __name__ == "__main__"` 的编排脚本（`prepare_data.py` 和 `cli.py` 是待迁移的例外，see 待清理项）
- 导入只在特定变体中存在的模块或路径

### 变体目录中禁止
- 复制 `gleamlm/` 里已有实现的函数体
- 10 行以下的纯 import 桩（不满足配方测试）
- 跨变体的 import（nano 不能 import lite，反过来也不行）

## 案例：`quantize.py` 的正确形态

| 位置 | 行数 | 内容 |
|---|---|---|
| `gleamlm/deploy/quantize.py` | ~100 | 提供 `quantize_to_fp16(input, output)` — 可导入的纯函数，变体无关 |
| `nano/quantize.py` | ~130 | 配方：argparse → load checkpoint → extract arch → build model → half() → save。显式写出 `NANO_ARCH = {d_model=512, ...}` |
| `lite/quantize.py` | ~130 | 同上，`LITE_ARCH = {d_model=768, num_heads=12, use_flash_attn=True}` |

## 影响

1. **变体目录是自包含配方**：新建 `pro/` 时，复制 `lite/` 目录，修改架构参数和超参即可运行，不需要理解 `gleamlm/` 内部实现
2. **核心库是单一真相源**：所有可复用逻辑有且仅有一份实现，配方脚本通过 import 委托执行层
3. **遵循 ADR-0005**（单一仓库 + 共享核心库），细化其边界："变体专用脚本" = 完整的端到端配方，而非线性 import 链
4. **遵循 ADR-0010**（实验驱动变体演进）：变体配方的参数是该变体实验结果的固化，配方脚本即实验记录

## 待清理项（已知偏离）

| 文件 | 问题 | 计划 |
|---|---|---|
| `gleamlm/inference/cli.py` | `if __name__ == "__main__"` 编排入口 | 保留在核心库（作为通用工具入口）；变体 `infer.py` 为变体默认入口 |
