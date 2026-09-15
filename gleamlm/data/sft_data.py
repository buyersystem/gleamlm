"""SFT 数据集 — JSONL → ChatML 格式 → loss mask。

Supports both single-turn and multi-turn conversation formats.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.chatml import format_chatml

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = [
    "你是一个有帮助的AI助手。",
    "你是一个友善的中文对话助手，请用简洁清晰的语言回答问题。",
    "你是一个知识渊博的助手，请准确回答问题。",
    "You are a helpful AI assistant.",
]


class SFTDataset(Dataset):
    """SFT dataset: JSONL -> ChatML format -> loss mask.

    Supports two formats, auto-detected from the first line:

    Single-turn (backward-compatible):
        {"instruction": "...", "output": "..."}

    Multi-turn:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    Loss mask: only the LAST assistant turn contributes to loss.
    In multi-turn mode, all prior turns (including earlier assistant replies)
    are treated as context and masked.

    加载期 precheck 剔除病理样本 (截断后 prompt 归零 / labels 全 -100),
    统计见 self.precheck_stats。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BBPETokenizer,
        max_seq_len: int = 512,
        inject_system_ratio: float = 0.2,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.inject_system_ratio = inject_system_ratio
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        self.multiturn: bool = False
        self.data: list[dict[str, Any]] = []

        raw_lines: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: skipping line {i} in {data_path}: {e}")
                    continue
                raw_lines.append(item)

        if not raw_lines:
            raise ValueError(f"No valid samples in {data_path}")

        first_has_messages = "messages" in raw_lines[0]

        if first_has_messages:
            self.multiturn = True
            for i, item in enumerate(raw_lines):
                msgs = item.get("messages")
                if not isinstance(msgs, list) or len(msgs) < 2:
                    logger.warning(f"Warning: skipping line {i} in {data_path}: invalid messages")
                    continue
                has_assistant = any(m.get("role") == "assistant" for m in msgs)
                if not has_assistant:
                    logger.warning(f"Warning: skipping line {i} in {data_path}: no assistant turn")
                    continue
                self.data.append({"messages": msgs})
            logger.info(f"Loaded {len(self.data)} multi-turn SFT samples from {data_path}")
        else:
            self.multiturn = False
            for i, item in enumerate(raw_lines):
                if "messages" in item:
                    msgs = item.get("messages")
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        has_assistant = any(m.get("role") == "assistant" for m in msgs)
                        if has_assistant:
                            self.data.append({"messages": msgs})
                            continue
                if "instruction" in item and "output" in item:
                    self.data.append({"instruction": item["instruction"], "output": item["output"]})
                else:
                    logger.warning(f"Warning: skipping line {i} in {data_path}: unknown format")
            single_count = sum(1 for d in self.data if "instruction" in d)
            multi_count = sum(1 for d in self.data if "messages" in d)
            logger.info(
                f"Loaded {len(self.data)} SFT samples from {data_path} "
                f"({single_count} single-turn, {multi_count} multi-turn)"
            )

        rng = random.Random(42)
        self._system_prompts: list[str] = []
        for _ in range(len(self.data)):
            if rng.random() < inject_system_ratio:
                self._system_prompts.append(rng.choice(SYSTEM_PROMPTS))
            else:
                self._system_prompts.append("")

        # 加载期全量预检: 剔除截断病理样本 (见 _precheck);
        # 同步过滤 _system_prompts 保持系统提示与样本 idx 对齐
        self.precheck_stats: dict[str, Any] = self._precheck()

    def __len__(self) -> int:
        return len(self.data)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_bos=False, add_eos=False)

    def _prompt_and_full_ids(self, idx: int) -> tuple[list[int], int]:
        """tokenize 单条样本 → (full_ids, prompt token 数)。

        单轮与多轮在此汇合, __getitem__ 与 precheck 共用同一入口,
        保证预检看到的截断/掩码输入与训练完全一致。
        """
        item = self.data[idx]
        if "messages" in item:
            messages = item["messages"]
            full_ids = self._encode(format_chatml(messages, add_generation_prompt=False))
            prompt_ids = self._encode(format_chatml(messages[:-1], add_generation_prompt=True))
        else:
            msgs: list[dict[str, str]] = []
            system_prompt = self._system_prompts[idx]
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": item["instruction"]})
            prompt_ids = self._encode(format_chatml(msgs, add_generation_prompt=True))
            full_ids = self._encode(
                format_chatml(
                    msgs + [{"role": "assistant", "content": item["output"]}],
                    add_generation_prompt=False,
                )
            )
        return full_ids, len(prompt_ids)

    def _mask_and_pad(
        self, full_ids: list[int], prompt_len: int
    ) -> tuple[list[int], list[int], int, int]:
        """截断(保尾) → prompt 区 loss mask → pad。

        返回 (input_ids, labels, P_adj, dropped): P_adj 为截断后仍保留的 prompt
        token 数 (0 = prompt 被整体切掉, 病理); dropped 为头部丢弃 token 数。
        """
        dropped = 0
        if len(full_ids) > self.max_seq_len:
            dropped = len(full_ids) - self.max_seq_len
            full_ids = full_ids[-self.max_seq_len :]
            prompt_len = max(0, prompt_len - dropped)

        input_ids = full_ids[:-1]
        labels = list(full_ids[1:])

        mask_end = min(prompt_len, len(labels))
        for i in range(mask_end - 1):
            labels[i] = -100

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return input_ids, labels, prompt_len, dropped

    def _precheck(self) -> dict[str, Any]:
        """加载期全量预检: 按 __getitem__ 同款逻辑扫描全部样本, 剔除病理并计数。

        病理 a: 超长样本保尾截断后 prompt 归零 → 全序列 (含残段上下文) 参与
        loss, 训练信号污染;
        病理 b (防御): labels 全 -100 → CE 无有效 token (nan) 静默白训。
        """
        total = len(self.data)
        healthy: list[int] = []
        prompt_lost: list[int] = []
        all_masked: list[int] = []
        truncated = 0
        lengths: list[int] = []
        for idx in range(total):
            full_ids, prompt_len = self._prompt_and_full_ids(idx)
            lengths.append(len(full_ids))
            _, labels, P_adj, dropped = self._mask_and_pad(full_ids, prompt_len)
            truncated += 1 if dropped else 0
            if P_adj == 0:
                prompt_lost.append(idx)
            elif all(label == -100 for label in labels):
                all_masked.append(idx)
            else:
                healthy.append(idx)

        if prompt_lost or all_masked:
            logger.warning(
                f"precheck 剔除 {len(prompt_lost) + len(all_masked)} 条病理样本"
                f" (prompt_lost={len(prompt_lost)}, all_masked={len(all_masked)})"
                f" / 共 {total} 条; 样例索引: {(prompt_lost + all_masked)[:3]}"
            )
        if not healthy:
            raise ValueError(
                "precheck 后无健康样本 (全部命中病理): 检查数据长度与 max_seq_len 设置"
            )
        self.data = [self.data[i] for i in healthy]
        self._system_prompts = [self._system_prompts[i] for i in healthy]

        lengths.sort()
        n = len(lengths)
        stats: dict[str, Any] = {
            "total": total,
            "ok": len(healthy),
            "prompt_lost": len(prompt_lost),
            "all_masked": len(all_masked),
            "truncated": truncated,
            "lengths": {
                "p50": lengths[n // 2],
                "p90": lengths[int(n * 0.9)],
                "p99": lengths[min(int(n * 0.99), n - 1)],
                "max": lengths[-1],
            },
        }
        logger.info(
            f"precheck: {stats['ok']}/{total} 条健康, truncated={truncated}, "
            f"token 长度 p50={stats['lengths']['p50']} p90={stats['lengths']['p90']} "
            f"p99={stats['lengths']['p99']} max={stats['lengths']['max']}"
        )
        return stats

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        full_ids, prompt_len = self._prompt_and_full_ids(idx)
        input_ids, labels, _, _ = self._mask_and_pad(full_ids, prompt_len)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return input_ids, labels
