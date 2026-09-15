"""rl_trainer.sample_responses 单测 (GRPO rollout 与难度过滤共用的采样函数)。"""

import torch

from gleamlm.data.rl_data import tokenize_prompts
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.rl_trainer import sample_responses
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

VOCAB_SIZE = 12002
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
D_FF = 512
SEQ_LEN = 64


class _StubEosModel(torch.nn.Module):
    """每步都给出 eos one-hot logits → 采样/贪心均立刻命中 eos。"""

    def __init__(self, vocab_size: int, eos_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def forward(self, ids: torch.Tensor):
        logits = torch.full((ids.size(0), ids.size(1), self.vocab_size), -1e9)
        logits[:, -1, self.eos_id] = 1e9
        return logits, None, None, None


class _StubFixedModel(torch.nn.Module):
    """每步输出固定非 eos token → 只能靠上限退出 (截断)。"""

    def __init__(self, vocab_size: int, fixed_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.fixed_id = fixed_id

    def forward(self, ids: torch.Tensor):
        logits = torch.full((ids.size(0), ids.size(1), self.vocab_size), -1e9)
        logits[:, -1, self.fixed_id] = 1e9
        return logits, None, None, None


class _StubMixedModel(torch.nn.Module):
    """row 0 立即 eos、row 1 永不 eos → 验证逐样本完成判定与 pad 占位。"""

    def __init__(self, vocab_size: int, eos_id: int, fixed_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.fixed_id = fixed_id

    def forward(self, ids: torch.Tensor):
        logits = torch.full((ids.size(0), ids.size(1), self.vocab_size), -1e9)
        logits[0, -1, self.eos_id] = 1e9
        logits[1, -1, self.fixed_id] = 1e9
        return logits, None, None, None


class TestSampleResponses:
    """完成判定语义: eos 收尾 → truncated=False; 上限退出 → truncated=True。"""

    def test_eos_finish_marks_not_truncated(self):
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        model = _StubEosModel(VOCAB_SIZE, tokenizer.eos_id)
        prompt_ids = torch.tensor([[tokenizer.bos_id, 100]])
        gen, trunc = sample_responses(
            model,
            tokenizer,
            prompt_ids,
            group_size=2,
            max_new_tokens=8,
            temperature=0.8,
            seq_len=SEQ_LEN,
        )
        assert len(gen) == 2 and len(trunc) == 2
        # [B] 逐样本掩码: 全样本立即出 eos → 无截断
        assert all(not t.any().item() for t in trunc)
        # 第一步即命中 eos break → 只多 1 个 token
        assert all(g.size(1) == prompt_ids.size(1) + 1 for g in gen)

    def test_max_new_tokens_truncation_flagged(self):
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        fixed = 100 if tokenizer.eos_id != 100 else 101
        model = _StubFixedModel(VOCAB_SIZE, fixed)
        prompt_ids = torch.tensor([[tokenizer.bos_id, 100]])
        gen, trunc = sample_responses(
            model,
            tokenizer,
            prompt_ids,
            group_size=2,
            max_new_tokens=5,
            temperature=0.8,
            seq_len=SEQ_LEN,
        )
        assert all(t.all().item() for t in trunc)
        assert all(g.size(1) == prompt_ids.size(1) + 5 for g in gen)

    def test_seq_len_cap_truncation_flagged(self):
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        fixed = 100 if tokenizer.eos_id != 100 else 101
        model = _StubFixedModel(VOCAB_SIZE, fixed)
        prompt_ids = torch.tensor([[tokenizer.bos_id, 100]])
        gen, trunc = sample_responses(
            model,
            tokenizer,
            prompt_ids,
            group_size=1,
            max_new_tokens=50,  # 远大于 seq_len 余量 → 由 seq_len 上限截停
            temperature=0.8,
            seq_len=prompt_ids.size(1) + 3,
        )
        assert trunc[0].all().item()
        assert gen[0].size(1) == prompt_ids.size(1) + 3

    def test_mixed_batch_per_sample_mask(self):
        """row 0 出 eos、row 1 截断 → 掩码逐样本独立 (非全列一刀切);
        row 0 的 eos 后一律 pad 占位 (防止 eos 后续写混入回答)。"""
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        fixed = 100 if tokenizer.eos_id != 100 else 101
        model = _StubMixedModel(VOCAB_SIZE, tokenizer.eos_id, fixed)
        prompt_ids = torch.tensor([[tokenizer.bos_id, 100], [tokenizer.bos_id, 100]])
        gen, trunc = sample_responses(
            model,
            tokenizer,
            prompt_ids,
            group_size=1,
            max_new_tokens=5,
            temperature=0.0,
            seq_len=SEQ_LEN,
        )
        assert trunc[0].tolist() == [False, True]
        # row 1 未完成 → 跑满 5 步 (同步 batch 等最慢样本)
        assert gen[0].size(1) == prompt_ids.size(1) + 5
        assert gen[0][0, prompt_ids.size(1)].item() == tokenizer.eos_id
        assert (gen[0][0, prompt_ids.size(1) + 1 :] == tokenizer.pad_id).all()

    def test_greedy_group_identical(self):
        """temperature=0 贪心 → 同 prompt 各轨迹完全相同 (教学断言: 这正是
        GRPO 需要采样而非贪心的原因, 贪心会退化为零方差组)。"""
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        model = GleamLMModel(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            d_ff=D_FF,
            dropout=0.0,
            max_seq_len=SEQ_LEN,
            tie_weights=True,
            pad_token_id=tokenizer.pad_id,
        ).eval()
        prompt_ids = tokenize_prompts(["你好"], tokenizer, SEQ_LEN)
        with torch.no_grad():
            gen, trunc = sample_responses(
                model,
                tokenizer,
                prompt_ids,
                group_size=2,
                max_new_tokens=8,
                temperature=0.0,
                seq_len=SEQ_LEN,
            )
        assert torch.equal(gen[0], gen[1])
        assert len(trunc) == 2
