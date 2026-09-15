"""industrial.rl_reward 守卫单测 — 1a 截断判定 / 奖励口径 / 零方差审计。

无 TRL 依赖 (CI 不装 industrial extra): 直接测 build_reward_fn 返回的奖励
函数与审计统计; 布局用例模拟 TRL 的展平输入 (同 prompt 的 N 个响应连续)。
"""

import pytest

from industrial.rl_reward import build_reward_fn

EOS = 2
_OK = [10, EOS]  # 以 eos 收尾 (完整回答)
_CUT = [10, 11]  # 末 token 非 eos (截断)


class TestTruncationGuard:
    """1a: 截断回答 clamp(max=0), 负分保留 (与 manual/grpo.py 同口径)。"""

    def test_truncated_hit_clamped_to_zero(self):
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(["2+2=?"], ["答案是 4 但没说完"], completion_ids=[_CUT], ground_truth=["4"])
        assert rewards == [0.0]  # 半截文本碰巧含 gt → 不给 +1.0

    def test_finished_hit_gets_plus_one(self):
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(["2+2=?"], ["答案是 4"], completion_ids=[_OK], ground_truth=["4"])
        assert rewards == [1.0]

    def test_truncated_penalty_kept(self):
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(["q"], [""], completion_ids=[_CUT], ground_truth=["4"])
        assert rewards == [-1.0]  # clamp(max=0) 只砍正分

    def test_missing_ids_skips_guard(self):
        fn, _ = build_reward_fn(EOS, 2)
        assert fn(["q"], ["答案是 4"], ground_truth=["4"]) == [1.0]

    def test_eos_none_skips_guard(self):
        fn, _ = build_reward_fn(None, 2)
        rewards = fn(["q"], ["答案是 4"], completion_ids=[_CUT], ground_truth=["4"])
        assert rewards == [1.0]

    def test_empty_completion_ids_treated_as_truncated(self):
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(["q"], ["答案是 4"], completion_ids=[[]], ground_truth=["4"])
        assert rewards == [0.0]


class TestRewardScore:
    """奖励口径: 有 gt 规则匹配 / 空 gt 与无 gt 列走启发式分级。"""

    def test_blank_gt_uses_heuristic(self):
        # 空 gt 串不再 `"" in response` 恒真全命中; 短文本启发式 0.0
        fn, _ = build_reward_fn(EOS, 2)
        assert fn(["q"], ["答案"], completion_ids=[_OK], ground_truth=[""]) == [0.0]
        assert fn(["q"], ["答案"], completion_ids=[_OK], ground_truth=["   "]) == [0.0]

    def test_missing_gt_column_uses_heuristic(self):
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(["q", "q"], ["短", "行1\n行2。"], completion_ids=[_OK, _OK])
        assert rewards[0] == 0.0
        assert rewards[1] == pytest.approx(0.3)

    def test_heuristic_not_constant(self):
        # 原"非空全 +1.0"常数兜底 → 组内零方差; 分级启发式有区分度
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(["q", "q"], ["a", "行1\n行2。"], completion_ids=[_OK, _OK])
        assert rewards[0] != rewards[1]

    def test_gt_column_mixed_blank_rows(self):
        # gt 列存在但部分行空白: 逐行判定 (空白行走启发式, 非整体切换)
        fn, _ = build_reward_fn(EOS, 2)
        rewards = fn(
            ["q", "q"], ["答案 4", "行1\n行2。"], completion_ids=[_OK, _OK], ground_truth=["4", ""]
        )
        assert rewards[0] == 1.0
        assert rewards[1] == pytest.approx(0.3)


class TestZeroVarianceAudit:
    """1b: 按组切块统计零方差 (TRL 展平布局假设 + 自证停用机制)。"""

    def test_zero_variance_group_counted(self):
        fn, stats = build_reward_fn(EOS, 2)
        # 组1 两条都对 (全 1.0, 零方差); 组2 一对一错
        fn(
            ["q1", "q1", "q2", "q2"],
            ["4", "4", "4", "不知道"],
            completion_ids=[_OK] * 4,
            ground_truth=["4"] * 4,
        )
        assert stats.total == 4 and stats.groups == 2
        assert stats.zero_var_groups == 1
        assert "zero_var_groups=1/2" in stats.summary()

    def test_layout_mismatch_disables_audit(self):
        fn, stats = build_reward_fn(EOS, 2)
        fn(["q1", "q2", "q1", "q2"], ["a", "b", "c", "d"], completion_ids=[_OK] * 4)
        assert not stats.layout_ok and stats.groups == 0
        assert "审计停用" in stats.summary()

    def test_wrong_prompts_len_disables_audit(self):
        fn, stats = build_reward_fn(EOS, 2)
        fn(["q1"], ["a", "b"], completion_ids=[_OK] * 2)
        assert not stats.layout_ok

    def test_truncated_counter(self):
        fn, stats = build_reward_fn(EOS, 2)
        fn(["q", "q"], ["x", "y"], completion_ids=[_OK, _CUT])
        assert stats.truncated == 1 and stats.total == 2

    def test_summary_without_audit_data(self):
        _, stats = build_reward_fn(EOS, 2)
        assert stats.summary() == "reward guard: truncated=0/0; 零方差审计无数据"
