"""TRL 奖励守卫 — 工业轨 GRPO/RLOO 与 手写轨同口径的 1a/1b 防御。

背景（对齐 manual/grpo.py 的 1a/1b 守门）:
  1a 截断奖励守卫: 未以 eos 收尾的截断回答 clamp(max=0) —— 半截文本碰巧
     包含 ground_truth 不再拿 +1.0, 只惩罚不受益。
  1b 零方差审计: TRL 的 rollout 在 trainer 内部, 无法像 手写轨那样动态
     重采样替换; 本模块按组统计零方差组数（训练结束打印 summary）。TRL
     内部零方差组优势≈0 —— 无梯度贡献（等价 手写轨"末轮仍零方差的行
     不进 loss"）, 只浪费采样预算、不破坏训练; 数据侧根治走
     data_tools/rl/filter_by_difficulty.py（pass_rate>0.9 的过易题预剔除）。

奖励口径与 手写轨共用 gleamlm.trainer.rl_trainer.compute_reward
（有 gt 规则匹配 / 无 gt 启发式分级）, 替换原来"无 gt 时非空全 +1.0"的
长度兜底 —— 常数奖励组内零方差、无优势梯度, 训练实为纯 KL 拉扯。

本模块零第三方依赖（标准库 + gleamlm 核心）, 可在无 TRL 环境单测。
"""

from __future__ import annotations

from typing import Any

from gleamlm.trainer.rl_trainer import compute_reward


class RewardGuardStats:
    """reward_funcs 逐次调用的截断/零方差审计（仅用于训练结束日志）。

    零方差判定按"同 prompt 的 N 个响应连续"切块（TRL GRPO/RLOO 的展平
    布局）。块内 prompt 不一致或长度不整除时停用审计（layout_ok=False）,
    只影响统计数字, 不影响奖励值与训练本身。
    """

    def __init__(self) -> None:
        self.total = 0  # 打分的 completion 总数
        self.truncated = 0  # 1a: 截断判定（未以 eos 收尾）的样本数
        self.groups = 0  # 完成零方差判定的组数
        self.zero_var_groups = 0  # 组内奖励全同（优势≈0）的组数
        self.layout_ok = True  # 展平布局假设未被违反

    def update(
        self,
        prompts: list[Any] | None,
        rewards: list[float],
        trunc_flags: list[bool],
        num_generations: int,
    ) -> None:
        n = len(rewards)
        self.total += n
        self.truncated += sum(trunc_flags)
        if not self.layout_ok:
            return
        if n == 0 or num_generations <= 0 or n % num_generations != 0:
            self.layout_ok = False
            return
        if prompts is None or len(prompts) != n:
            self.layout_ok = False
            return
        for s in range(0, n, num_generations):
            blk = prompts[s : s + num_generations]
            first = str(blk[0])
            if any(str(p) != first for p in blk[1:]):
                # 同 prompt 的 N 个响应不再连续 → 无法按组判定, 停用审计
                self.layout_ok = False
                return
            self.groups += 1
            grp = rewards[s : s + num_generations]
            if max(grp) - min(grp) < 1e-6:
                self.zero_var_groups += 1

    def summary(self) -> str:
        base = f"reward guard: truncated={self.truncated}/{self.total}"
        if not self.layout_ok:
            return base + "; 零方差审计停用 (TRL 展平布局与预期不符)"
        if self.groups == 0:
            return base + "; 零方差审计无数据"
        return (
            base + f", zero_var_groups={self.zero_var_groups}/{self.groups}"
            " — 零方差组无优势梯度; 根治: data_tools/rl/filter_by_difficulty.py 预过滤"
        )


def build_reward_fn(eos_token_id: int | None, num_generations: int) -> tuple[Any, RewardGuardStats]:
    """构造 TRL reward_funcs 用的奖励函数（闭包携带 eos 判定与审计统计）。

    Args:
        eos_token_id: HF tokenizer 的 eos id; None 时停用 1a 截断判定。
        num_generations: 每组响应数（TRL 展平布局按此切块审计）。

    Returns:
        (reward_fn, stats): reward_fn 签名符合 TRL reward_funcs 约定
        （prompts/completions + completion_ids/数据集列 kwargs）;
        stats 累计截断/零方差审计, 训练结束打印 summary()。
    """
    stats = RewardGuardStats()

    def default_reward(
        prompts: list[Any],
        completions: list[str],
        completion_ids: list[Any] | None = None,
        ground_truth: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """规则 + 启发式奖励（口径同 手写轨 compute_reward）+ 1a 截断守卫。

        TRL 传入 completions（str 列表）、completion_ids（list[list[int]],
        生成的实际 token, 含 eos 若已生成）与数据集各列（ground_truth）。
        空/缺失 ground_truth 的行走启发式 —— 逐行判定, 不再整体切换。
        """
        rewards: list[float] = []
        trunc_flags: list[bool] = []
        for i, comp in enumerate(completions):
            gt = ground_truth[i] if ground_truth is not None and i < len(ground_truth) else None
            if gt is not None and not str(gt).strip():
                gt = None  # 空 gt 串: `"" in response` 恒真 → 防全命中, 视为无 gt
            reward = compute_reward(comp or "", gt)
            if eos_token_id is not None and completion_ids is not None:
                ids = completion_ids[i] if i < len(completion_ids) else []
                # 空生成或末 token 非 eos = 截断（上限截停 / 提前结束）
                truncated = len(ids) == 0 or int(ids[-1]) != int(eos_token_id)
            else:
                truncated = False  # 缺 ids / eos 未定义 → 无法判定, 不守卫
            if truncated:
                reward = min(reward, 0.0)  # 1a: 截断只惩罚不受益（负分保留）
            rewards.append(reward)
            trunc_flags.append(truncated)
        stats.update(prompts, rewards, trunc_flags, num_generations)
        return rewards

    return default_reward, stats
