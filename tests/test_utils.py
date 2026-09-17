"""纯函数单元测试 — get_lr_cosine/wsd, dpo_loss, compute_log_probs, format_chatml, assert_same_architecture, metrics 哨兵行, metric window, gpu 探测回退"""

import math

import torch

from gleamlm.trainer.dpo_loss import compute_log_probs, dpo_loss
from gleamlm.trainer.schedulers import get_lr_cosine, get_lr_wsd
from gleamlm.utils.chatml import format_chatml
from gleamlm.utils.gpu import gpu_stats
from gleamlm.utils.meter import MetricWindow, window_max
from gleamlm.utils.metrics import SENTINEL, emit_metric, format_metric_line, parse_metric_line


def assert_same_architecture(
    checkpoint_config: dict[str, int],
    current_config: dict[str, int],
    source: str = "checkpoint",
) -> None:
    critical_keys = [
        "vocab_size",
        "d_model",
        "num_layers",
        "num_heads",
        "num_kv_heads",
    ]
    mismatches: list[str] = []
    for key in critical_keys:
        ckpt_val = checkpoint_config.get(key)
        cur_val = current_config.get(key)
        if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
            mismatches.append(f"  {key}: {source}={ckpt_val}, current={cur_val}")
    if mismatches:
        raise ValueError(
            f"Architecture mismatch between {source} and current model:\n"
            + "\n".join(mismatches)
            + "\nRefusing to load. Verify model architecture matches the checkpoint."
        )


# ---- LR 调度 ----


def test_lr_cosine_warmup():
    lr = get_lr_cosine(step=5, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    # step=5, warmup_steps=10, 5/10=0.5
    assert lr == 0.5


def test_lr_cosine_end():
    lr = get_lr_cosine(step=999, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    assert abs(lr - 0.1) < 1e-4


def test_lr_cosine_midpoint():
    lr = get_lr_cosine(step=500, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    assert abs(lr - 0.55) < 0.01


def test_lr_cosine_clamped_past_decay_view():
    """step 越过衰减视野 (lr_decay_steps < 实际步数) 必须停在 min_lr, 不得余弦回升。"""
    lr_end = get_lr_cosine(step=999, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    lr_over = get_lr_cosine(step=5000, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    assert abs(lr_over - lr_end) < 1e-9


def test_lr_wsd_warmup():
    lr = get_lr_wsd(
        step=5, total_steps=1000, warmup_ratio=0.02, stable_ratio=0.8, min_lr_ratio=0.05
    )
    assert lr == 5 / 20  # step=5, warmup_steps=20


def test_lr_wsd_stable():
    lr = get_lr_wsd(
        step=400, total_steps=1000, warmup_ratio=0.02, stable_ratio=0.8, min_lr_ratio=0.05
    )
    assert lr == 1.0


def test_lr_wsd_decay_end():
    lr = get_lr_wsd(
        step=999, total_steps=1000, warmup_ratio=0.02, stable_ratio=0.8, min_lr_ratio=0.05
    )
    assert abs(lr - 0.05) < 1e-4


# ---- DPO ----


def test_dpo_loss_equal():
    """chosen=rejected 时 loss 应为 -log sigmoid(0) = -log(0.5)"""
    policy_cho = torch.tensor([-2.0])
    policy_rej = torch.tensor([-2.0])
    ref_cho = torch.tensor([-3.0])
    ref_rej = torch.tensor([-3.0])
    loss = dpo_loss(policy_cho, policy_rej, ref_cho, ref_rej, beta=1.0)
    expected = -math.log(torch.sigmoid(torch.tensor(0.0)).item())
    assert abs(loss.item() - expected) < 1e-5


def test_dpo_loss_chosen_preferred():
    """policy 偏好 chosen 时 loss 应低于 policy 偏好 rejected"""
    # policy prefers chosen: p_cho > r_cho, p_rej = r_rej → term > 0 → low loss
    loss_correct = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-2.0]),
        torch.tensor([-5.0]),
        torch.tensor([-2.0]),
        beta=1.0,
    )
    # policy prefers rejected: p_rej > r_rej, p_cho = r_cho → term < 0 → high loss
    loss_wrong = dpo_loss(
        torch.tensor([-5.0]),
        torch.tensor([0.0]),
        torch.tensor([-5.0]),
        torch.tensor([-2.0]),
        beta=1.0,
    )
    assert loss_correct.item() < loss_wrong.item()


def test_dpo_loss_beta_effect():
    """更大的 beta 放大偏好信号，使正确模型的 loss 更低"""
    loss_low_beta = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-2.0]),
        torch.tensor([-5.0]),
        torch.tensor([-2.0]),
        beta=0.1,
    )
    loss_high_beta = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-2.0]),
        torch.tensor([-5.0]),
        torch.tensor([-2.0]),
        beta=1.0,
    )
    assert loss_high_beta.item() < loss_low_beta.item()


# ---- compute_log_probs ----


def test_compute_log_probs_basic():
    logits = torch.zeros(1, 3, 5)  # [B=1, seq=3, vocab=5]
    input_ids = torch.tensor([[1, 1, 1]])
    mask = torch.tensor([[1.0, 1.0]])
    result = compute_log_probs(logits, input_ids, mask)
    assert result.shape == (1,)
    assert abs(result.item() - 2 * math.log(0.2)) < 1e-5


def test_compute_log_probs_mask():
    logits = torch.zeros(2, 3, 5)
    input_ids = torch.tensor([[1, 1, 1], [1, 1, 1]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = compute_log_probs(logits, input_ids, mask)
    assert result.shape == (2,)
    assert abs(result[0].item() - math.log(0.2)) < 1e-5
    assert abs(result[1].item() - math.log(0.2)) < 1e-5


# ---- ChatML ----


def test_chatml_single_message():
    result = format_chatml([{"role": "user", "content": "hi"}])
    assert result == "<|im_start|>user\nhi<|im_end|>\n"


def test_chatml_with_generation_prompt():
    result = format_chatml(
        [{"role": "system", "content": "Be helpful."}], add_generation_prompt=True
    )
    assert result == ("<|im_start|>system\nBe helpful.<|im_end|>\n<|im_start|>assistant\n")


def test_chatml_multi_turn():
    result = format_chatml(
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    )
    assert result == ("<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n")


# ---- assert_same_architecture ----


def test_assert_same_architecture_match():
    # 不应抛出异常
    assert_same_architecture(
        {"vocab_size": 12002, "d_model": 512}, {"vocab_size": 12002, "d_model": 512}
    )


def test_assert_same_architecture_mismatch():
    import pytest

    with pytest.raises(ValueError):
        assert_same_architecture({"vocab_size": 12002}, {"vocab_size": 999})


def test_assert_same_architecture_partial():
    # 一方缺 key 时不报错
    assert_same_architecture({"d_model": 512}, {"vocab_size": 12002, "d_model": 512})


# ---- metrics 哨兵行（训练指标通道契约） ----


def test_metric_roundtrip():
    line = format_metric_line(split="train", step=7, total=100, loss=1.234, lr=3e-4)
    assert line.startswith(SENTINEL)
    assert parse_metric_line(line) == {
        "split": "train",
        "step": 7,
        "total": 100,
        "loss": 1.234,
        "lr": 3e-4,
    }


def test_metric_emit_and_parse(capsys):
    emit_metric(split="val", step=3, loss=1.2, ppl=3.32)
    out = capsys.readouterr().out.strip()
    assert parse_metric_line(out) == {"split": "val", "step": 3, "loss": 1.2, "ppl": 3.32}


def test_metric_parse_tolerates_whitespace():
    # 日志行拼接可能带回车/空白；两端空白应被容忍
    line = "  " + format_metric_line(step=1, loss=0.5, lr=1e-3) + "\r\n"
    assert parse_metric_line(line) == {"step": 1, "loss": 0.5, "lr": 1e-3}


def test_metric_parse_glued_to_tqdm_frame():
    # tqdm 帧以 \r 分帧（无换行）: 哨兵 print 会直接粘在帧尾（真实 DPO 日志 18/18 如此）
    line = (
        "DPO Epoch 0:  4%| | 38/881 [00:08<02:25, 5.80it/s, loss=0.5977, lr=9.96e-07]"
        + SENTINEL
        + '{"split":"train","step":25,"loss":0.59767115,"lr":9.96e-07}'
    )
    assert parse_metric_line(line) == {
        "split": "train",
        "step": 25,
        "loss": 0.59767115,
        "lr": 9.96e-07,
    }


def test_metric_parse_rejects_non_sentinel_and_malformed():
    assert parse_metric_line("step 1/10 (10.0%)  loss=1.5000  lr=0.000100") is None
    assert parse_metric_line(SENTINEL + "{broken json") is None
    assert parse_metric_line(SENTINEL + "[1, 2]") is None  # 非 dict


def test_window_max_is_none_safe_and_keeps_peaks():
    """grad_norm 的窗口口径：记 max 而不是末值 —— 预警发散靠尖峰。"""
    assert window_max(None, None) is None  # 未启用裁剪：整窗都没值
    assert window_max(None, 1.5) == 1.5  # 首个 step 建基
    assert window_max(1.5, None) == 1.5  # 中途没记录（该步没做 optimizer step）
    assert window_max(1.5, 0.4) == 1.5  # 后续更小：保留尖峰
    assert window_max(1.5, 9.9) == 9.9  # 出现尖峰：必须被记住


def test_metric_window_loss_is_mean_but_grad_norm_is_peak():
    """同一窗口里两个口径**刻意不同**：loss 关心趋势（均值），grad_norm 关心尖峰（max）。

    回归点：grad_norm 曾按「窗口内最后一个 step」记，中间 log_interval-1 个 step 的
    尖峰无声消失 —— 那正是要提前预警发散的那一段。
    """
    w = MetricWindow(torch.device("cpu"))
    for loss, g in [(1.0, 0.8), (3.0, 0.9), (2.0, 12.0), (2.0, 1.0), (2.0, 0.7)]:
        w.add_loss(loss)
        w.add_grad_norm(g)
    assert w.batches == 5  # loss 的分母
    assert w.loss is not None and math.isclose(w.loss, 2.0)  # 均值
    assert w.grad_norm == 12.0  # 窗口内尖峰
    assert w.grad_norm != 0.7, "与末值口径必须可区分，否则本测试没有区分度"

    w.reset()
    assert w.loss is None and w.grad_norm is None and w.batches == 0


def test_metric_window_reset_keeps_instantaneous_snapshots(monkeypatch):
    """reset() 只清窗口量；tok_per_s / 显存是「最近一次」的快照，不清。"""
    clock = iter([10.0, 11.0])
    monkeypatch.setattr("gleamlm.utils.meter.time.perf_counter", lambda: next(clock))
    w = MetricWindow(torch.device("cpu"))
    w.start()
    w.stop(100.0)
    w.add_loss(1.0)
    w.reset()
    assert w.loss is None and w.batches == 0
    assert w.tok_per_s == 100.0, "瞬时量不该被 reset 清掉（且必须仍等于刚才那次的值）"


def test_metric_window_throughput_uses_callers_token_count(monkeypatch):
    """tok_per_s = 调用方给的 tokens ÷ 该段耗时 —— **口径由调用点决定**。

    训练型给 batch×seq（DPO 给 2×batch×seq），生成型给本迭代新生成的 token 数；
    本类不猜，所以这里用两条不同的 tokens 验证同一段耗时会给出不同速率。
    """
    clock = iter([100.0, 102.0])
    monkeypatch.setattr("gleamlm.utils.meter.time.perf_counter", lambda: next(clock))
    w = MetricWindow(torch.device("cpu"))
    w.start()
    assert w.stop(1000.0) == 500.0  # dt=2s → 500 tok/s
    assert w.tok_per_s == 500.0 and w.last_dt == 2.0

    clock = iter([200.0, 202.0])
    monkeypatch.setattr("gleamlm.utils.meter.time.perf_counter", lambda: next(clock))
    w.start()
    assert w.stop(64.0) == 32.0  # 同样 2s，生成型只算它自己生成的 64 个 token


def test_metric_window_stop_without_start_raises():
    """没 start() 就 stop() 必须报错，不能静默给出一个假的 0 tok/s。"""
    w = MetricWindow(torch.device("cpu"))
    try:
        w.stop(1.0)
    except RuntimeError as e:
        assert "start()" in str(e)
    else:
        raise AssertionError("未 start() 就 stop() 应当抛 RuntimeError")


def test_metric_window_emit_kwargs_shape_and_cpu_fallback():
    """哨兵行只吃这四个键；CPU 上显存为 0（核心库 CPU fallback 是硬约束，CI 也跑到）。"""
    w = MetricWindow(torch.device("cpu"))
    assert set(w.emit_kwargs()) == {"loss", "grad_norm", "tok_per_s", "gpu_mem"}
    assert w.loss is None and w.grad_norm is None, "空窗口要如实给 None，不能撒谎成 0"
    w.refresh_gpu()
    assert w.gpu_mem == 0.0 and w.gpu_mem_total == 0.0 and w.gpu_util == 0.0
    assert w.emit_kwargs()["gpu_mem"] == 0.0


def test_metric_window_stop_does_not_refresh_gpu(monkeypatch):
    """stop() 不得刷显存 —— 查询较重（缺 NVML 时起 nvidia-smi 子进程），
    刷新节奏由调用方在「上报点」控制（见 MetricWindow docstring）。"""
    calls = []

    def spy(device, local_rank=0):
        calls.append(local_rank)
        return 1.0, 2.0, 3.0, 4.0

    monkeypatch.setattr("gleamlm.utils.meter.gpu_stats", spy)
    w = MetricWindow(torch.device("cuda"))
    w.start()
    w.stop(64.0)
    assert calls == [], "stop() 触发显存查询会把刷新节奏打乱"
    w.refresh_gpu()
    assert calls == [0] and w.gpu_mem == 2.0


def test_gpu_stats_falls_back_to_smi_row_by_local_rank(monkeypatch):
    """主路径失败（无 pynvml 的 Windows 实测如此）回退 nvidia-smi：按 local_rank 取行。"""

    def no_nvml(*a, **k):
        raise RuntimeError("pynvml missing")

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_nvml)
    monkeypatch.setattr(
        "gleamlm.utils.gpu.subprocess.check_output",
        lambda *a, **k: "55, 2048, 8192\n10, 512, 8192\n",
    )
    # 第二行 = local_rank 1；MiB → GiB
    assert gpu_stats(torch.device("cuda"), local_rank=1) == (10.0, 0.5, 0.0, 8.0)


def test_gpu_stats_all_zero_when_probes_fail(monkeypatch):
    """两条路径都失败 → 全 0 降级（监控降为「无数据」，不抛异常打断训练）。"""

    def no_nvml(*a, **k):
        raise RuntimeError("pynvml missing")

    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_nvml)
    monkeypatch.setattr("gleamlm.utils.gpu.subprocess.check_output", no_smi)
    assert gpu_stats(torch.device("cuda")) == (0.0, 0.0, 0.0, 0.0)
