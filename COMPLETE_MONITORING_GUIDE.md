# CogVLA 训练监控指南（单一适配入口）

本指南只保留**一个入口**来监控 CogVLA 训练过程，避免在训练脚本中堆一堆 adapter。
底层仍然保留模块化的 observer/adapters，但你只需要关心一个对象：`CogVLAMonitoring`。

> 通用监控设计与输出格式，请先看：`/workspace/laiminxin/vla-opt/monitor/README.md`。

---

## 0) 总结：为什么只需要一个 adapter？

- CogVLA 有 3 个核心交互点（FiLM / MoE / LFP），再加上训练指标。
- **用户侧**只需一个 facade：`CogVLAMonitoring`。
- **内部**用多个 observer 记录不同子系统，但这些由 facade 自动管理。

---

## 1) 快速接入（训练脚本）

### 1.1 初始化（run_dir/run_id 已就绪后）

```python
from monitor.pruning.adapters.cogvla import CogVLAMonitoring, CogVLAMonitoringConfig

monitor_cfg = CogVLAMonitoringConfig(
    enable_train_monitor=True,
    enable_lfp_monitor=True,
    enable_film_monitor=True,
    enable_moe_monitor=True,
    enable_aggr_monitor=True,
    monitor_log_interval=100,
    detailed_log_interval=500,
    pruning_event_log_interval=10,
    enable_tensorboard=False,
    film_sample_layers=[0, 9, 18, 26],
)

monitor = CogVLAMonitoring(
    run_dir=run_dir,
    run_id=run_id,
    config_name=cfg.dataset_name,
    cfg=monitor_cfg,
)
```

### 1.2 绑定模型（必须在 DDP/LoRA 之后）

```python
vla = wrap_ddp(vla, device_id, find_unused=True)
monitor.attach(vla)
```

### 1.3 训练循环中记录

```python
monitor.log_step(step)

with monitor.step(step, mode="train") as h:
    h.log(loss=loss_value, lr=lr, extra={"curr_action_l1_loss": cur_l1})
```

### 1.4 训练结束后收尾

```python
summary_paths = monitor.finalize()
print(summary_paths)
```

---

## 2) 开关控制（与 finetune.py 一致）

这些开关与 `vla-scripts/finetune.py` 的配置保持一致：

- `enable_monitor`：训练指标（loss/lr/...）
- `enable_lfp_monitor`：LFP 剪枝监控（仅 `use_lfp=True` 时生效）
- `enable_film_monitor`：FiLM 调制监控（仅 `use_film=True` 时生效）
- `enable_moe_monitor`：MoE Router 监控（仅 `use_aggr=True` 时生效）
- `enable_aggr_monitor`：Aggregation Tokens 监控（仅 `use_aggr=True` 时生效）

---

## 3) CogVLA 模块插桩位置（理解后才知道挂哪里）

CogVLA 是三段式结构：

1. **Stage 1：视觉骨干 + FiLM**
   - 位置：`FiLMedVisionTransformerBlock`（`prismatic/models/film_vit_wrapper.py`）
   - 监控点：FiLM 的 `gamma/beta`（由文本均值嵌入生成）

2. **Stage 2：MoE Fusion**
   - 位置：`PrismaticForConditionalGeneration.aggregation_router`（`prismatic/extern/hf/modeling_prismatic.py`）
   - 监控点：MoEAggregator 输出的 SigLIP/DINOv2 权重

3. **Stage 3：LFP Routing**
   - 位置：`LlamaDecoderLFPLayer`（`prismatic/models/modeling_llama.py`）
   - 监控点：每层视觉 token 的保留比例

4. **Aggregation Tokens**
   - 位置：Vision Backbone Aggregator（`prismatic/models/vit_wrapper_reg.py`）
   - 监控点：压缩比（固定参数）

`CogVLAMonitoring` 已经按这些位置绑定，**不用再手动找模块**。

---

## 4) 输出结构（与 monitor 统一）

```text
runs/{run_id}/
├── monitor_logs/              # 训练指标
│   ├── {run_id}.jsonl
│   ├── {run_id}_summary.json
│   └── {run_id}_meta.json
├── lfp_logs/                  # LFP 剪枝
├── film_logs/                 # FiLM 调制
├── moe_logs/                  # MoE 权重
└── aggr_logs/                 # Aggregation 压缩比
```

---

## 5) 常见问题

### Q: 为什么不直接在训练脚本里堆 4 个 adapter？
A: 这是**内部实现**需要。用户侧只需要 `CogVLAMonitoring` 一个对象，
它会统一管理 attach/detach、log_step、finalize 和输出结构。

### Q: 如何只监控训练 loss？
A: 只开 `enable_train_monitor=True`，其它全部 False。

---

## 6) 进一步阅读

- `/workspace/laiminxin/vla-opt/monitor/README.md`
- `/workspace/laiminxin/vla-opt/monitor/pruning/README.md`
