# CogVLA LIBERO 评估指南

## 环境准备

```bash
conda activate openvla
cd /workspace/laiminxin/vla-opt/third_party/CogVLA
```

## 模型

CogVLA 官方提供的 checkpoint 需要从 GitHub/HuggingFace 下载：
- [CogVLA GitHub](https://github.com/JiuTian-VL/CogVLA)

## 评估命令

### 使用脚本

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts-sh/eval.sh
```

### 手动运行

```bash
# LIBERO-Spatial
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/cogvla-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50

# LIBERO-Object
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/cogvla-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 50

# LIBERO-Goal
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/cogvla-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task 50

# LIBERO-10
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/cogvla-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --num_trials_per_task 50
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrained_checkpoint` | 模型路径 | 必填 |
| `--task_suite_name` | 任务套件 | `libero_spatial` |
| `--center_crop` | 中心裁剪 (必须开启) | `True` |
| `--num_trials_per_task` | 每任务试验次数 | 50 |
| `--seed` | 随机种子 | 7 |

## 日志输出

评估结果自动保存到 `rollouts/` 目录

## 输出指标

> ⚠️ **当前脚本仅输出成功率，不包含延迟/显存等性能指标**

| 指标 | 是否支持 |
|------|----------|
| 成功率 | ✅ |
| 每任务成功数 | ✅ |
| 回放视频 | ✅ |
| 延迟分析 | ❌ |
| 显存使用 | ❌ |
| 推理速度 | ❌ |

## 硬件要求

- **推理**: 1× GPU (~16GB VRAM)
- **论文环境**: NVIDIA A800, Python 3.10, PyTorch 2.2.0

## 注意事项

1. **`--center_crop True` 必须开启** (训练时使用了随机裁剪)
2. **建议使用训练时相同的 GPU** 进行测试，否则性能可能下降
