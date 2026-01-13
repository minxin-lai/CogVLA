#!/bin/bash
# Simplified CogVLA evaluation script (referencing LightVLA)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set LIBERO dataset path
export LIBERO_DATASET_PATH="/workspace/laiminxin/datasets/libero_rlds"
# Ensure tracer is in PYTHONPATH
export PYTHONPATH="/workspace/laiminxin/vla-opt/third_party/CogVLA:/workspace/laiminxin/vla-opt:${PYTHONPATH}"

echo "=== CogVLA Evaluation ==="
echo "Start time: $(date)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRACE_OUT_DIR="runs/libero_spatial_eval_${TIMESTAMP}"
echo "Trace output directory: ${TRACE_OUT_DIR}"

# Default Checkpoint
CKPT="/workspace/laiminxin/models/CogVLA_runs/cogvla-libero/openvla-7b+libero_4_task_suites_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--cogvla-libero-4task--80000_chkpt"

# Ensure LIBERO is installed
python -c "import libero" >/dev/null 2>&1 || {
  echo "ERROR: Python package 'libero' not found."
  echo "Install it first (see third_party/CogVLA/docs/LIBERO.md):"
  echo "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git"
  echo "  pip install -e LIBERO"
  echo "  pip install -r experiments/robot/libero/libero_requirements.txt"
  exit 1
}

# Record initial GPU memory
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0
fi

CUDA_VISIBLE_DEVICES=5 \
time python "${SCRIPT_DIR}/experiments/robot/libero/run_libero_eval.py" \
  --pretrained_checkpoint "${CKPT}" \
  --task_suite_name libero_spatial \
  --center_crop true \
  --num_trials_per_task 1 \
  --seed 7 \
  --trace_out_dir "${TRACE_OUT_DIR}" \
  --trace_max_dumps_per_run 10 \
  --trace_save_policy_images true \
  --trace_dump_routing true \
  --trace_dump_attn true \
  --trace_attn_layers "31" \
  --trace_dump_moe true

echo ""
echo "=== Evaluation Complete ==="
echo "End time: $(date)"

if [ -d "${TRACE_OUT_DIR}/dumps" ] && [ -n "$(find "${TRACE_OUT_DIR}/dumps" -name '*.pt' | head -n 1)" ]; then
  python -m tracer.plot_routing_overlays --exp_dir "${TRACE_OUT_DIR}"
else
  echo "WARNING: no dumps found under ${TRACE_OUT_DIR}/dumps; skip overlay plotting."
fi
