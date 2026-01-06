# CogVLA 如何“搭建在 OpenVLA 之上”：迁移 / 集成文档（基于本仓代码）

> 适用范围：你手头有一个“OpenVLA 风格”的代码库（HF `AutoModelForVision2Seq` + `trust_remote_code` + RLDS 训练脚手架），想把 CogVLA 的优化模块（路由/稀疏化/聚合）迁移进去，或理解 CogVLA 是如何复用 OpenVLA 的训练与推理流程的。

本仓的 CogVLA 并不是把 OpenVLA 当成纯依赖库来调用，而是**保留 OpenVLA/Prismatic 的工程骨架**（`prismatic/` + `vla-scripts/` + `experiments/`）并在关键位置加上 CogVLA 的模块与训练/保存/加载逻辑。因此目录结构与 OpenVLA 类似。

---

## 0. 入口与代码分层（你需要先认清这几类文件）

**HF Remote Code（“让 AutoModel 能加载本仓逻辑”的关键）**
- `prismatic/extern/hf/configuration_prismatic.py`：`OpenVLAConfig`，新增 CogVLA 开关（`use_aggr/use_lfp/...`）。
- `prismatic/extern/hf/modeling_prismatic.py`：`OpenVLAForActionPrediction` 的 HF 版本实现；集成 MoE 融合、LFP、以及 action-token 注意力补丁。
- `prismatic/extern/hf/processing_prismatic.py`：processor/image processor（推理/训练都用）。

**训练脚本（“OpenVLA fine-tune 脚手架 + CogVLA 开关接入”）**
- `vla-scripts/finetune.py`：主训练入口；负责把 `--use_aggr/--use_lfp/...` 写入 config、包 LoRA、解冻路由模块、保存/合并 checkpoint。
- `vla-scripts/merge_lora_weights_and_save.py`：当 `--merge_lora_during_training False` 时，离线把 LoRA + non-lora trainables 合并回 base 权重。

**推理/部署（“按 OpenVLA 方式跑起来，同时补齐 CogVLA 的额外组件加载”）**
- `experiments/robot/openvla_utils.py`：`get_vla/get_vla_action`；负责加载模型、按需包裹 vision backbone、加载单独保存的 `vision_backbone--*.pt`、加载 `dataset_statistics.json`。
- `vla-scripts/deploy.py`：FastAPI/uvicorn server，调用 `get_vla_action`。

---

## 1. CogVLA 的优化模块清单（模块 -> 开关 -> 代码位置 -> 作用）

### 1.1 Instruction-driven “专家路由/融合”（MoE 聚合 SigLIP + DINOv2）

**核心思路**
- 当使用 fused vision backbone（两路视觉编码器）时，不再只做固定 concat+project，而是用**文本嵌入驱动的 gating**对两路视觉特征做加权融合。
- 注意这里的 gating 粒度是 **per-sample**：先对文本 token 做均值池化得到一个向量，再输出 2 个专家权重；该权重对整段视觉 token 序列做缩放并求和（不是 per-token 动态路由）。

**开关**
- `OpenVLAConfig.vision_aggregate_type = 'moe'`（或 `'concat'`）

**实现位置**
- `prismatic/models/router.py`：`MoEAggregator`（MLP router，输出专家权重，做加权求和）。
- `prismatic/extern/hf/modeling_prismatic.py`：
  - 初始化：`PrismaticForConditionalGeneration.__init__` 中创建 `aggregation_router + featurizer_proj + fused_featurizer_proj`（`vision_aggregate_type == 'moe'`）。
  - 运行时：`_aggregate_patch_features()` 按 `vision_aggregate_type` 走 `moe` 或 `concat`。

**注意点**
- `vision_aggregate_type == 'moe'` 强制要求 `config.use_fused_vision_backbone == True`（两路专家存在才有 MoE）。

---

### 1.2 Visual Token Compression（Aggregation Tokens：把 256 patches 压到 K 个 token）

**核心思路**
- 在 ViT 输入序列末尾追加 `K = num_vision_aggr` 个**可训练聚合 token**，并将它们当作“视觉 token 输出”（而不是原始 patch tokens）。
- 这样 OpenVLA 的 prefix 长度显著变短（`256 -> K`），训练/推理时 LLM 的跨模态 token 数减少。

**开关**
- `OpenVLAConfig.use_aggr = True`
- `OpenVLAConfig.num_vision_aggr = K`（例如 32/64）

**实现位置**
- `prismatic/models/vit_wrapper_reg.py`
  - `FiLMedPrismaticVisionBackboneAggregator`：FiLM + aggregation tokens（`use_film=True` 且 `use_aggr=True`）。
  - `PrismaticVisionBackboneAggregator`：无 FiLM，仅 aggregation tokens（`use_film=False` 且 `use_aggr=True`）。
  - 两者都通过 monkey-patch `timm VisionTransformer.forward`，将返回的“patch 输出”替换为聚合 token（并重写 `get_num_patches()` 返回 `K`）。

**训练/推理都会受影响的点**
- `vision_backbone.get_num_patches()` 的语义被改变：当 `use_aggr=True` 时它返回 `num_vision_aggr`，后续所有基于 patch 数计算索引的位置都必须跟着变（训练脚本里显式处理了）。

---

### 1.3 LFP Sparsification（LLM 内部对视觉 tokens 逐层稀疏化/剪枝）

**核心思路**
- 在 Llama 的部分 decoder layers 上引入 `TokenRouter`，对序列中的 tokens 打分并选择保留子集；主要作用在 **visual tokens** 上，从而减少后续层的注意力/MLP 计算。
- 代码里通过 `router_factor` 控制每层保留的视觉 token 比例；`lfp_type` 支持按层衰减/交错等策略。

**开关（写在 text_config 上）**
- `OpenVLAConfig.text_config.use_lfp = True`
- `OpenVLAConfig.text_config.lfp_average_factor`：平均保留比例
- `OpenVLAConfig.text_config.lfp_type`：如 `shiftedcos_decay_0.85_0.15`/`deep_all`/`interleave`
- `OpenVLAConfig.text_config.lfp_enable_film`：是否在 router 中对 vision tokens 做 FiLM（用 text 汇聚向量调制 vision）

**实现位置**
- `prismatic/models/modeling_llama.py`
  - `replace_llama_forward()`：monkey-patch `transformers LlamaModel.__init__`，将部分层替换成 `LlamaDecoderLFPLayer`。
  - `LlamaDecoderLFPLayer.forward()`：prefill 阶段（`seq_len > 1`）执行 top-k token 选择；decode 阶段（`seq_len == 1`）不剪枝。
- `prismatic/extern/hf/modeling_prismatic.py`
  - `PrismaticForConditionalGeneration.__init__`：当 `config.text_config.use_lfp` 时，先调用 `replace_llama_forward()` 再实例化 `AutoModelForCausalLM.from_config()`。

**关键限制**
- `LlamaDecoderLFPLayer.forward()` 里有 `assert self.config._attn_implementation != "flash_attention_2"`：LFP 这条路径不支持 flash-attn-2。

---

### 1.4 Parallel Action Chunking 的注意力补丁（Action tokens 之间允许非因果互看）

**核心思路**
- OpenVLA 风格的并行 action 生成会把一段 action tokens（+ stop token）拼到序列末尾，并希望它们在同一 forward 里互相可见（类似一个局部“非因果块”）。
- 这里通过改写 SDPA attention 的 mask，把 action-token 区块的 causal mask 清零。

**实现位置**
- `prismatic/models/modeling_llama.py`：`llama_sdpa_attention_forward()`（标注了 `CATTEN`）
- `prismatic/extern/hf/modeling_prismatic.py`：import 时调用 `replace_llama_spda_forward()`，全局 monkey-patch `transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward`。

---

## 2. CogVLA 如何“接入 OpenVLA 模型”（HF 加载机制与本地 checkpoint 的关键处理）

### 2.1 `auto_map`：让 HF 知道要用本仓的类

当 `cfg.vla_path/cfg.pretrained_checkpoint` 是**本地目录**时，需要：
- 注册 auto classes（否则 `AutoModelForVision2Seq` 不知道 `openvla` 类型映射到哪个类）；
- 修改 checkpoint 目录下的 `config.json:auto_map`，指向本仓实现。

对应代码：
- `experiments/robot/openvla_utils.py:get_vla()`：本地 checkpoint 时 `AutoConfig.register(...)` + `update_auto_map(...)`。
- `vla-scripts/finetune.py`：本地 base model 时 `AutoConfig.register(...)`；并在主进程执行 `update_auto_map(...)`。

### 2.2 `check_model_logic_mismatch`：把“当前代码版本”同步进 checkpoint 目录

`experiments/robot/openvla_utils.py:check_model_logic_mismatch()` 会把：
- `prismatic/extern/hf/modeling_prismatic.py`
- `prismatic/extern/hf/configuration_prismatic.py`

复制到 checkpoint 目录根下（并在不一致时备份旧文件）。

目的：当你在本仓修改了模型逻辑并希望**加载本地 checkpoint 时生效**，就必须保证 checkpoint 目录里引用到的是最新版本的 python 文件（HF `trust_remote_code` 会从 checkpoint repo/dir 的“remote code”里 import）。

---

## 3. 训练流程：CogVLA 在 OpenVLA fine-tune 脚手架上做了哪些接入

以 `vla-scripts/finetune.py` 为主线，按执行顺序梳理：

### 3.1 载入 base OpenVLA + 写入 CogVLA 配置
- 从 HF 拉 base model：`snapshot_download(repo_id=cfg.vla_path)`（当 `cfg.vla_path` 是 hub id 时）
- `vla_cfg = AutoConfig.from_pretrained(cfg.vla_path)` 后写入新增开关：
  - `vla_cfg.use_aggr / use_film / num_vision_aggr / vision_aggregate_type`
  - `vla_cfg.text_config.use_lfp / lfp_*`
  - 代码位置：`vla-scripts/finetune.py:931`

### 3.2 LoRA 注入 + “非 LoRA 训练参数”的处理
- LoRA 目标模块来自 `find_all_linear_names()`，但会排除：
  - MoE/Router（如 `aggregation_router`、LLM router）
  - vision aggregator 的投影层（如 `featurizer_proj/fused_featurizer_proj`）
  - FiLM 的 `scale/shift`
  - 代码位置：`vla-scripts/finetune.py:958`
- 然后显式解冻：
  - MoE 组件：`aggregation_router + featurizer_proj + fused_featurizer_proj`（`vla-scripts/finetune.py:971`）
  - LFP router：`language_model` 中名称包含 `router` 的参数（`vla-scripts/finetune.py:980`）

> 这些“非 LoRA”训练参数会在保存/合并 checkpoint 时以 `non_lora_trainables--*.pt` 的形式单独落盘（见 3.4）。

### 3.3 视觉侧 wrapper（FiLM / Aggregation Tokens）
- 若 `use_film` 或 `use_aggr`：
  - 用 `FiLMedPrismaticVisionBackbone` 或 `*Aggregator` 覆盖 `vla.model.vision_backbone`
  - 代码位置：`vla-scripts/finetune.py:988`

### 3.4 Checkpoint 结构（这是推理能否跑通的关键）

`save_training_checkpoint()` 会写入（主进程）：
- `checkpoint_dir/`：processor 与（可选）merged 模型权重
- `checkpoint_dir/lora_adapter/`：LoRA adapter（始终保存）
- `checkpoint_dir/dataset_statistics.json`：动作/本体归一化统计（推理时用来 unnormalize）
- 额外组件（按需）：
  - `action_head--*.pt`
  - `proprio_projector--*.pt`
  - `noisy_action_projector--*.pt`
  - **`vision_backbone--*.pt`（当 `use_film` 或 `use_aggr` 时）**
  - 代码位置：`vla-scripts/finetune.py:631`

合并策略分两种：
- **在线合并（`--merge_lora_during_training True`）**：保存 `merged_vla.save_pretrained(checkpoint_dir)`（`vla-scripts/finetune.py:711`）
- **离线合并（`--merge_lora_during_training False`）**：
  - 不会在 checkpoint_dir 写入 merged model 权重；
  - 会额外保存 `non_lora_trainables--*.pt`；
  - 之后必须跑 `vla-scripts/merge_lora_weights_and_save.py` 把 LoRA + non-lora 合并回 base 权重。

**非常重要：vision backbone 为什么要单独存 `vision_backbone--*.pt`？**
- 这取决于你的“合并策略”：
  - 如果你用本仓的离线合并脚本 `vla-scripts/merge_lora_weights_and_save.py`，它会从 `--base_checkpoint` 重建模型、加载 LoRA + `non_lora_trainables--*.pt`，但**不会重建/合并 `use_film/use_aggr` 的 vision wrapper**；因此需要额外的 `vision_backbone--*.pt` 在推理阶段回灌。
  - 如果你选择在训练阶段直接把“包含 wrapper 的完整模型”`save_pretrained()` 到 checkpoint（并且后续推理直接加载它），理论上可以不依赖单独的 `vision_backbone--*.pt`；但本仓脚本默认仍会保存该文件以便兼容离线合并工作流。

---

## 4. 推理流程：如何正确加载 CogVLA checkpoint 并跑 action

### 4.1 加载模型（含 vision backbone 的“二次包裹 + state_dict 回灌”）

入口：`experiments/robot/openvla_utils.py:get_vla()`

关键点：
- `AutoModelForVision2Seq.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)` 加载 merged 权重
- 若 `vla.config.use_film or vla.config.use_aggr`：
  - 调用 `_apply_film_to_vla()`（函数名历史原因）创建对应 vision wrapper（FiLM / Aggregation Tokens）
  - 从 `vision_backbone--*.pt` 读取并 `load_state_dict` 回灌到 wrapper 里
  - 代码位置：`experiments/robot/openvla_utils.py:297`
- 从 `dataset_statistics.json` 读取 `norm_stats`（推理 unnormalize 依赖）：
  - 代码位置：`experiments/robot/openvla_utils.py:391`

### 4.2 生成动作（单机推理或 server 推理）

入口：`experiments/robot/openvla_utils.py:get_vla_action()`
- 本仓推理默认使用 HF 模型类 `prismatic/extern/hf/modeling_prismatic.py:OpenVLAForActionPrediction.predict_action()`；`prismatic/models/vlas/openvla.py` 里的 `OpenVLA.predict_action()`（走 `generate`）不是这条主推理路径的入口。
- 组 prompt：`In: What action should the robot take to ...?\nOut:`
- `processor(prompt, image)` 得到 `input_ids/attention_mask/pixel_values`
- 多图输入：把 wrist 图的 `pixel_values` 在 channel 维拼接（`get_vla_action():790`）
- 调用 `vla.predict_action(...)`：
  - 不带 `action_head`：走离散 action token decode
  - 带 `action_head`：走 L1 regression 或 diffusion（见 `prismatic/extern/hf/modeling_prismatic.py:1030`）

部署入口：`vla-scripts/deploy.py`（FastAPI `/act`）

---

## 5. 迁移到你自己的 OpenVLA 代码库：建议的最小迁移集合（按依赖顺序）

> 下面以“你已有 OpenVLA/Prismatic 风格工程”为前提，描述需要迁移/对齐的最小文件集合与注意事项。

### 5.1 必迁移：配置与 HF remote code
- `prismatic/extern/hf/configuration_prismatic.py`
  - `OpenVLAConfig` 新增字段：`vision_aggregate_type/num_vision_aggr/use_aggr/use_film`
  - `text_config` 上新增字段：`use_lfp/lfp_type/lfp_average_factor/lfp_enable_film`
- `prismatic/extern/hf/modeling_prismatic.py`
  - MoE 聚合初始化与 `_aggregate_patch_features()`
  - `config.text_config.use_lfp` 时的 `replace_llama_forward()`
  - import-time 的 `replace_llama_spda_forward()`

### 5.2 必迁移：CogVLA 模块实现
- `prismatic/models/router.py`：`MoEAggregator`
- `prismatic/models/vit_wrapper_reg.py`：aggregation tokens wrapper（含 `get_num_patches()` 语义变化）
- `prismatic/models/modeling_llama.py`：
  - LFP layer + monkey patch
  - SDPA attention mask（CATTEN）
- `prismatic/models/film_vit_wrapper.py`：FiLM vision block wrapper

### 5.3 必迁移：训练/保存/合并/推理脚手架的“契约”

如果你只迁移模型不迁移脚手架，很容易在 checkpoint 结构上踩坑。最少需要对齐这些“契约”：
- **训练侧**
  - 如果你采用“离线合并 LoRA”的工作流：`use_aggr/use_film` 时需要保存 `vision_backbone--*.pt`（wrapper 的参数不会被离线合并脚本写进最终权重）
  - `merge_lora_during_training False` 时必须保存 `non_lora_trainables--*.pt`
- **合并侧**
  - `merge_lora_weights_and_save.py` 需要把 `non_lora_trainables--*.pt` load 回 base 再 merge LoRA
  - 仍不包含 vision backbone wrapper 权重（推理侧需要依赖 `vision_backbone--*.pt` 回灌）
- **推理侧**
  - `use_aggr/use_film` 时必须“二次包裹”vision backbone 并加载 `vision_backbone--*.pt`
  - 必须加载 `dataset_statistics.json` 得到 `norm_stats`（否则 `unnorm_key`/unnormalize 会失败）

对应实现可以直接参考：
- 训练保存：`vla-scripts/finetune.py:631`
- 离线合并：`vla-scripts/merge_lora_weights_and_save.py:44`
- 推理加载：`experiments/robot/openvla_utils.py:254`

---

## 6. 常见问题 / 排错清单

- **我想复现实验脚本参数长什么样**：参考 `scripts-sh/finetune.sh`（训练）、`scripts-sh/merge.sh`（离线合并）、`scripts-sh/eval_aloha_deploy.sh`（部署）。
- **典型训练/合并/部署命令（与 `scripts-sh/*.sh` 一致）**：
  - 训练：`torchrun ... vla-scripts/finetune.py --vla_path ... --use_film True --use_aggr True --num_vision_aggr 64 --use_lfp True --lfp_enable_film True --merge_lora_during_training False ...`
  - 合并：`python vla-scripts/merge_lora_weights_and_save.py --base_checkpoint ... --lora_finetuned_checkpoint_dir ...`
  - 部署：`python vla-scripts/deploy.py --pretrained_checkpoint ... --num_images_in_input 3 --use_proprio True --center_crop True --unnorm_key ...`
- **LFP 报错 `flash_attention_2` 不支持**：把模型加载的 attention 实现改成 `eager` 或 `sdpa`（LFP layer 内有硬断言）。
- **加载本地 checkpoint 后逻辑没生效**：确认 `config.json` 的 `auto_map` 已被 `update_auto_map()` 写入，并且 checkpoint 根目录下存在最新的 `modeling_prismatic.py/configuration_prismatic.py`（`check_model_logic_mismatch()` 会同步）。
- **推理时报找不到 `vision_backbone--*.pt`**：说明训练时 `use_film/use_aggr` 开了但没保存该文件；或你只保存了 merged 权重但丢了额外组件文件。
- **只 merge 了 LoRA 但没 merge non-lora**：MoE/LFP router 等参数可能在 `non_lora_trainables--*.pt`，需要一起 load 回 base 再 merge。

---

## 7. 适配到“其它模型”的关键检查点（把坑提前写出来）

把 CogVLA 的方法迁移到“不是 OpenVLA-7B / Prismatic”的模型时，优先逐项核对这些假设（否则容易 silent-wrong）：

1) **Token 布局假设（非常关键）**
- `prismatic/extern/hf/modeling_prismatic.py:963` 的回归/离散路径用 `NUM_PATCHES + NUM_PROMPT_TOKENS` 来切片定位 action tokens。
- LFP 选择逻辑在 `prismatic/models/modeling_llama.py:312` 默认假设序列形如：`[BOS, vision_tokens..., text_tokens..., action_tokens..., (pad)...]`，并通过 `force_select_mask` 强制保留非视觉 tokens（仅从视觉 tokens 里剪）。
- 如果你换了 prompt 模板、action token 的放置位置、或 action chunk 的定义（`ACTION_DIM/NUM_ACTIONS_CHUNK`），这些切片与 mask 都需要重算。

2) **LLM 架构假设**
- LFP 只实现了对 `transformers` 的 Llama 结构做 layer 替换（`replace_llama_forward()`）与 SDPA attention patch（`replace_llama_spda_forward()`）。
- 换成 Mistral/Phi/Gemma 等，需要：
  - 重写对应 decoder layer 的“token 选择 + attention_mask 重建 + position_ids 处理”
  - 或者把“剪枝”下沉到你自己的 attention 实现里（不建议直接照搬 monkey-patch）。

3) **视觉编码器假设（Aggregation Tokens / FiLM）**
- Aggregation Tokens 这套实现是对 `timm.models.vision_transformer.VisionTransformer` 做 monkey-patch（`vit_wrapper_reg.py`），并假设存在 `patch_embed/_pos_embed/blocks/norm_pre` 等接口。
- 如果你的视觉 backbone 不是 timm ViT（例如 ConvNext、SAM、或 HF 的 ViT 实现），需要重写 aggregation token 的注入点与输出提取逻辑。

4) **Fused Backbone / 输入通道假设**
- 多图输入在 `experiments/robot/openvla_utils.py:get_vla_action()` 里通过 `pixel_values` 在 channel 维拼接实现；fused backbone 场景下每张图默认 6 通道（`3 + 3`）再在 wrapper 内 split。
- 如果你没有“SigLIP + DINOv2 双编码器”这种 fused 结构，`vision_aggregate_type='moe'` 也就没有意义（需要改成多专家或单专家逻辑）。

5) **保存/加载契约（决定推理能否复现训练模型）**
- 如果你沿用本仓的“离线合并”方式：最终 `save_pretrained()` 的权重不包含 vision wrapper，必须额外加载 `vision_backbone--*.pt`（`experiments/robot/openvla_utils.py:_apply_film_to_vla()`）。
- 如果你改成“训练阶段保存完整模型并直接推理加载”，则需要确保你的保存目录里也包含正确的 remote code（`auto_map` 指向正确类），并避免两套路径混用。
