# 🎓 CogVLA 训练全流程深度解析

本文档以**训练全流程**为主线，讲解CogVLA的四大核心优化模块如何在训练过程中协同工作。

---

## 📋 目录

1. [阶段一：启动与配置](#阶段一启动与配置)
2. [阶段二：模型初始化](#阶段二模型初始化)
3. [阶段三：LoRA注入与参数解冻](#阶段三lora注入与参数解冻)
4. [阶段四：Vision Backbone Wrapper注入](#阶段四vision-backbone-wrapper注入)
5. [阶段五：前向传播](#阶段五前向传播)
6. [阶段六：Checkpoint保存](#阶段六checkpoint保存)
7. [阶段七：推理加载](#阶段七推理加载)
8. [完整数据流总结](#完整数据流总结)

---

## 阶段一：启动与配置

### 1.1 命令行启动

```bash
torchrun --nproc_per_node=8 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --use_film True \
    --use_aggr True \
    --num_vision_aggr 64 \
    --use_lfp True \
    --lfp_enable_film True \
    --vision_aggregate_type moe \
    --dataset_name aloha_scoop_food
```

### 1.2 配置注入到模型

**关键代码位置**：`vla-scripts/finetune.py:931-942`

这是**关键第一步**：将CogVLA的4个模块开关写入配置

```python
# 加载base OpenVLA配置
vla_cfg = AutoConfig.from_pretrained(cfg.vla_path)

# ====== 模块3：LFP配置注入 ======
vla_cfg.text_config.use_lfp = True              # 启用LFP
vla_cfg.text_config.lfp_average_factor = 0.5    # 平均保留50%
vla_cfg.text_config.lfp_type = "shiftedcos_decay_0.85_0.15"
vla_cfg.text_config.lfp_enable_film = True      # LFP中启用FiLM

# ====== 模块2：Aggregation Tokens配置 ======
vla_cfg.use_aggr = True                         # 启用压缩
vla_cfg.use_film = True                         # 配合FiLM
vla_cfg.num_vision_aggr = 64                    # 压缩到64个token

# ====== 模块1：MoE Router配置 ======
vla_cfg.vision_aggregate_type = 'moe'           # MoE融合模式
```

**🔥 这里发生了什么？**

当调用`AutoModelForVision2Seq.from_pretrained(vla_path, config=vla_cfg)`时，模型会根据这些开关初始化对应组件。

---

## 阶段二：模型初始化

### 2.1 AutoModel加载触发组件创建

**关键代码位置**：`vla-scripts/finetune.py:946-952`

```python
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.vla_path,
    config=vla_cfg,  # 包含CogVLA配置
    trust_remote_code=True
)
```

**内部发生的事情**（在`prismatic/extern/hf/modeling_prismatic.py`的`__init__`中）：

```python
def __init__(self, config: PrismaticConfig):
    # 1️⃣ 如果config.text_config.use_lfp为True
    if config.text_config.use_lfp:
        replace_llama_forward()  # Monkey-patch LLM，注入LFP layers
        
    # 2️⃣ 创建base LLM（已被monkey-patch）
    self.language_model = AutoModelForCausalLM.from_config(config.text_config)
    
    # 3️⃣ 如果vision_aggregate_type == 'moe'
    if config.vision_aggregate_type == 'moe':
        # 创建MoE Router
        self.aggregation_router = MoEAggregator(
            num_experts=2,
            seq_dim=config.llm_backbone_config.hidden_size
        )
        # 为两个专家各创建投影层
        self.featurizer_proj = nn.Linear(...)      # SigLIP投影
        self.fused_featurizer_proj = nn.Linear(...)  # DINOv2投影
```

### 2.2 模型结构

**此时模型结构**：

```
vla
├── vision_backbone (原始：两个独立ViT)
│   ├── featurizer (SigLIP)
│   └── fused_featurizer (DINOv2)
├── aggregation_router (MoE Router) ← 新增
├── featurizer_proj ← 新增
├── fused_featurizer_proj ← 新增
└── language_model (Llama)
    ├── layers[0-31]
    │   ├── layers[5,10,15,20,25,30] → LlamaDecoderLFPLayer ← 被monkey-patch替换
    │   │   └── router: FiLMedTokenRouter ← 新增
    │   └── 其他层：标准LlamaDecoderLayer
    └── sdpa_attention → llama_sdpa_attention_forward ← 被全局monkey-patch
```

---

## 阶段三：LoRA注入与参数解冻

### 3.1 LoRA配置：排除CogVLA组件

**关键代码位置**：`vla-scripts/finetune.py:958-969`

```python
lora_config = LoraConfig(
    r=32,
    target_modules=find_all_linear_names(
        vla, 
        excluded_names=[
            'featurizer_proj',        # MoE组件
            'fused_featurizer_proj',  # MoE组件
            'aggregation_router',     # MoE组件
            'router',                 # LFP Router
            'scale', 'shift'          # FiLM components
        ]
    ),
)
vla = get_peft_model(vla, lora_config)
```

**🤔 为什么排除这些？**

LoRA只适用于原始的大型线性层（如attention、MLP）。CogVLA的新增组件：
- **MoE Router**：小型MLP，需要全量训练
- **LFP Router**：每层都有，需要全量训练
- **FiLM scale/shift**：动态调制参数，需要全量训练

### 3.2 显式解冻CogVLA组件

**关键代码位置**：`vla-scripts/finetune.py:971-984`

```python
# 解冻MoE Router及其投影层
if hasattr(vla.model, 'aggregation_router'):
    for n, p in vla.model.aggregation_router.named_parameters():
        p.requires_grad = True
    for n, p in vla.model.featurizer_proj.named_parameters():
        p.requires_grad = True
    for n, p in vla.model.fused_featurizer_proj.named_parameters():
        p.requires_grad = True

# 解冻所有LFP Routers
if vla.config.text_config.use_lfp:
    for n, p in vla.model.language_model.named_parameters():
        if 'router' in n:  # 匹配所有层的router
            p.requires_grad = True
```

### 3.3 训练参数分布

**此时训练参数分布**：

```
LoRA参数（梯度更新）:
  - language_model的attention q_proj, k_proj, v_proj, o_proj
  - language_model的MLP gate_proj, up_proj, down_proj
  
全量训练参数（梯度更新）:
  - aggregation_router: ~131K参数
  - featurizer_proj: ~4M参数
  - fused_featurizer_proj: ~5M参数
  - language_model.layers[*].router: ~32K × 6层 = ~192K参数
  
冻结参数（不更新）:
  - vision_backbone的所有参数
  - language_model的embeddings、layernorm等
```

---

## 阶段四：Vision Backbone Wrapper注入

> 📖 **详细说明**：关于Vision Backbone的完整执行流程（包括aggregation tokens和FiLM调制机制），请参考：[vision_backbone_flow.md](./vision_backbone_flow.md)

### 4.1 运行时Wrapper注入

**关键代码位置**：`vla-scripts/finetune.py:990-1021`

```python
if cfg.use_aggr:
    # 选择wrapper类
    wrapper_class = (FiLMedPrismaticVisionBackboneAggregator  # use_film=True
                     if cfg.use_film 
                     else PrismaticVisionBackboneAggregator)  # use_film=False
    
    # ⚠️ 注意：这里用vla.model.而不是vla.
    # 因为vla已被LoRA包裹，必须修改内部base_model
    vla.model.vision_backbone = wrapper_class(
        vision_backbone=vla.model.vision_backbone,  # 原始backbone
        llm_dim=4096,
        num_vision_aggr=64
    )
```

### 4.2 Wrapper内部机制

**关键代码位置**：`prismatic/models/vit_wrapper_reg.py`

```python
class FiLMedPrismaticVisionBackboneAggregator:
    def __init__(self, vision_backbone, llm_dim, num_vision_aggr):
        self.vision_backbone = vision_backbone  # 保留原始
        
        # 1️⃣ 创建可训练的聚合tokens
        self.vision_aggr_featurizer = nn.Parameter(
            torch.randn(1, num_vision_aggr, embed_dim)  # [1,64,1024]
        )
        
        # 2️⃣ Monkey-patch ViT：注入aggregation tokens
        self._wrap_vit(vision_backbone.featurizer, self.vision_aggr_featurizer)
        
        if use_fused:
            self.vision_aggr_fused_featurizer = nn.Parameter(...)
            self._wrap_vit(vision_backbone.fused_featurizer, ...)
    
    def _wrap_vit(self, vit, vision_aggr):
        # A. 用FiLM wrapper包裹每个Transformer block
        for block in vit.blocks:
            block_wrapper = FiLMedVisionTransformerBlock(
                block=block,
                vision_dim=vit.num_features,
                llm_dim=self.llm_dim
            )
        
        # B. 替换ViT的forward方法
        vit.forward = partial(vit.get_intermediate_layers, vision_aggr=vision_aggr)
```

### 4.3 Vision Backbone结构变化

**现在Vision Backbone结构变成**：

```
vision_backbone (Wrapper)
├── vision_backbone (原始)
│   ├── featurizer (SigLIP ViT) ← 被monkey-patch
│   │   ├── blocks[0-26]  → 每层包裹了FiLMedBlock
│   │   └── forward → 改为输出aggr_tokens
│   └── fused_featurizer (DINOv2 ViT) ← 被monkey-patch
├── vision_aggr_featurizer [1,64,1024] ← 可训练参数
└── vision_aggr_fused_featurizer [1,64,1152] ← 可训练参数
```

---

## 阶段五：前向传播

### 5.1 完整数据流追踪

**假设输入**：
- 图像：3张（primary + 2 wrist）
- 文本："把苹果放到碗里"
- 动作：7步轨迹

### Step 1: 图像编码（带Aggregation Tokens）

> 💡 **详细机制说明**：关于vision backbone内部如何实现256→64 token压缩和FiLM调制，请参考：[vision_backbone_flow.md](./vision_backbone_flow.md)

**关键代码位置**：`prismatic/extern/hf/modeling_prismatic.py:_process_vision_features()`

```python
# 在modeling_prismatic.py:forward()中
pixel_values = batch["pixel_values"]  # [B, 3*6, 224, 224]
language_embeddings = self.get_input_embeddings()(input_ids)  # [B, 20, 4096]

# 调用被wrapper的vision backbone
patch_features = self.vision_backbone(pixel_values, language_embeddings)
# 内部流程:
#  1. 文本压缩: mean(dim=1) → [B, 4096]
#  2. 分离3张图: 每张6通道 → SigLIP(3ch) + DINOv2(3ch)
#  3. 每个ViT: 256 patches → 拼接64个aggr_tokens → Transformer → 输出64个tokens
#  4. FiLM调制: 每层用文本生成gamma/beta调制视觉特征
```

**输出格式**：`[(siglip1, dino1), (siglip2, dino2), (siglip3, dino3)]`
- 每个tuple: `([B,64,1024], [B,64,1152])`
- 3张图 × 2编码器 = 6组压缩后的视觉特征

### Step 2: MoE融合

**关键代码位置**：`prismatic/extern/hf/modeling_prismatic.py:_aggregate_patch_features()`

```python
# 回到modeling_prismatic.py:_process_vision_features()
all_image_embeds = []
for img_patches in patch_features:  # 遍历3张图
    # 调用聚合函数
    image_embeds = self._aggregate_patch_features(
        img_patches,  # (siglip_patches, dino_patches)
        language_embeddings
    )
    all_image_embeds.append(image_embeds)
```

**_aggregate_patch_features内部**：

```python
def _aggregate_patch_features(self, patch_features, language_embeddings):
    if self.config.vision_aggregate_type == 'moe':
        # 1. 提取两个专家的features
        patches_siglip, patches_dino = patch_features  # [B,64,1024], [B,64,1152]
        
        # 2. 投影到LLM维度
        proj_siglip = self.featurizer_proj(patches_siglip)      # [B,64,4096]
        proj_dino = self.fused_featurizer_proj(patches_dino)    # [B,64,4096]
        
        # 3. MoE Router决策
        avg_lang = language_embeddings.mean(dim=1)  # [B, 4096]
        fused = self.aggregation_router(
            [proj_siglip, proj_dino],  # 两个专家
            avg_lang                    # 文本condition
        )
        # Router内部：
        #   ratios = softmax(MLP(avg_lang))  # [B, 2] 例如：[0.7, 0.3]
        #   output = ratios[0] * proj_siglip + ratios[1] * proj_dino
        
        return fused  # [B, 64, 4096]
```

**拼接3张图的结果**：

```python
image_embeds = torch.cat(all_image_embeds, dim=1)  # [B, 192, 4096] (64*3)
```

### Step 3: 构建多模态序列

```python
# 文本embedding
input_embeddings = self.get_input_embeddings()(input_ids)  # [B, seq_len, 4096]

# 插入视觉tokens到BOS后面
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],      # [BOS]
    image_embeds,                     # 192个视觉tokens
    input_embeddings[:, 1:, :]       # 文本 + 动作tokens
], dim=1)  # [B, 1+192+20+43, 4096] = [B, 256, 4096]
```

**序列布局**：

```
[BOS] [V1...V192] [拿起苹果放到碗里] [A1...A43_STOP]
  ↑      ↑              ↑                    ↑
 token  视觉           文本指令            动作序列
  0     1-192         193-212           213-255
```

### Step 4: LLM处理（带LFP剪枝）

> 📖 **详细机制说明**：关于LFP剪枝的完整机制（打分、选择、压缩、恢复流程及加速原理），请参考：[LFP_mechanism.md](./LFP_mechanism.md)

**关键代码位置**：`prismatic/models/modeling_llama.py:LlamaDecoderLFPLayer.forward()`

```python
outputs = self.language_model(
    inputs_embeds=multimodal_embeddings,
    attention_mask=attention_mask,
    ...
)
```

**在LLM内部**（经过monkey-patch）：

```python
# Layer 0-4: 标准LlamaDecoderLayer
hidden = standard_layers[0-4](hidden)  # 所有256个tokens

# Layer 5: LlamaDecoderLFPLayer（第1个剪枝层）
def forward(self, hidden_states):  # [B, 256, 4096]
    # 1. Router打分
    if self.config.lfp_enable_film:
        # FiLMed Router：用文本调制视觉
        router_logits = self.router(hidden_states, attn_mask, num_vision=192)
    else:
        router_logits = self.router(hidden_states)  # [B, 256, 2]
    
    keep_probs = softmax(router_logits)[:, :, 1]  # [B, 256]
    
    # 2. 强制保留非视觉tokens
    force_mask = torch.zeros_like(keep_probs)
    force_mask[:, 0] = inf              # 保留BOS
    force_mask[:, 193:] = inf           # 保留文本+动作
    
    # 3. Top-K选择
    router_factor = shifted_cos(layer_idx=5)  # 例如0.85
    keep_len = 1 + int(192*0.85) + 20 + 43 = 227
    
    _, indices = topk(keep_probs + force_mask, k=227)
    # indices例如：[0, 1, 3, 7, ..., 150, 193, 194, ..., 255]
    # 丢弃了约30个不重要的视觉tokens
    
    # 4. Gather保留的tokens
    kept_hidden = gather(hidden_states, indices)  # [B, 227, 4096]
    
    # 5. 重建attention mask（关键！）
    kept_mask = gather(gather(attn_mask, indices, dim=2), indices, dim=3)
    
    # 6. 标准transformer层
    output = super().forward(kept_hidden, kept_mask, ...)
    
    # 7. Scatter回原位置
    hidden_states = scatter(hidden_states, indices, output)  # [B, 256, 4096]
    
    return hidden_states

# Layer 6-9: 标准层
# Layer 10: LFP Layer (router_factor=0.75) → 保留144个视觉
# ...
# Layer 30: LFP Layer (router_factor=0.20) → 保留38个视觉
# Layer 31: 标准层
```

**剪枝进度可视化**：

```
Layer 0:  [BOS] + 192视觉 + 20文本 + 43动作 = 256 tokens
Layer 5:  [BOS] + 163视觉 + 20文本 + 43动作 = 227 tokens (丢29个)
Layer 10: [BOS] + 144视觉 + 20文本 + 43动作 = 208 tokens (丢48个)
Layer 15: [BOS] + 115视觉 + 20文本 + 43动作 = 179 tokens (丢77个)
Layer 20: [BOS] + 77视觉  + 20文本 + 43动作 = 141 tokens (丢115个)
Layer 25: [BOS] + 57视觉  + 20文本 + 43动作 = 121 tokens (丢135个)
Layer 30: [BOS] + 38视觉  + 20文本 + 43动作 = 102 tokens (丢154个)
```

### Step 5: Parallel Action Chunking（SDPA层）

**关键代码位置**：`prismatic/models/modeling_llama.py:llama_sdpa_attention_forward()`

在每个attention层中：

```python
# llama_sdpa_attention_forward（被全局monkey-patch）

# 标准QKV计算
Q, K, V = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)

# 构建causal mask
causal_mask = ... # 标准下三角mask [B, 1, seq, seq]

# ========== CATTEN修改 ==========
# 找到action tokens区域并清零mask
num_act = 43
for idx in range(batch_size):
    # 清零最后43个tokens之间的mask
    causal_mask[idx, :, -43:, -43:] = 0

# 标准的位置：
#   A1只能看: [BOS, V, T, A1]
#   A2只能看: [BOS, V, T, A1, A2]
# 修改后：
#   A1能看: [BOS, V, T, A1, A2, ..., A43]  ← 看到所有动作
#   A2能看: [BOS, V, T, A1, A2, ..., A43]  ← 同样
# ================================

attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
```

**效果对比**：

```
原始attention pattern:
  BOS  V1  ...  T20  A1  A2  A3
A1  1   1   ...  1   1   0   0
A2  1   1   ...  1   1   1   0
A3  1   1   ...  1   1   1   1

修改后（Parallel Action):
  BOS  V1  ...  T20  A1  A2  A3
A1  1   1   ...  1   1   1   1  ← 能看到A2,A3!
A2  1   1   ...  1   1   1   1  ← 三个动作互相可见
A3  1   1   ...  1   1   1   1
```

### Step 6: 损失计算

```python
# 获取最后一层hidden states
last_hidden = outputs.hidden_states[-1]  # [B, 256, 4096]

# 提取动作部分 (indices 213-255)
action_hidden = last_hidden[:, 213:256, :]  # [B, 43, 4096]

# 通过LM head预测
logits = self.lm_head(last_hidden)  # [B, 256, vocab_size]

# 计算交叉熵
loss = cross_entropy(logits, labels, ignore_index=-100)
```

---

## 阶段六：Checkpoint保存

### 6.1 保存策略的关键设计

**关键代码位置**：`vla-scripts/finetune.py:631-753`

```python
def save_training_checkpoint(...):
    checkpoint_dir = run_dir / f"step_{log_step}"
    adapter_dir = checkpoint_dir / "lora_adapter"
    
    # 1️⃣ 保存LoRA adapter (总是)
    vla.module.save_pretrained(adapter_dir)
    
    # 2️⃣ 保存Vision Backbone (如果use_aggr或use_film)
    if vla_config.use_film or vla_config.use_aggr:
        torch.save(
            vla.module.vision_backbone.state_dict(),
            checkpoint_dir / "vision_backbone--checkpoint.pt"
        )
    
    # 3️⃣ 保存非LoRA可训练参数
    if cfg.merge_lora_during_training == False:
        non_lora_trainables = get_peft_state_non_lora(
            vla.named_parameters(),
            excluded_names=['vision_backbone']
        )
        # 包含：
        #  - aggregation_router
        #  - featurizer_proj
        #  - fused_featurizer_proj
        #  - language_model.layers[*].router
        
        torch.save(
            non_lora_trainables,
            checkpoint_dir / "non_lora_trainables--checkpoint.pt"
        )
    
    # 4️⃣ 保存数据集统计信息
    save_dataset_statistics(
        train_dataset.dataset_statistics,
        checkpoint_dir / "dataset_statistics.json"
    )
```

### 6.2 Checkpoint目录结构

**保存后的目录结构**：

```
runs/step_10000/
├── lora_adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors  # LoRA权重
├── vision_backbone--checkpoint.pt   # Wrapper + aggr_tokens
├── non_lora_trainables--checkpoint.pt  # MoE Router + LFP Routers
├── dataset_statistics.json
├── config.json
├── preprocessor_config.json
└── tokenizer.json
```

### 6.3 为什么这样设计？

**问题**：LoRA不包含全部训练参数

```
保存的LoRA adapter只包含：
  ✓ attention的q_proj, k_proj, v_proj, o_proj的LoRA矩阵
  ✓ MLP的gate_proj, up_proj, down_proj的LoRA矩阵

缺失：
  ✗ aggregation_router (全量训练)
  ✗ featurizer_proj/fused_featurizer_proj (全量训练)
  ✗ language_model.layers[*].router (全量训练)
  ✗ vision_backbone的wrapper和aggr_tokens
```

**解决方案**：
1. `vision_backbone--*.pt` 单独保存（因为wrapper不在base model里）
2. `non_lora_trainables--*.pt` 单独保存（MoE + LFP routers）
3. 推理时需要三步加载（见下一阶段）

---

## 阶段七：推理加载

### 7.1 完整加载流程

**关键代码位置**：`experiments/robot/openvla_utils.py:get_vla()`

```python
def get_vla(cfg):
    # Step 1: 加载merged model（base + LoRA merged）
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True
    )
    # 此时有：base model + merged LoRA
    # 缺少：vision wrapper, MoE router, LFP routers
    
    # Step 2: 如果use_aggr或use_film，重建wrapper
    if vla.config.use_film or vla.config.use_aggr:
        # 2.1 创建wrapper
        if vla.config.use_aggr:
            wrapper_class = (FiLMedPrismaticVisionBackboneAggregator
                           if vla.config.use_film
                           else PrismaticVisionBackboneAggregator)
            vla.vision_backbone = wrapper_class(
                vision_backbone=vla.vision_backbone,
                llm_dim=vla.llm_dim,
                num_vision_aggr=vla.config.num_vision_aggr
            )
        
        # 2.2 加载保存的wrapper权重
        vision_state = torch.load(
            checkpoint_dir / "vision_backbone--checkpoint.pt"
        )
        vla.vision_backbone.load_state_dict(vision_state)
    
    # Step 3: 加载dataset statistics（用于动作unnormalize）
    stats_path = checkpoint_dir / "dataset_statistics.json"
    vla.norm_stats = json.load(open(stats_path))
    
    return vla
```

### 7.2 推理前向传播

```python
def get_vla_action(vla, image, instruction):
    # 1. 准备输入
    prompt = f"In: {instruction}?\nOut:"
    inputs = processor(prompt, image)
    
    # 2. VLA推理
    with torch.no_grad():
        output = vla.predict_action(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            unnorm_key='aloha_scoop_food'  # 用于unnormalize
        )
    
    # 3. Unnormalize动作
    normalized_actions = output  # [-1, 1]
    actions = vla._unnormalize_actions(
        normalized_actions,
        unnorm_key='aloha_scoop_food'
    )
    # 使用dataset_statistics.json中的min/max还原
    
    return actions  # 真实机器人动作空间
```

---

## 完整数据流总结

### 训练时的完整前向传播

```
图像 [B,3*6,224,224]
  ↓
Vision Backbone (Wrapper)
  ├→ ViT + Aggr Tokens → [B,64,1024] (SigLIP)
  └→ ViT + Aggr Tokens → [B,64,1152] (DINOv2)
  ↓
MoE Router (用文本condition)
  Router决策: 0.7*SigLIP + 0.3*DINOv2
  ↓
拼接3张图 → [B,192,4096]
  ↓
构建多模态序列
  [BOS] + 192视觉 + 20文本 + 43动作 = 256 tokens
  ↓
LLM Layers (32层)
  ├→ Layer 0-4: 标准层 (256 tokens)
  ├→ Layer 5: LFP (剪到227 tokens)
  ├→ Layer 10: LFP (剪到208 tokens)
  ├→ Layer 15: LFP (剪到179 tokens)
  ├→ Layer 20: LFP (剪到141 tokens)
  ├→ Layer 25: LFP (剪到121 tokens)
  ├→ Layer 30: LFP (剪到102 tokens)
  └→ Layer 31: 标准层
     ↓ (每层的SDPA都用了Parallel Action Chunking)
  ↓
LM Head → Logits [B,256,vocab_size]
  ↓
Cross Entropy Loss
  ↓
Backward (更新LoRA + MoE + LFP + Aggr Tokens)
```

### 关键设计权衡总结

| 阶段 | 设计点 | 原因 |
|------|--------|------|
| **配置注入** | 用`vla_cfg`传递开关 | 让AutoModel自动初始化对应组件 |
| **LoRA排除** | 排除MoE/LFP/FiLM | 这些小模块需要全量训练，LoRA不适合 |
| **Wrapper运行时注入** | 训练前包裹backbone | base权重已freeze，只能在外面加wrapper |
| **Vision分开保存** | `vision_backbone--*.pt` | Wrapper不在base model里，merge时漏掉 |
| **三步加载** | merged + wrapper + stats | 兼容离线merge workflow |

---

## 💡 实践建议

### 1. 调试策略

从简单到复杂，逐步启用优化：

```bash
# Step 1: 基础训练（确保流程能跑通）
--use_film False --use_aggr False --use_lfp False

# Step 2: 添加Aggregation Tokens（最显著的优化）
--use_aggr True --num_vision_aggr 64

# Step 3: 添加FiLM（提升特征质量）
--use_film True

# Step 4: 添加MoE Router（智能融合）
--vision_aggregate_type moe

# Step 5: 添加LFP（推理加速）
--use_lfp True --lfp_enable_film True
```

### 2. 监控指标

在训练过程中监控这些关键指标：

- **MoE Router权重分布**：观察SigLIP vs DINOv2的权重变化
- **LFP保留率**：每层实际保留的视觉token比例
- **Aggregation Tokens梯度**：确保聚合token在学习
- **训练速度**：对比启用前后的iteration time

### 3. 内存优化顺序

如果遇到OOM，按照这个顺序启用优化：

1. **先用Aggregation Tokens**（减少序列长度，效果最显著）
2. **再加LFP**（进一步减少计算量）
3. **最后考虑MoE Router**（略微增加参数量，但提升质量）

### 4. 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 推理时找不到`vision_backbone--*.pt` | 训练时没保存wrapper | 检查`use_aggr`或`use_film`是否为True |
| `non_lora_trainables`加载失败 | 离线merge时漏加载 | 确保merge脚本加载了这个文件 |
| LFP报错不支持flash-attn-2 | LFP与flash-attn冲突 | 设置`attn_implementation='sdpa'` |
| MoE Router权重全是NaN | 学习率过大 | 降低学习率或增加warmup steps |

---

## 📚 相关文档

- [LFP_mechanism.md](./LFP_mechanism.md) - LFP剪枝机制完整详解（打分、选择、压缩、加速原理）
- [vision_backbone_flow.md](./vision_backbone_flow.md) - Vision Backbone执行流程（Aggregation Tokens与FiLM调制）
- [vision_language_interaction_flow.md](./vision_language_interaction_flow.md) - 视觉-语言交互机制详解
- [CogVLA_INTEGRATION.md](./CogVLA_INTEGRATION.md) - 模块实现细节和集成指南
- [训练脚本示例](../scripts-sh/finetune.sh) - 完整的训练命令参考
- [推理脚本示例](../scripts-sh/eval_aloha_deploy.sh) - 部署和推理示例

---

**本文档版本**：2025-12-23  
**适用CogVLA版本**：基于OpenVLA-7B的实现
