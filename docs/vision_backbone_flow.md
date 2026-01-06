# Vision Backbone 执行流程完整梳理

> 从输入到输出的完整数据流追踪

---

## 🎯 一句话总结

**输入**: 图像(18通道) + 文本指令 → **输出**: 3张图 × 2个编码器 = 6组压缩特征

---

## 📥 输入准备

```python
# 输入数据
pixel_values:        [B, 18, 224, 224]  # 3张图 × 6通道
language_embeddings: [B, 20, 4096]       # 文本指令，20个token

# B = batch size (例如8)
# 18通道 = 3张图 × (SigLIP 3通道 + DINOv2 3通道)
```

---

## 🔄 执行流程（5个步骤）

### **步骤1: 文本压缩**

```python
# 在 FiLMedPrismaticVisionBackboneAggregator.forward()
avg_lang = language_embeddings.mean(dim=1)  # [B, 20, 4096] → [B, 4096]
```

**作用**: 将20个token的指令压缩成1个全局语义向量

---

### **步骤2: 分离图像**

```python
# 分离3张图（每张6通道）
images = torch.split(pixel_values, [6, 6, 6], dim=1)
# → img1[B,6,224,224], img2[B,6,224,224], img3[B,6,224,224]
```

**3张图通常是**:
- img1: Primary camera（主视角）
- img2: Wrist camera 1（手腕相机1）
- img3: Wrist camera 2（手腕相机2）

---

### **步骤3: 循环处理每张图**

```python
for img in images:  # 处理3次
    # 3.1 分离双编码器输入
    img_regular, img_fused = torch.split(img, [3, 3], dim=1)
    # img_regular: [B, 3, 224, 224]  ← 给SigLIP
    # img_fused:   [B, 3, 224, 224]  ← 给DINOv2
    # 注意: 是同一张RGB图，复制了两份！
    
    # 3.2 通过双编码器
    patches_siglip = self.vision_backbone.featurizer(img_regular, avg_lang)
    patches_dino = self.vision_backbone.fused_featurizer(img_fused, avg_lang)
```

**双编码器**:
- **SigLIP**: 擅长语义识别（"这是苹果"）
- **DINOv2**: 擅长几何定位（"在左边"）

---

### **步骤4: 单个ViT内部处理**（关键！）

以`featurizer`（SigLIP）为例：

```python
# 4.1 调用被替换的forward
featurizer(img_regular, avg_lang)
  ↓ (forward已被monkey-patch)
  ↓
# 4.2 实际执行 get_intermediate_layers
get_intermediate_layers(img_regular, avg_lang, vision_aggr=[1,64,1024], n=25)
  ↓
# 4.3 内部调用 _intermediate_layers
_intermediate_layers():
    x = patch_embed(img_regular)              # [B, 256, 1024] - 图像分patch
    x = cat([x, vision_aggr_tokens])          # [B, 320, 1024] - 拼接64个聚合token
    
    # 4.4 通过27层Transformer
    for blk in blocks (0-26):
        # 每层都做FiLM调制
        gamma = MLP(avg_lang)  # [B, 1024] 缩放系数
        beta = MLP(avg_lang)   # [B, 1024] 偏移系数
        
        x = attention(x)       # 320个token互相看
        x = x * (1+gamma) + beta  # 🔥 文本调制视觉！
        x = mlp(x)
  ↓
# 4.5 提取aggregation tokens
output = x[:, 256:]  # [B, 320, 1024] → [B, 64, 1024]
                     # 丢弃前256个patch，只保留64个聚合token
```

**关键机制**:
- **Aggregation Tokens**: 256个patch压缩到64个token
- **FiLM调制**: 用文本`avg_lang`动态调整视觉特征

---

### **步骤5: 收集输出**

```python
# 处理完3张图后
all_patches = [
    (siglip1[B,64,1024], dino1[B,64,1152]),  # 图1
    (siglip2[B,64,1024], dino2[B,64,1152]),  # 图2
    (siglip3[B,64,1024], dino3[B,64,1152])   # 图3
]
return all_patches
```

---

## 📊 数据流图解

```
pixel_values [B,18,224,224] + language_embeddings [B,20,4096]
                    ↓
            avg_lang = mean()  [B,4096]
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      img1        img2        img3
    [B,6,224]   [B,6,224]   [B,6,224]
        ↓           ↓           ↓
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
SigLIP DINOv2  SigLIP DINOv2  SigLIP DINOv2
    ↓     ↓       ↓     ↓       ↓     ↓
  [64] [64]     [64] [64]     [64] [64]  ← aggregation tokens
    └───┬───┘   └───┬───┘   └───┬───┘
        ↓           ↓           ↓
     输出1        输出2        输出3
```

---

## 🎯 核心要点

### 1️⃣ **双编码器架构**
- 同一张图送入两个编码器
- SigLIP (语义) + DINOv2 (几何) = 互补特征

### 2️⃣ **Aggregation Tokens**
- 每个ViT: 256 patches → 64 aggregation tokens
- 压缩比: 75% (大幅减少后续LLM计算量)

### 3️⃣ **FiLM调制**
- 文本指令动态影响视觉编码
- 公式: `x_new = x * (1 + γ(text)) + β(text)`
- 不同指令 → 不同视觉表示

### 4️⃣ **输出格式**
- 3张图 × 2个编码器 = 6组特征
- 每组: [B, 64, embed_dim]
- 后续通过MoE Router融合

---

## 💻 核心代码实现详解

### **机制1: 256 Patches → 64 Aggregation Tokens**

**代码位置**: `prismatic/models/vit_wrapper_reg.py:_intermediate_layers`

```python
# 步骤1: 标准patch embedding
x = self.patch_embed(x)      # [B, 3, 224, 224] → [B, 256, 1024]
x = self._pos_embed(x)        # 添加位置编码
x = self.norm_pre(x)

# 🔥 步骤2: 拼接aggregation tokens (关键!)
vision_aggr_batch = vision_aggr.expand(x.shape[0], -1, -1)  
# vision_aggr:       [1, 64, 1024]  (可学习参数)
# vision_aggr_batch: [B, 64, 1024]  (expand到batch size)

x = torch.cat([x, vision_aggr_batch], dim=1)
# 拼接结果: [B, 320, 1024]
#          [256个patches + 64个aggregation tokens]

# 步骤3: 通过Transformer交互
for i, blk in enumerate(self.blocks):  # 27层
    x = blk(x, language_embeddings)    
    # 所有320个token一起做self-attention
    # aggregation tokens通过attention "吸收" patch信息

# 🔥 步骤4: 只保留aggregation tokens (在get_intermediate_layers中)
outputs = [out[:, 256:] for out in outputs]  # [B, 320, 1024] → [B, 64, 1024]
#              ↑ 跳过前256个patches，只保留后64个aggregation tokens
```

**核心原理**: 
- Aggregation tokens通过27层的self-attention，不断与256个patch交互
- 最终"学会"如何总结整张图像的信息
- 输出时丢弃patches，只保留aggregation tokens → 实现75%压缩

---

### **机制2: FiLM调制公式实现**

**代码位置**: `prismatic/models/film_vit_wrapper.py:FiLMedVisionTransformerBlock.forward`

```python
def forward(self, x, average_language_embedding):
    # 🔥 步骤1: 从文本生成 gamma 和 beta
    gamma = self.scale(average_language_embedding)  # [B, 4096] → [B, 1024]
    beta = self.shift(average_language_embedding)   # [B, 4096] → [B, 1024]
    #        ↑ Linear层                ↑ 文本全局向量
    
    # 步骤2: Attention
    x = x + self.block.attn(self.block.norm1(x))
    
    # 🔥 步骤3: FiLM调制 (L72)
    x = x * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(...)
    #   ↑     ↑ 缩放                                              ↑ 偏移
    # 原始   1+γ (初始化接近identity)                           β
    
    # 步骤4: MLP
    x = x + self.block.mlp(self.block.norm2(x))
    return x
```

**展开说明**:
```python
# gamma, beta shape: [B, 1024]
# x shape: [B, 320, 1024]  (320个token)

# 广播机制:
gamma_expanded = gamma.view(B, 1, 1024)  # [B, 1, 1024]
beta_expanded = beta.view(B, 1, 1024)    # [B, 1, 1024]

# 对每个token的每个维度做:
x[b,i,d] = x[b,i,d] * (1 + gamma[b,d]) + beta[b,d]
#          原始值      文本决定的缩放    文本决定的偏移
```

**作用示例**:
```python
# 指令: "pick up the red apple"
gamma = [0.5, -0.2, 0.8, ...]  # 强调某些维度
beta = [0.1, 0.05, -0.1, ...]  # 调整基线

# 对视觉特征 x = [2.0, 1.5, 0.3, ...]
x_new[0] = 2.0 * (1+0.5) + 0.1 = 3.1   # 放大
x_new[1] = 1.5 * (1-0.2) + 0.05 = 1.25 # 缩小
x_new[2] = 0.3 * (1+0.8) - 0.1 = 0.44  # 调整

# 结果: 文本"告诉"视觉编码器哪些特征更重要
```

---

## 🔍 常见疑问

**Q: 为什么6通道？**  
A: 同一张RGB图复制两份，分别给SigLIP(前3通道)和DINOv2(后3通道)

**Q: aggregation tokens如何工作？**  
A: 通过self-attention与256个patch交互，"吸收"全图信息，最终只保留这64个token

**Q: FiLM在哪里作用？**  
A: 在ViT的每一层，attention之后、MLP之前 (L72: `x = x * (1+γ) + β`)

**Q: 为什么要mean(dim=1)？**  
A: 把变长指令(20个token)压缩成固定长度(1个向量)，便于FiLM使用

**Q: Batch size B是什么？**  
A: 训练时同时处理的样本数量。mean(dim=1)后每个样本都有自己的全局语义向量

---

## 📍 后续处理：MoE Router（Vision Backbone外部）

> ⚠️ **重要**：MoE Router **不属于** Vision Backbone 内部，但它是 Vision Backbone 输出的**下一个处理步骤**

### 架构位置

```python
# modeling_prismatic.py
class PrismaticForConditionalGeneration:
    def __init__(self):
        # 1️⃣ Vision Backbone（独立组件）
        self.vision_backbone = PrismaticVisionBackbone(...)
        
        # 2️⃣ MoE Router（模型顶层，与vision_backbone平级）
        self.aggregation_router = MoEAggregator(...)
        self.featurizer_proj = nn.Linear(...)       # SigLIP投影
        self.fused_featurizer_proj = nn.Linear(...) # DINO投影
```

**处理流程**：
```
Vision Backbone → 投影层 → MoE Router → 多模态序列
  (内部组件)    (线性变换)  (外部组件)   (送入LLM)
```

---

### MoE Router 工作机制

**输入**（来自Vision Backbone）：
```python
patches_siglip: [B, 64, 1024]  # SigLIP输出（已被FiLM调制）
patches_dino:   [B, 64, 1152]  # DINO输出（已被FiLM调制）
avg_lang:       [B, 4096]       # 文本全局向量
```

**步骤1：投影到统一维度**
```python
proj_siglip = self.featurizer_proj(patches_siglip)      # [B, 64, 4096]
proj_dino = self.fused_featurizer_proj(patches_dino)    # [B, 64, 4096]
```

**步骤2：MoE Router 决定融合权重**
```python
# 代码位置: prismatic/models/router.py
def forward(self, inputs_embeds, seq_embeds):
    # inputs_embeds = [proj_siglip, proj_dino]
    # seq_embeds = avg_lang
    
    # 文本 → 权重
    logits = self.router(seq_embeds)  # [B, 4096] → [B, 2]
    # router = MLP: Linear(4096→4096) → GELU → Linear(4096→2)
    
    ratios = torch.softmax(logits, dim=-1)  # [B, 2]
    # 例如: [0.7, 0.3]
    
    # 加权求和
    output = ratios[:, 0].view(-1,1,1) * proj_siglip + \
             ratios[:, 1].view(-1,1,1) * proj_dino
    # [B, 64, 4096]
    
    return output
```

**输出**：
```python
fused_features: [B, 64, 4096]  # 融合后的视觉tokens
```

---

### 🔄 MoE Router vs FiLM 对比

| 维度 | FiLM调制 | MoE Router |
|------|----------|------------|
| **位置** | Vision Backbone内部 | Vision Backbone外部 |
| **层级** | ViT每层（27层） | 1次全局操作 |
| **交互深度** | 深度（修改特征值） | 浅层（加权平均） |
| **粒度** | Per-dimension（1024维） | Per-sample（2个标量） |
| **机制** | `x = x*(1+γ)+β` | `out = w1*x1 + w2*x2` |
| **信息流** | 文本→视觉特征 | 文本→权重→视觉 |
| **作用** | 调制特征内容 | 选择专家权重 |

---

### 📊 完整数据流（含MoE）

```
pixel_values [B,18,224,224] + language_embeddings [B,20,4096]
                    ↓
            avg_lang = mean()  [B,4096]
                    ↓
        ┌───────────┼───────────┐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Vision Backbone 内部处理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ↓           ↓           ↓
      img1        img2        img3
    [B,6,224]   [B,6,224]   [B,6,224]
        ↓           ↓           ↓
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
🔥 FiLM调制（27层，深度交互）
    ↓     ↓       ↓     ↓       ↓     ↓
🎯 Aggregation（Self-Attention压缩）
    ↓     ↓       ↓     ↓       ↓     ↓
SigLIP DINOv2  SigLIP DINOv2  SigLIP DINOv2
  [64] [64]     [64] [64]     [64] [64]
    └───┬───┘   └───┬───┘   └───┬───┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Vision Backbone 输出
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ↓           ↓           ↓
    投影层（线性变换）
        ↓           ↓           ↓
    [B,64,4096] [B,64,4096] [B,64,4096]
        ↓           ↓           ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎚️ MoE Router（条件化加权）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   0.7*SigLIP + 0.3*DINO (每张图)
        ↓           ↓           ↓
    [B,64,4096] [B,64,4096] [B,64,4096]
        └───────────┴───────────┘
                    ↓
        concatenate (拼接3张图)
                    ↓
            [B, 192, 4096]
                    ↓
        送入 LLM 构建多模态序列
```

---

### 💡 关键理解

1. **架构分层**：
   - **Vision Backbone**：负责特征提取（双编码器+FiLM+Aggregation）
   - **MoE Router**：负责专家融合（在Backbone外部）

2. **交互层次**：
   - **FiLM**（深层）：文本细粒度调制视觉特征（1024维）
   - **MoE**（浅层）：文本决定专家权重（2个标量）

3. **设计哲学**：
   - FiLM: "什么特征重要？"（What）
   - MoE: "用哪个专家？"（Which）

4. **没有Cross-Attention**：
   - MoE Router 只是简单的加权平均
   - 不是注意力机制，只是条件化加权

---

### 🔗 相关文档

- 完整的视觉-语言交互流程：[vision_language_interaction_flow.md](./vision_language_interaction_flow.md)
- CogVLA训练流程：[CogVLA_TRAINING_FLOW.md](./CogVLA_TRAINING_FLOW.md)
