# 第 5 章：编码器、解码器与残差结构

对应代码：`nmt/layers.py`、`nmt/encoder.py`、`nmt/decoder.py`、`nmt/model.py`

## 三个"零件"

Transformer 的层看起来复杂，其实是三种零件反复堆叠：

### 1. 残差连接：`x = x + f(x)`

为什么必须有？深网络里梯度要一层层往回传，中间任何一处把梯度乘小，后面的层就学不动。
`x = x + f(x)` 给梯度留了一条**恒等高速公路**：即使 `f` 的梯度很小，`∂(x + f(x))/∂x` 里始终有一个 1。

这也是后面的 LayerNorm 位置（pre/post-norm）之争的根源：norm 放在残差**里面**还是**外面**，直接决定这条高速公路通不通畅。

### 2. 层归一化：LayerNorm

对**单个样本的特征维度**做归一化（减均值、除标准差），和 batch size 无关。

* 不需要像 BatchNorm 那样维护滑动统计量，训练和推理行为一致；
* 天然适合变长序列和任意 batch size。

论文的 `eps=1e-6`，本项目保持一致（PyTorch 的 `nn.LayerNorm` 默认是 `1e-5`，做对比实验时要注意这个差异 —— 我们的 `verify.py` 专门为此传了 `layer_norm_eps=1e-6`）。

### 3. 前馈网络：`FFN(x) = W₂·act(W₁·x + b₁) + b₂`

对每个位置**独立**作用，位置之间不交流（交流全靠注意力）。中间维度 `d_ff` 通常是 `4 × d_model`。

它在做什么？注意力负责"从别的位置取信息"，FFN 负责"在当前位置把信息加工一遍"。
一种流行的理解是：FFN 相当于一个 key-value 记忆库，`W₁` 的行是"模式匹配器"，`W₂` 的列是"写回的内容"。

**参数大头在这**：base 模型每层约 2/3 的参数在 FFN 里（8/3·d² vs 4·d²）。

## 编码器：两层结构，堆 6 次

```
x → 自注意力（每个词看整句）→ 残差+Norm
  → 前馈网络（逐位置加工）  → 残差+Norm
```

输出叫 **memory**，形状 `[B, S, d_model]`。
它已经不再是"每个词孤立的词向量"，而是"每个词 + 上下文"的表示。

## 解码器：三层结构，多一个交叉注意力

```
x → 因果自注意力（只能看已经写出来的部分）  → 残差+Norm
→ 交叉注意力（Q 来自德文，K/V 来自英文 memory）→ 残差+Norm
  → 前馈网络                                → 残差+Norm
```

**交叉注意力就是"翻译"发生的地方**：每写一个德语词，它都在问"英文原句里哪几个词跟我现在写的这个词有关"。

三种注意力的 K/V 来源，记这张表就够了：

| 位置 | Query 来自 | Key / Value 来自 | 掩码 |
| --- | --- | --- | --- |
| 编码器自注意力 | 英文 | 英文 | padding |
| 解码器自注意力 | 德文 | 德文 | padding + 因果 |
| 解码器交叉注意力 | 德文 | **英文 memory** | padding（英文侧） |

## pre-norm 还是 post-norm

```
post-norm（原论文）：x = LayerNorm(x + Dropout(Sublayer(x)))
pre-norm（现代模型）：x = x + Dropout(Sublayer(LayerNorm(x)))   ← 堆叠末尾再补一次 LayerNorm
```

两者的差别，说到底是"归一化挡在残差高速公路上，还是站在它的支路里"。

| | post-norm | pre-norm |
| --- | --- | --- |
| 训练稳定性 | 需要 warmup，层数一多容易崩 | 稳，几十层也能直接训 |
| 效果上限 | 同规模下通常略好 | 略差，但能靠更深补回来 |
| 现在的地位 | 论文原版、教学首选 | 所有大模型的默认选择 |

本项目两种都支持，配置里 `norm_first=True` 就是 pre-norm。**建议你两个都训一遍做对比**，
这是理解"训练稳定性"最直观的一次实验。

## 组装：一次前向的完整形状变化

以 `d_model=64`、词表 300、英文 7 个 token、德文 5 个 token 为例：

```
src [1, 7]                          tgt [1, 5]
  │ 词嵌入 + 位置编码                  │
  ▼                                  ▼
[1, 7, 64]                        [1, 5, 64]
  │ 编码器 × 2 层                     │ 解码器 × 2 层
  ▼                                  │
memory [1, 7, 64] ──────────────────►│ (交叉注意力反复查询它)
                                     ▼
                                  [1, 5, 64]
                                     │ 输出层 Linear(64 → 300)
                                     ▼
                              logits [1, 5, 300]
```

用 `python -m nmt.inspect --shapes` 可以把每个子模块的输入输出形状全打出来，
比看图更直观。

## 自己动手

```bash
# 1. 打印模型结构和参数量分布
python -c "
from nmt.config import base_config
from nmt.model import Transformer
print(Transformer(base_config().model).describe())
"

# 2. 打一层完整前向的形状追踪
python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes \
    --text "he will finish the project tomorrow."

# 3. pre-norm / post-norm 对比实验
python -m nmt.train --preset small --override model.norm_first=false
python -m nmt.train --preset small --override model.norm_first=true
#    看前 2000 步的 loss 曲线：post-norm 在 warmup 不足时会出现尖峰

# 4. 层数消融：1 层、3 层、6 层的 BLEU 差多少？
python -m nmt.train --preset base --override model.num_encoder_layers=3 \
    --override model.num_decoder_layers=3
```

## 一句话总结

编码器把英文读成"上下文化的向量"，解码器拿着它一个词一个词写德语，
残差保证梯度能传回去，LayerNorm 保证数值尺度和层数无关。
