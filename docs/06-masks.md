# 第 6 章：掩码与"我怎么知道写对了"

对应代码：`nmt/masks.py`、`nmt/verify.py`

掩码是 Transformer 里**最容易写错、错了又最难发现**的地方。这一章把两件事讲清楚：
掩码到底在屏蔽什么，以及怎么**用工具证明**自己写对了。

## 两种必须屏蔽的位置

### 1. padding

一个 batch 里的句子长度不同，短的补 `<pad>`。如果注意力能读到 `<pad>`：

* 短句的表示会随"补了多少个 pad"变化，同一个句子在不同 batch 里表现不一样；
* 模型可能学会"看 pad 的位置来数长度"这种毫无意义的特征。

### 2. 未来（因果掩码）

训练时我们把**完整的中文译文**喂给解码器（这叫 teacher forcing），
同时要求"预测第 t 个位置时只能看到 0..t-1"。如果不加因果掩码，
模型直接看答案，训练 loss 会飞一样地降到 0，推理时却什么都不会 —— 这是最典型的"看起来训练成功"的假象。

## 本项目的掩码约定

```
mask 是 bool 张量，True 表示"不允许看"
```

在注意力里被填成 `torch.finfo(dtype).min`：

```python
scores = scores.masked_fill(mask, mask_value(scores.dtype))
```

### 为什么不用 `-1e9` / `-inf`

混合精度下 `float16` 能表示的最大值只有 65504，`-1e9` 会直接溢出成 `-inf`。
如果某个 query 位置**所有 key 都被屏蔽**（padding 位置就是这样），
`softmax([-inf, -inf, ...])` 会算出 `nan`，然后整个 batch 的 loss 变成 nan。

用 `torch.finfo(dtype).min`（float16 下是 -65504）既保证 softmax 后是 0，又不会溢出。
这一行代码值得你在自己的项目里抄走。

## 三处掩码的形状

| 用在哪 | 屏蔽什么 | 形状 | 为什么长这样 |
| --- | --- | --- | --- |
| 编码器自注意力 | 英文 padding | `[B, 1, 1, S]` | 要和注意力分数 `[B, h, Tq, Tk]` 广播 |
| 解码器自注意力 | 中文 padding + 因果 | `[B, 1, T, T]` | 行列都是中文位置 |
| 解码器交叉注意力 | 英文 padding | `[B, 1, 1, S]` | 查询中文、被看的是英文 |

因果掩码是严格上三角：

```
位置   0  1  2  3
 0     .  X  X  X
 1     .  .  X  X
 2     .  .  .  X
 3     .  .  .  .
```

注意对角线是"."（可以看自己）。如果连自己都屏蔽，那一行的 softmax 分母就是 0。

## 怎么证明自己写对了

### 手段一：不变量测试

`tests/test_model.py` 里有两个"物理定律"级别的测试：

```python
def test_decoder_is_causal():
    # 只改第 4 个位置的 token，前 3 个位置的输出必须一字不变

def test_incremental_decoding_matches_full_forward():
    # 带 KV cache 的逐步解码 == 整句一次前向（差异 < 1e-5）
```

这两个测试能抓住绝大多数掩码和 cache 的 bug，而且**不依赖任何训练结果**。
写自己的 Transformer 时，几乎应该把这两个测试当成"编译检查"。

### 手段二：和官方实现逐层对齐

```bash
python -m nmt.verify --preset tiny --dtype float64 --show-internals
```

它会：

1. 建一个结构完全相同的 `nn.Transformer`；
2. 把我们的权重搬过去（Q/K/V 拼成官方的 `in_proj_weight`）；
3. 喂同样的输入，逐层比较输出。

跑出来的结果大概是：

```
编码器输出最大差异（只看有效位置）  5.607e-07
解码器输出最大差异（只看有效位置）  4.813e-07
判定：一致 ✅ —— 最大差异 5.607e-07 / 数值量级 3.12 = 相对误差 1.797e-07
```

（这是本项目在 float64 下的实测值。用 float32 时差异在 1e-6 量级。）

### 为什么不是"零差异"

这是这一章最值钱的一段。

官方实现在推理模式下有几条**融合快速路径**：

* 编码器收到 `key_padding_mask` 时会切到 nested tensor，直接把 padding 位置的输出填 0；
* 某些组合下会走 C++ 融合算子，内部按 float32 计算。

结果是**官方自己两条调用路径在 float64 下就能差 7e-7**（带掩码时甚至差 3.0）。
所以：

* 我们和官方逐层调用之间的相对误差 1e-7 量级 → 这是**融合算子的精度地板**，不是实现错误；
* 如果你真的写错了（掩码反向、残差接错、交叉注意力喂错张量），差异会是 **0.1~1** 的量级，高下立判。

知道"精度地板在哪"，比记住"应该完全相等"有用得多 —— 这也是 `verify.py` 里
用**相对误差**而不是绝对误差做判据的原因。

## 自己动手

```bash
# 1. 把因果掩码写反（把 triu 改成 tril），跑测试
python -m pytest tests/test_model.py -q
#    应该看到 test_decoder_is_causal 直接失败

# 2. 让 padding 位置参与注意力（把掩码全设为 False），观察验证 loss 的变化

# 3. 看看官方两条路径自己差多少
python -m nmt.verify --preset tiny --dtype float64 --show-internals

# 4. 把 mask_value 换成 -1e9，再用 float16 训练，看会不会 nan
python -m nmt.train --preset small --override train.amp_dtype=float16 --max-steps 50
```

## 一句话总结

掩码决定"哪里不许看"；因果掩码写错会让训练假成功，padding 掩码写错会让模型不稳定；
而**测试 + 和官方对齐**是唯一能让你睡好觉的办法。
