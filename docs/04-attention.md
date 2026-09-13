# 第 4 章：注意力机制

对应代码：`nmt/attention.py`

这一章是整个 Transformer 的核心，也是本书里最值得反复读的一章。

## 从"翻译一个词要看哪里"说起

翻译 "The agreement was signed **after** months of negotiation" 时，
德语的"nach monatelangen Verhandlungen"里，`nach` 要对应英文的 "after"。
模型怎么知道该看哪个词？

答案就是注意力：**给每个位置算一个"该看哪里"的权重分布，然后按权重把信息加权求和。**

## 三个矩阵：Q、K、V

每个位置从这个词自己的向量出发，通过三个不同的线性层得到三个角色：

| 名字 | 含义 | 直觉 |
| --- | --- | --- |
| Query（Q） | 我在找什么 | 我现在想写"之后"，我要找时间状语 |
| Key（K） | 我是什么 | "after" 这个位置说：我是个时间连接词 |
| Value（V） | 我提供什么 | 如果决定看你，你能给我什么信息 |

公式（论文 3.2.1）：

```
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V
```

代码就是三行：

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # [B, h, Tq, Tk] 两两相关度
scores = scores.masked_fill(mask, mask_value)        # 不许看的位置打成极小值
weights = softmax(scores, dim=-1)                    # 归一化成概率
out = weights @ v                                    # [B, h, Tq, d_v] 按权重求和
```

## 为什么必须除以 `sqrt(d_k)`

这是最常被问到、也最值得算清楚的一步。

假设 Q、K 的每个分量独立、均值 0、方差 1。点积 `q · k = Σ q_i k_i` 是 d_k 个独立项之和，所以：

```
E[q·k] = 0,    Var[q·k] = d_k
```

`d_k = 64` 时点积的标准差就是 8，而最大可能到几十。softmax 对输入尺度非常敏感：
输入一大，它就被推向"最大值输出 1、其余输出 0"的饱和区，梯度几乎为 0（这就是 softmax 梯度消失）。

除以 `sqrt(d_k)` 把方差拉回 1，softmax 待在梯度健康的区间。**这不是经验技巧，是量纲上的必然。**

## 多头：为什么要切成 8 份

一个注意力头只能学到一种"关注模式"。但语言里的关系是多样的：

* 头 1 学主谓关系；
* 头 2 学"形容词 → 修饰的名词"；
* 头 3 学"英文的 after → 德语的 nach"。

多头就是**并行做 8 次注意力，每次用自己的 Q/K/V 投影**，最后把 8 个结果拼起来再过一层线性。

实现上最关键的一行：

```python
x = x.view(B, T, n_heads, d_k).transpose(1, 2)   # [B,T,d_model] → [B,h,T,d_k]
```

`d_model = 512`、`n_heads = 8` 时 `d_k = 64`。
所谓"多头"，在代码里就是一次 reshape + 一次 transpose，然后把 head 维度并进 batch 维度做批量矩阵乘 —— **不需要写 for 循环**。

注意：多头**不增加**计算量（每个头只有 1/8 的维度），它改变的是"学什么"，不是"算多少"。

## 掩码：哪三个地方要屏蔽

注意力本身很"贪心"——它想看完所有位置。有两类位置必须挡住：

1. **padding**：`<pad>` 是凑长度补出来的，读到它等于往句子里掺噪声，而且会让短句的表示随 padding 数量变化。
2. **未来**：解码器在生成第 t 个字时，绝不能看到 t 之后的内容，否则训练时等于抄答案（考试时又没有答案可抄）。

本项目的约定：**掩码是 bool 张量，True 表示不允许看**，在注意力里被填成 `torch.finfo(dtype).min`。

为什么不用 `-1e9`？因为在 float16 混合精度下 `-1e9` 会溢出成 `-inf`，如果某一行**全部**被屏蔽，`softmax(-inf)` 会得到 `nan`。用 `finfo.min` 既能保证 softmax 后是 0，又不会溢出。这个细节在混合精度训练里能救你一次。

（详见第 6 章）

## KV cache：为什么推理能快几十倍

自回归生成时，第 t 步的 query 只有 1 个位置，但需要前 t 个位置的 K 和 V。

朴素做法：每一步都重新算一遍整段历史 → 生成 100 个字就做了 100 次满长度前向，`O(T²)` 的总计算变成 `O(T³)`。

标准做法：把已经算过的 K、V 存起来（cache），每步只计算新位置的 K、V 然后拼接：

```python
k = torch.cat([cache["k"], k], dim=2)     # 历史 + 当前
v = torch.cat([cache["v"], v], dim=2)
cache["k"], cache["v"] = k, v
```

交叉注意力更省：英文那侧的 K/V 是固定的，**整个解码过程中只算一次**，之后每步复用。

代价是显存：cache 大小约 `2 × 层数 × heads × 长度 × d_k × 精度字节数`。

本项目的 `tests/test_model.py::test_incremental_decoding_matches_full_forward` 会验证：
**带 cache 的增量解码和整句一次前向的结果完全一致**（float32 下差异在 1e-6 以内）。
这个测试非常值钱 —— cache 写错是那种"能跑、但悄悄生成垃圾"的 bug。

## 用工具看注意力

```bash
# 终端里画字符热力图 + 生成完整 HTML 报告
python -m nmt.inspect --checkpoint checkpoints/best.pt \
    --text "The committee postponed the meeting until next Monday." \
    --html attention.html
```

值得观察的现象：

* **编码器自注意力**：相邻的词、以及"助动词和主动词"之间往往有很强的连接；
* **解码器交叉注意力**：斜对角线附近权重高 → 词序基本一致；德语动词在从句里跑到句尾
  （`... , dass er das Problem bespricht`），这时能看到明显偏离对角线的跳转；
  冠词、介词这类高频词往往均匀地看很多位置。
* **不同层不同头分工不同**：浅层的头偏向局部（看邻居），深层的头偏向全局（看整句）。

## 自己动手

```bash
# 1. 关掉缩放，看训练会不会变差（是不是真的需要 sqrt(d_k)）
#    改 nmt/attention.py 里那一行，跑几十步看 loss

# 2. 把 n_heads 从 8 改成 1，参数量几乎不变，看 BLEU 掉多少
python -m nmt.train --preset small --override model.n_heads=1

# 3. 故意把因果掩码的方向写反（改成下三角），跑测试看会不会被抓住
python -m pytest tests/test_model.py -q
```

## 一句话总结

注意力 = 用 Query 和 Key 算出"该看哪"，用 Value 取"看来的信息"，用 mask 决定"哪里不许看"，用 cache 避免重复计算。
