# 第 3 章：词嵌入与位置编码

对应代码：`nmt/embedding.py`

## 词嵌入：id 变向量

一个 `[vocab_size, d_model]` 的查表矩阵。`embedding(ids)` 就是按 id 取行：

```
ids   [B, T]        →      x   [B, T, d_model]
[[1, 306, 88]]             [[[0.01, -0.2, ...], [0.3, 0.1, ...], ...]]
```

### 为什么要乘 `sqrt(d_model)`

论文 3.4 节说"词嵌入乘 `sqrt(d_model)`"，很多人抄了这个数却不知道为什么。

原因在下一节的**位置编码**：位置编码的数值范围固定在 `[-1, 1]`。而词嵌入如果用默认初始化（`N(0,1)`），每个分量标准差是 1，和位置编码量级差距不大；但本项目按标准做法把词嵌入初始化成 `N(0, 0.02)`，量级就比位置编码小了 50 倍。

两者是**相加**的：

```
输入 = 词嵌入 + 位置编码
```

如果量级差太多，位置信息会把词义信息淹掉（或者反过来）。乘 `sqrt(d_model)` 把词嵌入放大回同一量级，让两个信号能公平地相加。

**这是个可以在项目里亲手验证的点**：把 `scale_embedding` 关掉再训一次，观察 loss 曲线。

### 权重共享

`tie_embeddings=True` 时，输出层和**目标词嵌入**共用同一个矩阵：

```python
self.generator.weight = self.tgt_embedding.embedding.weight
```

为什么合理？因为这个矩阵的每一行既是"德语词 w 的输入向量"，又是"预测德语词 w 的输出向量"。
共享带来三个好处：参数少一份（词表 16k、d_model 512 时省 800 万参数）；
输出的词向量和输入在同一空间里，几何上更自洽；小数据上更不容易过拟合。

注意输出的 `nn.Linear(..., bias=False)` —— 有 bias 就没法共享了，而且实验里 bias 也没带来什么好处。

## 位置编码：注意力不知道顺序

自注意力是一个**置换等变**的操作：你把输入句子的顺序打乱，输出也只是跟着打乱而已。
也就是说，光靠注意力，模型根本不知道"他打了我"和"我打了他"的区别。

所以必须显式注入位置信息。论文用的是正弦编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

同一个位置 `pos` 上，不同维度对应不同的波长：低维波长短（变化快，区分相邻词），高维波长长（变化慢，区分远距离）。

### 为什么用固定的 sin/cos，而不是可学习的 embedding

1. 不增加参数，也不受训练时见过的最大长度限制；
2. **相对位置可以线性表示**：对任意固定位移 `k`，`PE(pos + k)` 能写成 `PE(pos)` 的线性变换（这就是 sin/cos 的合角公式）。模型因此更容易学到"看前面第 3 个词"这种相对概念，而不是死记"第 7 个位置"。

现在的模型（BERT、GPT）大多改用**可学习位置嵌入**或 RoPE，但对这个项目来说，正弦编码的好处是：**你能一眼看出位置信息是从哪来的**，不用怀疑是训练出来的。

### 实现上的小技巧

不要真的去算 `10000^(2i/d)`（容易溢出），而是在 log 空间算倒数：

```python
div_term = exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))
```

结果完全一样，数值上更稳。

## 自己动手

```bash
# 1. 画一张位置编码图，看不同维度的波长
python -c "
import torch
from nmt.embedding import sinusoidal_table
pe = sinusoidal_table(80, 128)
ramp = ' .:-=+*#%@'
for row in pe.T[:24]:
    print(''.join(ramp[min(9, int((v+1)/2*9))] for v in row))
"

# 2. 验证相对位置的线性性质
#    固定 pos，看 PE(pos+k) 与 PE(pos) 之间是不是简单的线性关系

# 3. 关掉嵌入缩放再训一轮，比较 loss 曲线
python -m nmt.train --preset small --override model.scale_embedding=false

# 4. 关掉权重共享
python -m nmt.train --preset small --override model.tie_embeddings=false
```

## 一句话总结

词嵌入给模型"这个词是什么意思"，位置编码给模型"这个词排在第几位"，两者量级一定要匹配。
