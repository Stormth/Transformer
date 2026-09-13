# 第 2 章：手写 BPE 分词器

对应代码：`nmt/bpe.py`、`nmt/dataset.py`

## 为什么不能直接按词切

英文按空格切词看起来很自然，但：

* 词表会爆炸（英语有几十万种词形）；
* 遇到没见过的词（人名、缩写、新词、拼错）只能吐 `<unk>`；
* 中文根本没有空格。

按字符切又走向另一个极端：序列太长（注意力开销 O(T²)），每个 token 的信息量太小。

**BPE 是折中**：从字符出发，反复把最常一起出现的相邻符号合并。结果是高频词变成一个 token（`the`、`problem`），低频词拆成几个有意义的片段（`unbelievable` → `un` + `believ` + `able`）。

## 怎么把句子切成"词"

`pre_tokenize` 按优先级匹配四类东西：

```
连续汉字串        他明天在学校讨论这个问题吗
英文单词（含撇号） don't
数字（含小数）     3.14
其它单字符         标点、符号
```

为什么中文要整串当一个"词"？因为 BPE 只在**词内部**合并。如果每个汉字都是独立的词，它们之间永远不会被合并，你就得不到"我们""可以"这种高频字组。把连续汉字串当成一个词，BPE 才会在里面学出常见的字组合。

## BPE 训练算法

```
1. 统计所有"词"的出现次数
2. 把每个词拆成符号序列       committee → c o m m i t t e e</w>
3. 反复：
     找出当前出现次数最多的相邻符号对 (a, b)
     在所有词里把 a b 合并成 ab
     把这一步记进合并规则表
4. 直到词表达到目标大小，或者最高频的符号对出现次数低于阈值（min_frequency）
```

**最容易忽视的工程点**：每一步只重算"包含这个 pair 的词"，而不是重扫全部语料。本项目用 `pair_counts`（pair → 次数）和 `pair_words`（pair → 包含它的词下标集合）两张表做到这一点。没有这个优化，十几万句语料在纯 Python 里做上万次全量重扫会慢到不可用。

> 还有一步可以更快：每轮用 `max()` 找最高频 pair 是 O(不同 pair 的数量)，
> 用堆（heapq + 惰性删除）可以把它降到 O(log n)。本项目的写法更直观，
> 想练手的话这是个很好的优化题目 —— 先在 `data/ready` 上记录一份基准时间，再改。

## 词尾标记 `</w>`：为什么必须有它

假设 `committee` 被切成 `c o mmit tee`。解码时如果直接把 token 用空格连起来，就得到 `c o mmit tee` —— 词被拆散了。

标准做法（BPE 原论文）是给每个词的**最后一个字符**加 `</w>`：

```
训练时： committee → c o m m i t t e e</w>
解码时： 遇到带 </w> 的 token 就知道"这个词结束了"，
        把它前面攒着的碎片直接连起来，再加一个空格
```

中文串不加这个标记 —— 汉字之间本来就不加空格，加了只会让词表多出一倍的汉字形式。

## 分词器怎么用

```python
from nmt.bpe import BPE

tokenizer = BPE.load("data/ready/vocab.json")
ids = tokenizer.encode("他明天在学校讨论这个问题吗？", add_bos=True, add_eos=True)
print(ids)                      # [1, 306, 88, 45, 2]
print(tokenizer.tokenize("committee will decide"))
print(tokenizer.decode(ids))    # 他明天在学校讨论这个问题吗？
```

## 批处理：分桶 + 动态 padding

对应代码：`nmt/dataset.py`

句子长度差异极大，如果用一个固定的 batch size：

* 一批里混着长句和短句 → 短句要补到长句的长度，大量算力浪费在 `<pad>` 上；
* 句子一长就 OOM（注意力是 O(T²)）。

两个标准做法：

1. **动态 batching**：一个 batch 里只补到本 batch 最长的那句，并限制"这个 batch 的 token 总数不超过 `max_tokens`"，batch 大小随长度自动伸缩。
2. **长度分桶**：按长度排序，把长度接近的句子放进同一批。代价是打乱随机性，用"桶内打乱 + batch 顺序打乱"来补偿。

代码里的 `LengthBucketSampler` 就是这两件事，`collate_batch` 负责补 `<pad>`。

## 自己动手

```bash
# 1. 词表大小实验：小词表序列更长，大词表嵌入参数更多
python -m nmt.corpus --vocab-size 8000 --skip-download
#    看打印出来的 "英文 ... token / 中文 ... token" 变化了多少

# 2. 看分词效果（`</w>` 就是词尾标记）
python -m nmt.translate --checkpoint checkpoints/best.pt --text "..." --show-tokens

# 3. 手工验证解码可逆
python -c "from nmt.bpe import BPE; t=BPE.load('data/ready/vocab.json'); s='he will finish the project tomorrow.'; print(t.encode(s)); print(t.decode(t.encode(s)))"
```

## 值得记住的三句话

* 词表大小是"序列长度"和"参数数量"之间的取舍，没有标准答案，只能实验。
* 分词器只能在**训练集**上训练。用全部数据训分词器，测试集的词汇会通过词表泄漏进来。
* 解码可逆性要专门测（见 `tests/test_bpe.py`）。
