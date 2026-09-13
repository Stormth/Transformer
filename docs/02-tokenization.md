# 第 2 章：手写 BPE 分词器

对应代码：`nmt/bpe.py`、`nmt/dataset.py`

## 为什么不能直接按词切

英语和德语按空格切词看起来很自然，但：

* 词表会爆炸（德语的复合词几乎无穷无尽）；
* 遇到没见过的词（人名、缩写、新词、拼错）只能吐 `<unk>`；
* 德语尤其严重：`Wirtschaftswachstum` 这种复合词在真实新闻里到处都是，
  而它只是 `Wirtschaft` + `Wachstum` 拼起来的。

按字符切又走向另一个极端：序列太长（注意力开销 O(T²)），每个 token 的信息量太小。

**BPE 是折中**：从字符出发，反复把最常一起出现的相邻符号合并。结果是：

* 高频词变成一个 token（`the`、`und`）；
* 复合词被拆成有意义的片段（`Wirtschaftswachstum` → `Wirtschaft` + `swachstum`）；
* 词尾的语法信息能单独建模（`en</w>`、`ung</w>`、`te</w>`），
  这对形态丰富的德语特别有用：名词复数、形容词词尾、动词变位都有规律可循。

## 怎么把句子切成"词"

`pre_tokenize` 按优先级匹配三类东西：

```
字母词（含连字符/撇号）   don't   E-Mail   well-known   nächste
数字（含小数/千分位）     3.14    1,000
其它单字符                标点、符号
```

两个容易写错的地方：

* **连字符和撇号要跟着词走**。`E-Mail` 要是被切成 `E`、`-`、`Mail`，
  BPE 就会把 `-</w>` 学成一个高频 token，白白浪费词表；德语里这种写法非常多。
* **变音字母是字母**。`ä ö ü ß` 必须当作普通字母处理（本项目用 `[^\W\d_]`，
  也就是"Unicode 字母"来匹配），否则 `für` 会被拆成 `f`、`ü`、`r`。

## BPE 训练算法

```
1. 统计所有"词"的出现次数
2. 把每个词拆成符号序列       Wirtschaftswachstum → W i r t ... m</w>
3. 反复：
     找出当前出现次数最多的相邻符号对 (a, b)
     在所有词里把 a b 合并成 ab
     把这一步记进合并规则表
4. 直到词表达到目标大小，或者最高频的符号对出现次数低于阈值（min_frequency）
```

**最容易忽视的工程点**：每一步只重算"包含这个 pair 的词"，而不是重扫全部语料。
本项目用 `pair_counts`（pair → 次数）和 `pair_words`（pair → 包含它的词下标集合）两张表做到这一点。
没有这个优化，几十万句语料在纯 Python 里做几万次全量重扫会慢到不可用。

> 这一步是数据准备里最慢的：32k 词表大约要 1 小时（10 万句对的训练子集）。
> `corpus.py` 会给它配一个带预计剩余时间的进度条。
> 想快一倍就用 `--vocab-size 16000` —— 合并次数少一半，时间也差不多减半。
> 想更快，就得换编译好的实现（sentencepiece / HF tokenizers），
> 代价是失去"每一行都能读懂"这个特性。

## 词尾标记 `</w>`：为什么必须有它

假设 `committee` 被切成 `c o mmit tee`。解码时如果直接把 token 用空格连起来，
就得到 `c o mmit tee` —— 词被拆散了。

标准做法（BPE 原论文）是给每个词的**最后一个字符**加 `</w>`：

```
训练时： committee → c o m m i t t e e</w>
解码时： 遇到带 </w> 的 token 就知道"这个词结束了"，
        把它前面攒着的碎片直接连起来，再加一个空格
```

## 分词器怎么用

```python
from nmt.bpe import BPE

tokenizer = BPE.load("data/ready/vocab.json")
ids = tokenizer.encode("Der Ingenieur bespricht das Problem.", add_bos=True, add_eos=True)
print(ids)                       # [1, 306, 88, 45, 2]
print(tokenizer.tokenize("Wirtschaftswachstum"))   # 看复合词怎么被拆开
print(tokenizer.decode(ids))     # Der Ingenieur bespricht das Problem.
```

## 批处理：分桶 + 动态 padding

对应代码：`nmt/dataset.py`

句子长度差异很大，如果用一个固定的 batch size：

* 一批里混着长句和短句 → 短句要补到长句的长度，大量算力浪费在 `<pad>` 上；
* 句子一长就 OOM（注意力是 O(T²)）。

两个标准做法：

1. **动态 batching**：一个 batch 里只补到本 batch 最长的那句，并限制"这个 batch 的 token 总数不超过 `max_tokens`"，batch 大小随长度自动伸缩。
2. **长度分桶**：按长度排序，把长度接近的句子放进同一批。代价是打乱随机性，用"桶内打乱 + batch 顺序打乱"来补偿。

代码里的 `LengthBucketSampler` 就是这两件事，`collate_batch` 负责补 `<pad>`。

## 自己动手

```bash
# 1. 词表大小实验：小词表序列更长，大词表嵌入参数更多
python -m nmt.corpus --vocab-size 16000 --skip-download
#    看打印出来的 "英文 ... token / 德文 ... token" 变化了多少

# 2. 观察复合词是怎么被拆开的
python -c "
from nmt.bpe import BPE
t = BPE.load('data/ready/vocab.json')
for w in ['Wirtschaftswachstum', 'Geschwindigkeitsbegrenzung', 'Bundeskanzleramt']:
    print(w, '->', t.tokenize(w))
"

# 3. 看一次真实翻译的分词（`</w>` 就是词尾标记）
python -m nmt.translate --checkpoint checkpoints/best.pt --text "..." --show-tokens

# 4. 手工验证解码可逆
python -c "
from nmt.bpe import BPE
t = BPE.load('data/ready/vocab.json')
s = 'Der Ingenieur bespricht das Problem morgen in der Schule.'
print(t.decode(t.encode(s)) == s)
"
```

## 值得记住的三句话

* 词表大小是"序列长度"和"参数数量"之间的取舍，没有标准答案，只能实验。
* 分词器只能在**训练集**上训练。用全部数据训分词器，测试集的词汇会通过词表泄漏进来。
* 解码可逆性要专门测（见 `tests/test_bpe.py`）。
