# 第 8 章：解码（怎么把模型用起来）

对应代码：`nmt/decoding.py`、`nmt/inference.py`、`nmt/translate.py`

训练是"给完整句子算概率"，推理是"从零开始一个字一个字写出来"。这两件事的代码几乎完全不同，
也是很多人写完训练就卡住的地方。

## 自回归生成的基本循环

```
输入：英文 src
1. 编码一次，得到 memory
2. 把 <bos> 喂给解码器（带 KV cache）
3. 取最后一个位置的 logits，选一个 token
4. 把这个 token 接到序列末尾，回到第 3 步
5. 生成 <eos> 或达到 max_len 就停
```

每一步只处理**一个新的位置**，历史信息从 KV cache 里取。
这就是为什么推理速度很大程度上取决于 cache 写得好不好（见第 4 章）。

## 贪心 vs 束搜索

### 贪心（greedy）

每步都选概率最大的词。快，但**一步选错就再也回不来**。

典型症状："他明天去学校" 被翻成 "He goes to school tomorrow" 这类句子时，
贪心可能在第一个词就选了个不合适的开头，后面只能硬接。

### 束搜索（beam search）

同时保留 K 条候选路径，每步把所有路径的下一步扩展出来，只留累计概率最高的 K 条。

代价是约 K 倍的时间，收益通常是 **+1~2 BLEU**。K 再大收益就迅速衰减，
而且 K 太大反而会让译文变得"平庸"（更长的句子被系统性偏好），幻觉也更多。

```bash
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --compare-decoders
```

## 两个必须处理的细节

### 1. 长度惩罚

对数概率是负数，越乘越小。直接按总分排序会**系统性偏向短句**（少乘几个负数就赢），
甚至生成空句子。

GNMT 的解法是除以一个和长度有关的因子：

```
score = log P(y) / ((5 + len(y)) / 6)^alpha
```

`alpha=0.6` 时短句略微受罚、长句略微占优。这是束搜索的默认设置，可以调：

```bash
--length-penalty 0.0    # 不加惩罚（偏向短句）
--length-penalty 1.0    # 强烈偏好长句
```

**观察长度比**（译文 token 数 / 参考 token 数）是最直观的诊断方式：
远小于 1 就调大惩罚，远大于 1 就调小。

### 2. `<eos>` 的处理

某条路径生成了 `<eos>` 就说明"这句写完了"：把它放进**完成池**，
不再参与后续扩展，但仍然是这个句子的候选答案。

三个容易写错的地方：

* 完成后它的分数不能再继续累加（否则长句分数一定更低）；
* 如果所有路径都完成了，整个 batch 就可以提前结束；
* 如果一条路径都没完成（被 max_len 截断），要退回"当前分数最高的活跃路径"，不能返回空。

另外还有 `min_len`：强制前几步不许生成 `<eos>`，防止"一个字都不写"。

### 3. 生成长度要设上限（而且是动态的）

训练不足的模型经常不会输出 `<eos>`，于是每句话都硬生成到 `max_len` 才停。
评测两万句时，这意味着大部分算力都花在"生成废话"上。

本项目的做法（`inference.translate_dataset`）是按输入长度封顶：

```
每批的生成长度上限 = min(max_len, max(16, 该批最长原文 × 2 + 10))
```

译文通常比原文长不了太多，2 倍已经很宽松；而这一行能把弱模型的评测时间砍掉好几倍。
想看模型"到底会不会自己停下来"，把 `adaptive_max_len` 关掉就现原形了。

## 束搜索 + KV cache 的实现要点

这是本项目里最"工程"的一段代码（`nmt/decoding.py`）。

把 `[B, K]` 条活跃路径拍平成 `[B*K]` 一个批次来计算，每步做一次 `topk` 选出新的 K 条路径，
然后把 KV cache 按"被选中的父路径"重排：

```python
flat_parents = (arange(B).unsqueeze(1) * K + new_parents).view(-1)
tokens = tokens.index_select(0, flat_parents)
for cache in self_caches + cross_caches:
    cache["k"] = cache["k"].index_select(0, flat_parents)   # 关键：cache 也要跟着重排
    cache["v"] = cache["v"].index_select(0, flat_parents)
```

**忘了重排 cache** 是这类实现最经典的 bug：程序不报错、能跑完，只是译文质量莫名其妙地差。

验证方式很便宜：**`beam_size=1` 的束搜索必须和贪心解码逐字一致**
（`tests/test_decoding.py::test_beam_size_one_equals_greedy` 就在测这个）。

## 命令行用法

```bash
# 翻一句（默认贪心，最快）
python -m nmt.translate --checkpoint checkpoints/best.pt --text "The meeting was postponed."

# 束搜索，通常更通顺
python -m nmt.translate --checkpoint checkpoints/best.pt --beam-size 4 --text "..."

# 批量翻译一个文件
python -m nmt.translate --checkpoint checkpoints/best.pt --file news.en --out news.zh --beam-size 4

# 交互模式（不带 --text/--file，逐行输入）
python -m nmt.translate --checkpoint checkpoints/best.pt

# 看分词结果，确认输入被切成了什么
python -m nmt.translate --checkpoint checkpoints/best.pt --text "..." --show-tokens
```

## 自己动手

```bash
# 1. 对比不同束宽（这一条命令把 1/2/4/8 全跑一遍）
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --compare-decoders

# 2. 看长度惩罚的影响
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --beam-size 4 --length-penalty 0.0
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --beam-size 4 --length-penalty 1.2

# 3. 把 min_len 设成 5，看译文开头会不会更稳
#    改 nmt/decoding.py 里 greedy_decode / beam_search_decode 的 min_len 默认值

# 4. 故意不重排 cache（注释掉那两行），跑一遍评测
#    你会发现 BLEU 掉得莫名其妙 —— 这就是它的可怕之处
```

## 一句话总结

贪心快、束搜索好；长度惩罚决定译文长短；KV cache 决定推理速度；
而"beam_size=1 等于贪心"是检验实现正确性的一把快刀。
