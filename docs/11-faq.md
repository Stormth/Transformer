# 第 11 章：常见坑与排错

这一章收集的是"自己写 Transformer 一定会撞上"的问题。每一条都对应本项目里的一段代码。

## 数据与分词

### 德语输出被拆成一堆碎片

症状：`Wirtschaft swachstum`、`Gesch wind igkeit s begrenzung`。

原因有两个，先分清是哪一个：

1. **BPE 没学过这个词**：它会被拆成能认识的最长片段，这是正常的子词行为，
   不是 bug。看 `python -m nmt.corpus` 打印的词表大小和序列长度就能判断。
2. **词表太小**：复合词只能靠碎片段拼。改大 `--vocab-size` 再看。

### 英文输出被拆成 c o mmit tee

缺少**词尾标记**。BPE 把一个词切成了几段，解码时无法判断哪些片段属于同一个词。
标准做法是训练时给每个词的最后一个字符加 `</w>`（见 `bpe.py::_word_to_symbols`）。

### 输出里出现 `</w>`

正常情况下 `decode()` 会把词尾标记吃掉。如果原样出现在译文里，说明分词器和模型对不上
（比如换了 `vocab.json` 但用的是旧 checkpoint）—— 重新训一个模型即可。

### 输出里出现 `<unk>`

模型预测出了词表里的 `<unk>`。可能原因：

* 训练数据里就有太多 `<unk>`（词表太小，或分词器训练数据不够）——
  跑 `python -m nmt.corpus` 看报告；
* 推理时输入的字符没进词表：德语人名、地名里的特殊字符很常见。

解决方向：调大词表、加数据。英德这种同字母表的语言对，`<unk>` 率通常很低（千分之几）。

### 变音字母 ä ö ü ß 被当成标点丢掉了

这是 **BLEU 切词**里的坑，不是分词器的：如果评测时用 `[A-Za-z]+` 去切词，
`für` 会变成 `f` 和 `r`，分数被人为压低。

本项目的 `tokenize_for_bleu` 用 `[^\W\d_]+`（Unicode 字母）匹配，`ä ö ü ß` 都会保留。
`tests/test_bleu.py::test_tokenize_keeps_umlauts_and_sharp_s` 专门守着这条。

### 训练 loss 降得很好，推理却一塌糊涂

先检查训练时 `src` 和 `tgt` 有没有错位。正确做法是：

```
解码器输入 = tgt[:, :-1]     以 <bos> 开头
label      = tgt[:, 1:]      以 <eos> 结尾
```

错位的典型症状就是"训练 loss 很低、生成的全是乱码"。

## 掩码与形状

### loss 变成 nan

最常见的两个原因：

1. **整行被屏蔽**：某一行 mask 全是 True，softmax 得到 nan。本项目用
   `torch.finfo(dtype).min` 而不是 `-1e9`，就是为了在 float16 下也不溢出（见第 6 章）。
2. **学习率太大**：梯度爆炸，几个 batch 内参数就变成 inf。先把 lr 调小一个数量级试试。

### 形状报错但看不出问题

用形状追踪看数据流：

```bash
python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes --text "He will finish the project tomorrow."
```

几个高发的形状错误：

| 报错 | 原因 |
| --- | --- |
| `view size is not compatible` | `transpose` 之后没 `.contiguous()` 就 `view` |
| 广播失败、维度对不上 | 掩码少了一个维度。注意力分数是 `[B, h, Tq, Tk]`，掩码至少要能广播到它 |
| `Expected target size ...` | `nn.CrossEntropyLoss` 把类别维当成第 1 维了。要么拍平，要么用 `loss.py` 里的包装 |

### 增量解码和整句前向结果不一致

三个检查点：

1. 位置编码有没有跟着偏移（第 t 步要用位置 t，本项目是 `position_offset`）；
2. cache 拼接顺序对不对（历史在前、当前在后）；
3. 增量解码不需要因果掩码（query 只有一步），但 **padding 掩码仍然需要**。

`tests/test_model.py::test_incremental_decoding_matches_full_forward` 就是在守这条线。

## 训练

### loss 卡在 1.x 不动

先确认是不是标签平滑的"地板"。eps=0.1 时理论下界在 0.5~0.9 附近（取决于词表大小），
到那儿不再降是正常的。判断方法是**同时看 BLEU**：loss 不动但 BLEU 在涨，说明还在学。

### 刚开始就震荡，loss 忽上忽下

**学习率太大了**。注意 Noam 的峰值公式 `scale × d_model^-0.5 × warmup^-0.5`：
把 warmup 调短、把 d_model 调小都会让峰值变大。启动日志里会打印峰值，先看那一行。

```bash
python -m nmt.train --preset small --override train.lr_scale=0.3
```

### 显存不够（CUDA out of memory）

按这个顺序调：

1. `--override train.max_tokens=8192`（最有效，直接减少一个 batch 的 token 数）
2. `--override train.accum_steps=2`（用梯度累积保持等效 batch size）
3. `--override train.max_src_len=128 --override train.max_tgt_len=128`
4. `--override model.d_model=256 --override model.d_ff=1024`
5. 确认 `train.amp=true`（默认已开）

注意注意力是 O(T²)，所以"长度上限"对显存的影响是平方级的。

### 训到一半被杀掉，能接着训吗

```bash
python -m nmt.train --preset base --save-dir checkpoints --resume auto
```

`last.pt` 里存了优化器状态、学习率步数、GradScaler 缩放因子、epoch/step 和历史指标，
续训曲线应该和没中断一样。只存模型权重是接不回去的，曲线会突然变差。

**注意：续训要用和原训练相同的 `--preset`**。因为模型结构是按命令行给的预设建的，
预设换掉会直接形状不匹配报错。

### 服务器上要不要设 num_workers

要。Windows 上多进程 DataLoader 容易出问题（默认是 0），Linux 服务器上设 4~8 能明显加速：

```bash
python -m nmt.train --preset base --override train.num_workers=4
```

## 评测与推理

### BLEU 一直是 0.00

训练早期很正常：BLEU 要求 1~4 gram 都有命中，早期译文全是高频词时四元命中率为 0，
整体就被定义成 0。看 **chrF** 更灵敏。如果训练好几轮 chrF 也不动，那才是真有问题。

### BLEU 很高但译文读起来不对

说明测试集和训练集太像，或者数据本身是模板生成的。本项目早期版本用合成语料时，
验证 BLEU 能到 99 —— 那个数字毫无意义。**一定要用真实语料和官方测试集。**

### 译文反复重复同一句

自回归模型的经典问题，训练不充分时尤其明显。缓解手段：

* 用束搜索代替贪心；
* 检查 `<eos>` 有没有被正确监督（最后一个 label 应该是 `<eos>`）；
* 调大长度惩罚；
* 加 repetition penalty（本项目没做，留作练习）。

### 推理很慢

确认 KV cache 生效了。没有 cache 时生成 T 个词的复杂度是 O(T³)，有 cache 是 O(T²)。
本项目的 `greedy_decode` / `beam_search_decode` 都带 cache。

另外注意 `beam_size=4` 大约是 3 倍计算量；评测时还会按输入长度动态限制生成长度
（见 `inference.translate_dataset` 的 `adaptive_max_len`），
所以弱模型跑评测不会一路生成到 `max_len`。

## 环境

### Failed to initialize NumPy: No module named 'numpy'

这条警告可以**直接无视**。本项目完全不依赖 numpy（分词器、BLEU、可视化全是
标准库 + torch），它会冒出来只是因为 torch 自己想把张量转成 numpy 数组时找不到库。

### 装了 tensorboard 却没有日志

要显式打开开关：

```bash
python -m nmt.train --preset base --override train.tensorboard=true
tensorboard --logdir checkpoints/tensorboard
```

不装也没关系 —— `checkpoints/train_log.csv` 里什么都有，Excel 直接能画图。

### 数据准备太慢

BPE 训练是瓶颈（32k 词表大约 1 小时）。三个办法：

```bash
python -m nmt.corpus --vocab-size 16000        # 合并次数减半，时间大致减半
python -m nmt.corpus --bpe-train-pairs 50000   # 用更小的子集训分词器
python -m nmt.corpus --skip-download           # 语料已经下载过就别再下
```

## 最后：一个排查思路

遇到"效果不对"的问题，按这个顺序排除：

1. **跑测试**：`python -m pytest tests -q`。结构性 bug 大部分会被不变量测试抓住；
2. **对齐官方**：`python -m nmt.verify --preset tiny --dtype float64`。差异在 1e-6 量级说明结构没问题；
3. **看形状**：`python -m nmt.inspect --shapes`。数据流对不上时，形状最容易暴露；
4. **看译文**：`python -m nmt.evaluate --show 8`。数字会骗人，句子不会；
5. **回到最小**：把数据缩到 200 句、模型缩到 1 层，看它能不能过拟合。
   连 200 句都过拟合不了，问题一定在代码而不是超参。
