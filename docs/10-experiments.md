# 第 10 章：可以做的实验清单

读完代码只是开始。这一章列出**改哪个开关、观察什么现象、预期会怎样**，
按"性价比"排序 —— 前面的实验便宜且信息量大。

## 一、先立基线

没有基线，所有对比都没有意义。先跑一次标准配置并把结果记下来：

```bash
python -m nmt.train --preset base --override train.epochs=10 --save-dir checkpoints/base
python -m nmt.evaluate --checkpoint checkpoints/base/best.pt --split test2014 --beam-size 4
```

记录四件事：dev BLEU、test2014 BLEU、chrF、长度比。之后每次实验只改一个变量。

## 二、便宜且涨点明显的（先做这些）

| # | 实验 | 命令 | 预期 |
| --- | --- | --- | --- |
| 1 | **束搜索** | `--compare-decoders` | 模型收敛后 K=4 通常 +1~2 BLEU，K=8 收益变小 |
| 2 | **训练轮数** | 改 `train.epochs` | 最容易被低估的一项：曲线还在涨就别停 |
| 3 | **数据量** | `python -m nmt.corpus --extra europarl` | 训练集从 29 万涨到 59 万句，通常明显涨点 |
| 4 | **学习率** | `--override train.lr_scale=0.3/1.0/3.0` | 影响最大的超参，峰值差 3 倍结果差很多 |
| 5 | **词表大小** | `--vocab-size 16000 / 32000 / 64000` | 德语是形态语言，词表通常越大越好，但训练更慢 |

## 三、理解结构本身

| # | 实验 | 命令 | 预期 |
| --- | --- | --- | --- |
| 6 | **pre-norm vs post-norm** | `--override model.norm_first=true/false` | pre-norm 前几百步更稳；post-norm 收敛后可能略好 |
| 7 | **多头 vs 单头** | `--override model.n_heads=1` | 参数量几乎不变，BLEU 通常掉 1~3 分 |
| 8 | **层数** | `--override model.num_encoder_layers=1/3/6` | 1 层明显不够，6 层之后边际收益递减 |
| 9 | **去掉位置编码** | 改 `embedding.py` 里的 `pos` 为恒等 | BLEU 崩掉 —— 位置编码必要性的最好证明 |
| 10 | **去掉交叉注意力的掩码** | 把 `make_cross_attn_mask` 直接返回 None | 变差且不稳定，验证掩码的必要性 |
| 11 | **权重共享** | `--override model.tie_embeddings=false` | 参数多一份，小数据上通常更差 |
| 12 | **嵌入缩放** | `--override model.scale_embedding=false` | 训练初期更难，收敛更慢 |

## 四、训练技巧

| # | 实验 | 命令 | 预期 |
| --- | --- | --- | --- |
| 13 | **warmup 长度** | `--override train.warmup_steps=200/4000` | 太短会震荡，太长浪费步数 |
| 14 | **标签平滑** | `--override train.label_smoothing=0.0/0.1/0.2` | 0 时 loss 更低但 BLEU 可能更差 |
| 15 | **梯度累积** | `--override train.accum_steps=4` | 等效大 batch，通常略涨点但更慢 |
| 16 | **混合精度** | `--override train.amp_dtype=bfloat16` | 4090 上速度相近，数值更稳 |
| 17 | **dropout** | `--override model.dropout=0.3` | 数据少时防过拟合，数据多时反而变差 |
| 18 | **梯度裁剪阈值** | `--override train.clip_grad=0.5/5.0` | 影响训练稳定性，太大等于没有 |

## 五、数据侧的实验（往往收益最大）

| # | 实验 | 怎么做 | 预期 |
| --- | --- | --- | --- |
| 19 | 放宽/收紧清洗规则 | 改 `CleanConfig` | 更干净的数据通常比更多数据更划算 |
| 20 | 长度过滤的阈值 | 改 `--max-src-len / --max-tgt-len` | 过滤长句会损失信息，但训练更稳 |
| 21 | 域外数据的影响 | `--extra ted2013` / `--extra multiun` | 常见的"训练 loss 更低、测试 BLEU 更低" |
| 22 | 反向数据增强 | 把源和目标互换训一遍 | 想清楚再动手 —— 这会把英译德变成德译英 |
| 23 | **复合词该怎么办** | 见下方说明 | 德语的核心难题 |

### 关于第 23 条：德语复合词与词表预算

德语靠复合词造词，理论上可以无限长：

```
Wirtschaftswachstum           经济 + 增长
Geschwindigkeitsbegrenzung    速度 + 界限 + 限定
Bundeskanzleramt              联邦 + 总理 + 局
```

词表里不可能装下所有复合词，所以它们必然被 BPE 拆开。**拆得对不对，直接影响 BLEU。**

三种改法（都值得动手比）：

```bash
# a) 加大词表：给复合词更多整体入表的机会（BPE 训练时间大约翻倍）
python -m nmt.corpus --vocab-size 64000 --skip-download

# b) 缩小词表：让复合词拆得更碎，序列变长但 <unk> 更少
python -m nmt.corpus --vocab-size 16000 --skip-download

# c) 英德各训一个 BPE（两张词表）
#    需要放弃权重共享，模型也要改成两套嵌入，代码改动大约 30 行
```

**判断标准不是"哪种词表更大"，而是"编码后的平均序列长度"和"测试集上的 <unk> 率"** ——
同一个模型、同样的句子，序列短一半，训练就快一倍。

看分词的直观办法：

```bash
python -c "
from nmt.bpe import BPE
t = BPE.load('data/ready/vocab.json')
for w in ['Wirtschaftswachstum', 'Geschwindigkeitsbegrenzung']:
    print(w, '->', t.tokenize(w))
"
```

## 六、读代码时顺手能做的验证

```bash
# 结构正确性：和官方实现逐层对齐
python -m nmt.verify --preset tiny --dtype float64 --show-internals

# 缓存正确性：增量解码 == 整句前向
python -m pytest tests/test_model.py -q

# 解码正确性：beam_size=1 == 贪心
python -m pytest tests/test_decoding.py -q

# 全部测试
python -m pytest tests -q
```

## 七、怎么做一份像样的实验记录

推荐一个表格，每次实验一行：

| 日期 | 配置 | 训练集 | dev BLEU | test2014 BLEU | chrF | 长度比 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 基线 | base, 10 epoch | NC 29 万 | — | — | — | — | 基线 |
| +beam4 | 同上 | NC 29 万 | — | — | — | — | 束搜索 +? |
| +europarl | 同上 | NC+EP 59 万 | — | — | — | — | 数据翻倍 +? |

三条经验：

1. **一次只改一个变量**，否则涨了跌了都不知道为什么；
2. **记录长度比**，它能解释很多"BLEU 为什么变了"；
3. **保存译文**，一个月后你还想回看某个实验结果时，BLEU 数字帮不上忙。
