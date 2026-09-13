# 第 9 章：评测

对应代码：`nmt/bleu.py`、`nmt/evaluate.py`

## BLEU 在算什么

BLEU 拿机器译文和参考译文比 n-gram（连续 n 个词）的重合度：

```
BLEU = BP × exp( Σ_{n=1..4} w_n × log p_n )

p_n  = 译文里"在参考译文中出现过"的 n-gram 占比（截断计数）
w_n  = 1/4，四个 n 均匀
BP   = 长度惩罚
```

三个必须知道的细节：

### 1. 截断计数

参考里 `der` 出现 3 次，译文里出现 10 次，最多只算命中 3 次。
否则重复同一个高频词就能刷高分数。

### 2. 长度惩罚 BP

```
BP = min(1, exp(1 - 参考长度 / 译文长度))
```

没有它的话，一句 `der`（1-gram 精度极高）就能骗到高分。译文比参考短就乘个小于 1 的因子，
比参考长则不奖不罚。

### 3. 切词方式必须固定

BLEU **不是跨切词方式可比的指标**。同一个模型，换成子词切分或者字符切分，分数就变了。
本项目英德两侧都用"转小写 + 只保留字母数字串"的简化 13a 分词，和 sacrebleu 的 `13a` 基本一致：

```
"Der Ingenieur, prüft!"  ->  ["der", "ingenieur", "prüft"]
```

注意 `ä ö ü ß` 都被当作字母保留下来。如果拿"只匹配 A-Za-z"的正则去切，
德语的变音字母会被当成标点丢掉，BLEU 会被人为压低 —— 这是个很容易犯的错误。

## BLEU 的局限（很重要）

* **不懂语义**：词全对了但格用错、语序错，BLEU 分不出来；
* **不惩罚"不像人话"**：语序全乱但词都对，BLEU 依然高；
* **对形态变化很苛刻**：德语名词复数、形容词词尾、动词变位错一个字母就算错，
  所以德语 BLEU 通常比英语之间互译低，不要和其它语言横向比；
* **短句极不稳定**：少一个词可能掉 20 分；
* **n-gram 命中数为 0 时 BLEU 直接是 0**：训练早期会长时间显示 0.00，
  这时候看 **chrF** 更有信息量。

`tests/test_bleu.py::test_very_short_sentences_score_zero` 把这个边界行为固定下来了：
一句 `Hallo` 完全译对，BLEU 仍然是 0 —— 因为 3-gram、4-gram 的命中数必然是 0。

## chrF：字符级 F 值

chrF 按**字符** n-gram（默认 1~6）算精确率和召回率的加权 F 值。

对德语特别友好：词形变化只影响几个字母，字符级重合仍然很高，
所以"主干译对了但词尾错了"这种情况，chrF 给出的分数比 BLEU 更接近直觉。
训练早期 BLEU 还是 0 的时候，chrF 往往已经在往上走 —— 用它判断"模型有没有在学"更灵敏。

本项目两个都报：

```
BLEU-4   18.42
chrF     52.31
长度比   1.03
```

## 用哪份测试集

| split | 来源 | 句数 | 说明 |
| --- | --- | --- | --- |
| dev | newstest2013 | 3000 | **调参、早停只看它** |
| test2014 | newstest2014 | 3003 | 论文报 27.3 BLEU 的那份，主报告用它 |
| test2017 / test2018 / test2019 | newstest2017~2019-ende | 3004 / 2998 / 1997 | 看域内稳定性 |

### 关于 newstest2014 的句数

你可能会看到某些论文写"newstest2014 有 2737 句"，而我们这里是 3003 句 ——
2737 是当年过滤掉空段和重复段之后的版本。两者的 BLEU 会差零点几。

所以：

* **和自己的消融实验横向比**（换结构、换数据量、换解码策略）→ 完全可靠；
* **和论文的绝对分数比** → 只能看量级，别抠小数点。

这不是偷懒，而是评测里应该养成的习惯：**先问"这个分数是在哪份数据、哪套切词、哪个版本上算的"。**

## 怎么用测试集才诚实

### 规则一：dev 调参，test 只报一次

dev（newstest2013）用来选超参、决定早停；test2014 只在最后跑一遍。
反复在 test 上调参，你的 test 分数就变成了 dev 分数。

### 规则二：必须报长度比

译文长短直接暴露"模型是不是在偷懒"。BLEU 高但长度比 0.6，说明它漏译了一大半内容。

### 规则三：一定要读译文

```bash
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --show 5 --save-pred preds.txt
```

BLEU 是压缩成一维的信号，它会漏掉重复、漏译、幻觉、数字错乱这类问题。
**看 20 句真实译文，比多看一个 BLEU 数字有用。**

## 完整评测流程

```bash
# 1. dev：看得比较勤，用来早停
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split dev

# 2. 主测试集 + 其它年份，看域内稳定性
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --beam-size 4
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2017 --beam-size 4

# 3. 解码策略对比
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --compare-decoders

# 4. 存下译文，人工检查
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --beam-size 4 \
    --save-pred preds/test2014.beam4.txt
```

## 自己动手

```bash
# 1. 看一段"BLEU 高但不通顺"的译文长什么样
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --show 8

# 2. 只评测长句 / 只评测短句，看 BLEU 差多少
#    这是理解"BLEU 为什么对长度敏感"的好办法

# 3. 自己实现 n-gram 精度，和 bleu.py 的结果对一遍
python -c "
from nmt.bleu import corpus_bleu
h = ['Der Ingenieur bespricht das Problem morgen in der Schule']
print(corpus_bleu(h, h).summary())
"
```
