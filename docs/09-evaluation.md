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

参考里 "the" 出现 3 次，译文里出现 10 次，最多只算命中 3 次。
否则重复同一个高频词就能刷高分数。

### 2. 长度惩罚 BP

```
BP = min(1, exp(1 - 参考长度 / 译文长度))
```

没有它的话，一句"的"字（1-gram 精度极高）就能骗到高分。译文比参考短就乘个小于 1 的因子，
比参考长则不奖不罚。

### 3. 中文必须按字切

中文没有空格，BLEU 必须先分词。主流做法是**把每个汉字当成一个 token**
（本项目 `tokenize_for_bleu(text, lang="zh")` 就是这么做的）。

所以中文 BLEU 和英文 BLEU **不能横向比**：中文按字切，n-gram 更容易命中，
数值通常会低一些。

## BLEU 的局限（很重要）

* **不懂语义**："我吃了一个苹果" 和 "一个苹果吃了我" 的 BLEU 可能不低；
* **不惩罚"不像人话"**：语序全乱但词都对，BLEU 依然高；
* **短句极不稳定**：一句话少一个词，BLEU 可能掉 20 分；
* **n-gram 命中数为 0 时 BLEU 直接是 0**：训练早期（译文全是高频词）会长时间显示 0.00，
  这时候看 **chrF** 更有信息量。

## chrF：字符级 F 值

chrF 按**字符** n-gram（默认 1~6）算精确率和召回率的加权 F 值。

对中文特别友好：字本身就是有意义的单位，而且字符级重合比词级更"宽容"。
训练早期 BLEU 还是 0 的时候，chrF 往往已经在往上走 —— 用它判断"模型有没有在学"更灵敏。

本项目两个都报：

```
BLEU-4   12.34
chrF     45.67
长度比   1.03
```

## 怎么用测试集才诚实

### 规则一：dev 调参，test 只报一次

dev（newsdev2017）用来选超参、决定早停；test2017/2018/2019 只在最后跑一遍。
反复在 test 上调参，你的 test 分数就变成了 dev 分数。

### 规则二：必须报长度比

译文长短直接暴露"模型是不是在偷懒"。BLEU 高但长度比 0.6，说明它漏译了一大半内容。

### 规则三：一定要读译文

```bash
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --show 5 --save-pred preds.txt
```

BLEU 是压缩成一维的信号，它会漏掉重复、漏译、幻觉、数字错乱这类问题。
**看 20 句真实译文，比多看一个 BLEU 数字有用。**

## 完整评测流程

```bash
# 1. dev：看得比较勤，用来早停
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split dev

# 2. 三个官方测试集都跑一遍，看域内稳定性
for split in test2017 test2018 test2019; do
    python -m nmt.evaluate --checkpoint checkpoints/best.pt --split $split --beam-size 4
done

# 3. 解码策略对比
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --compare-decoders

# 4. 存下译文，人工检查
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --beam-size 4 \
    --save-pred preds/test2019.beam4.txt
```

## 关于"和论文对比"

本项目用的是 OPUS 整理过的 WMT-News 版本，和 sacrebleu 官方测试集相比：

* 清洗规则会丢掉少量句子（太长、编码坏行、对齐错位）；
* 因此句子数略有出入，BLEU 会有小数点后一两位的差别。

所以：

* **和自己的消融实验横向比**（换个结构、换个数据量）→ 完全可靠；
* **和论文的绝对分数比** → 只能看量级，别抠小数点。

这不是偷懒，而是评测里应该养成的习惯：**先问"这个分数是在哪份数据、哪套参数上算的"。**

## 自己动手

```bash
# 1. 看一段"BLEU 高但不通顺"的译文长什么样（束搜索 K=1 时更容易出现）
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --show 8

# 2. 只评测长句 / 只评测短句，看 BLEU 差多少
#    这是理解"BLEU 为什么对长度敏感"的好办法

# 3. 自己实现 n-gram 精度，和 bleu.py 的结果对一遍
python -c "
from nmt.bleu import corpus_bleu
h=['他明天去学校']
r=['他明天去学校']
print(corpus_bleu(h,r).summary())
"
```
