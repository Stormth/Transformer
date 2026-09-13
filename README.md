# 从零实现 Transformer：中英翻译完整工程

一个"能跑通、能读、能改"的 Transformer 教学项目。全部代码从零手写（不调用 `nn.Transformer`），
覆盖**数据 → 分词 → 建模 → 训练 → 评估 → 推理 → 内部可视化**的完整链路。

设计目标只有三个：

1. **每个文件都能读懂**：核心文件 100~300 行，逐行中文注释 + 张量形状标注；
2. **几分钟能看到结果**：笔记本 GPU 上 5 分钟训练就能得到可用的翻译模型；
3. **能动手做实验**：所有关键设计（pre/post-norm、beam search、warmup、词表大小）都做成了开关。

只依赖 PyTorch。分词器（BPE）、BLEU、注意力可视化全部用标准库手写。

---

## 1. 五分钟跑通

```bash
# 0) 环境：本项目不需要 numpy / sacrebleu / tokenizers，只要 torch
pip install -r requirements.txt

# 1) 准备数据：生成 24000 句平行语料 + 训练 BPE 分词器
python -m src.build_data

# 2) 训练（笔记本 GPU 约 4 分钟，CPU 约 20 分钟）
python -m src.train

# 3) 翻译
python -m src.translate --checkpoint checkpoints/best.pt --text "他明天在学校讨论这个问题吗？"

# 4) 评估：测试集 BLEU + 贪心/束搜索对比
python -m src.evaluate --checkpoint checkpoints/best.pt --split test
python -m src.evaluate --checkpoint checkpoints/best.pt --compare-decoders

# 5) 打开模型内部：张量形状追踪 + 注意力热力图（生成 HTML）
python -m src.inspect_attention --checkpoint checkpoints/best.pt --text "他明天在学校讨论这个问题吗？"

# 6) 跑单元测试，确认你的改动没有破坏关键不变量
python -m pytest tests -q
```

> **Windows 提示**：如果终端里中文显示成乱码，先执行 `$env:PYTHONUTF8=1`（或 `chcp 65001`）。

训练脚本每一步都会打印 loss、困惑度、逐词准确率、梯度范数、学习率、吞吐量，
每轮在验证集上算 BLEU 并存 `checkpoints/best.pt`，同时把曲线写进 `checkpoints/train_log.csv`。

---

## 2. 目录结构

```
transformer-from-scratch/
├── src/
│   ├── bpe.py               纯 Python BPE：训练 / 编码 / 解码 / 保存
│   ├── synth.py             模板化平行语料生成器（含语序、形态、疑问句等真实翻译难点）
│   ├── build_data.py        第 1 步：生成语料 → 切分 → 训练分词器 → 落盘
│   ├── data.py              Dataset / 动态 padding / 长度分桶批采样
│   ├── masks.py             padding mask 与 causal mask（True = 屏蔽）
│   ├── attention.py         缩放点积注意力 + 多头注意力（含 KV cache）
│   ├── layers.py            残差 + LayerNorm + 位置前馈网络
│   ├── embedding.py         词嵌入 + 正弦位置编码
│   ├── encoder.py           编码器层 / 编码器堆叠
│   ├── decoder.py           解码器层 / 解码器堆叠（支持整句前向与单步增量）
│   ├── model.py             组装 Transformer + 掩码工具 + 贪心 / 束搜索
│   ├── loss.py              标签平滑交叉熵
│   ├── scheduler.py         Noam 学习率调度（warmup + 逆平方根衰减）
│   ├── bleu.py              纯 Python BLEU-4
│   ├── inference.py         文本进 / 文本出的统一推理接口
│   ├── train.py             第 2 步：训练循环、验证、checkpoint、CSV 日志
│   ├── translate.py         第 3 步：命令行翻译工具
│   ├── evaluate.py          第 4 步：BLEU 评估与解码策略对比
│   ├── inspect_attention.py 第 5 步：形状追踪 + 注意力可视化（HTML + 终端热力图）
│   └── config.py            所有超参数集中管理
├── data/
│   ├── natural/probe.tsv    80 句人工口语（模板之外，用来观察真实表现与未登录词）
│   └── processed/           build_data 的产物：train/dev/test.tsv、tokenizer.json、meta.json
├── tests/test_core.py       20 个正确性测试（掩码、缓存、束搜索、损失、能否过拟合……）
└── checkpoints/             训练产物：best.pt / last.pt / config.json / train_log.csv
```

---

## 3. 学习路线（按这个顺序读源码）

### 第 1 站：数据与分词 —— `src/bpe.py`、`src/data.py`、`src/build_data.py`

**目标**：搞清楚"一句话是怎么变成模型输入的一串数字的"。

要点：

* 按词分词会遇到没见过的词（OOV），按字符分词序列又太长，BPE 是折中方案；
* `pre_tokenize` 是个容易被忽略但很关键的设计：中文按字切、英文按词切、标点单独成单元；
* `</w>` 标记单元边界，解码时靠它把子词拼回原词；中文则是在最后去掉汉字之间的空格；
* 训练时必须维护 `pair -> 词频` 和 `pair -> 包含它的单元集合`，否则每轮重扫语料会慢到不可用；
* **词表 = 特殊符号 + 基础字符表 + 每条合并规则的产物**，缺了基础字符表，遇到生词就会变 `<unk>`；
* 分词器只能在**训练集**上训练，否则测试集词汇会通过词表泄漏。

动手实验：

```bash
python -m src.build_data --num-pairs 24000 --vocab-size 4000
# 看 data/processed/meta.json 里的 src_len_p95 / unk 率 / 词表大小
python -m src.build_data --vocab-size 200     # 词表极小会怎样？再看 unk 率变化
```

自检问题：为什么中文用字符级而不是词级？`<unt>`… 不对，是 `<unk>` 什么时候会出现？

### 第 2 站：掩码 —— `src/masks.py`

**目标**：把"哪些位置不许看"这件事彻底弄明白。这是 Transformer 里最容易写错、错了又最难发现的地方。

约定：`mask` 为 `True` 的位置被填成 `-1e9`（不是 `-inf`，因为整行被屏蔽时 `-inf` 会产出 `nan`）。

| 用在哪 | 需要什么掩码 | 形状 |
| --- | --- | --- |
| 编码器自注意力 | 源句 padding | `[B, 1, 1, Ts]` |
| 解码器自注意力 | 目标句 padding **或** 因果掩码 | `[B, 1, Tt, Tt]` |
| 解码器交叉注意力 | 源句 padding | `[B, 1, 1, Ts]` |

自检问题：如果因果掩码写成下三角会发生什么？（提示：`tests/test_core.py::test_decoder_is_causal` 会抓住它）

### 第 3 站：注意力 —— `src/attention.py`

**目标**：把公式 `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` 和代码逐行对上，并理解 KV cache。

要点：

* `[B,T,d_model]` 先 `view` 成 `[B,T,h,d_k]` 再 `transpose` 成 `[B,h,T,d_k]`，多头就是一次 batch 矩阵乘；
* 除以 `√d_k` 是为了把点积的方差拉回 1，防止 softmax 饱和；
* 三种用法：训练时全序列前向 / 增量解码时拼接历史 K/V（`append=True`）/ 交叉注意力复用恒定 K/V（`append=False`）；
* **踩坑记录**：第一步的 `past_kv` 是 `None`，"有没有缓存"不能作为"要不要返回缓存"的判断依据，
  否则第一步的 K/V 被丢掉，训练 loss 正常下降但推理输出乱码。所以用显式的 `use_cache` 参数控制。

动手实验：把 `attention()` 里的 `1/math.sqrt(d_k)` 删掉，看 loss 曲线怎么变。

### 第 4 站：积木 —— `src/layers.py`、`src/embedding.py`、`src/encoder.py`、`src/decoder.py`

**目标**：理解"一层"由哪些子层组成，以及 pre-norm / post-norm 的区别。

```
编码器层: 自注意力(双向) → FFN          （后接残差 + LayerNorm）
解码器层: 自注意力(因果) → 交叉注意力 → FFN
```

要点：

* 注意力负责"位置间交换信息"，FFN 负责"逐位置加工信息"，两者参数量约 1:2；
* 位置编码必须显式注入，因为注意力本身对位置置换等变；
* **踩坑记录**：初始化时不能把一维参数全部置零——LayerNorm 的 weight 默认是 1，
  置零后每个子层输出都会被清零，loss 永远卡在 `log(词表大小)` 附近。这个 bug 我们一起踩过了，
  见 `model.py::_reset_parameters`。

动手实验：`python -m src.train --norm-first --out-dir checkpoints_prenorm`，对比两者收敛速度。

### 第 5 站：组装、损失、调度 —— `src/model.py`、`src/loss.py`、`src/scheduler.py`

**目标**：走通一次完整前向，理解标签平滑与 warmup 为什么有用。

要点：

* 解码器输入 = 目标句右移一位（`shift_right`），标签 = 原目标句；这样"输入/标签"永远来自同一张量，不会错位；
* 标签平滑把 one-hot 目标改成 `(1-ε)` 与 `ε/(V-1)` 的混合，抑制过度自信；
* `<pad>` 必须从 loss 中排除，且分母用**有效 token 数**，否则不同长度的 batch 之间 loss 不可比；
* Noam：`lr = scale·d_model^(-1/2)·min(step^(-1/2), step·warmup^(-3/2))`，先线性升温再逆平方根衰减；
* 梯度裁剪 1.0 是 Transformer 的标配安全阀。

动手实验：`--label-smoothing 0` 与 `--label-smoothing 0.2` 各跑一遍，看验证 BLEU 的差别。

### 第 6 站：训练与推理 —— `src/train.py`、`src/inference.py`、`src/translate.py`、`src/bleu.py`

**目标**：理解 teacher forcing、验证流程、BPE 解码，以及贪心与束搜索的取舍。

要点：

* 训练可以整句并行（因果掩码保证不看未来），推理只能一个词一个词生成；
* 训练/推理不一致（exposure bias）是自回归模型的固有代价；
* 束搜索保留 beam 条候选，用**长度归一化后的整句分数**挑最终结果：
  `lp = ((5+len)/6)^α`，没有它模型会偏爱短句；
* 长度分桶 + token 预算组 batch 能显著减少 padding 浪费（见 `data.py::LengthBucketBatchSampler`）。

动手实验：

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --compare-decoders
```

### 第 7 站：打开模型看内部 —— `src/inspect_attention.py`

```bash
python -m src.inspect_attention --checkpoint checkpoints/best.pt \
    --text "他明天在学校讨论这个问题吗？" --head 0
```

会得到：一次前向的形状追踪、终端 ASCII 热力图、以及 `attention_report.html`（浏览器打开可看所有层）。

读图提示：

* 编码器自注意力：双向，能看到整句；
* 解码器自注意力：下三角，看不到未来；
* 交叉注意力：**行是生成的英文词，列是中文词**，这是最直观的"对齐"图，
  你会看到 `studies` 对应 `学习`、`at the office` 对应 `在公司` 这类对齐逐渐成型。

---

## 4. 张量形状总表

设 `B`=batch、`Ts`=源句长、`Tt`=目标句长、`d`=d_model、`h`=头数、`d_k=d/h`、`V`=词表大小。

| 阶段 | 代码 | 形状 |
| --- | --- | --- |
| 源 id | `src` | `[B, Ts]` |
| 词嵌入 | `src_embed(src)` | `[B, Ts, d]` |
| + 位置编码 | `positional(x)` | `[B, Ts, d]` |
| 编码器输出 memory | `encoder(x, src_mask)` | `[B, Ts, d]` |
| 多头拆分 | `_split_heads` | `[B, h, Ts, d_k]` |
| 注意力分数 | `QKᵀ/√d_k` | `[B, h, Ts, Ts]` |
| 注意力输出 | `softmax·V` | `[B, h, Ts, d_k]` → `[B, Ts, d]` |
| 目标 id（右移） | `shift_right(tgt)` | `[B, Tt]` |
| 解码器输出 | `decoder(...)` | `[B, Tt, d]` |
| logits | `generator(x)` | `[B, Tt, V]` |
| 增量解码单步 | `_step(token)` | 输入 `[B,1]` → logits `[B,V]` |
| KV 缓存 | `(k, v)` | `[B, h, T, d_k]` |

---

## 5. 实测结果

> 硬件：NVIDIA GeForce RTX 3050 Laptop（4GB 显存）；
> 模型 1.84M 参数（d_model=192, 4 头, 2+2 层, d_ff=384）；词表 607；
> 数据 22000 训练句 / 1000 验证 / 1000 测试；**20 个 epoch 约 3.5 分钟**（约 12 秒/epoch）。

### 5.1 完整训练（22000 句）

```bash
python -m src.train --out-dir checkpoints --epochs 20
python -m src.evaluate --checkpoint checkpoints/best.pt --split test
```

| 指标 | 数值 |
| --- | --- |
| 测试集 BLEU-4（贪心） | **99.69** |
| 验证集 BLEU（贪心，前 200 句） | 99.80 |
| 验证集 perplexity | 2.67 |
| 逐词准确率 | 99.9% |
| 总耗时 | 约 3.5 分钟（20 epoch） |

也就是说：**这套模板语料里的翻译规律，模型在 1 个 epoch 内就基本学会了**。
看到 BLEU 秒到 99 不要得意——这说明任务太简单（模板数据的信息量有限），
而不是模型有多强。想看"真正的学习曲线"，把训练数据缩小到 1500 句：

### 5.2 学习曲线（只用 1500 句训练）

```bash
python -m src.train --out-dir checkpoints_small --max-train-sentences 1500 --epochs 20 \
    --d-model 128 --d-ff 256 --warmup-steps 200 --lr-scale 0.7
```

| epoch | 训练 loss | 验证 loss | 验证 perplexity | **验证 BLEU** |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.976 | 5.278 | 195.99 | 0.00 |
| 2 | 4.742 | 4.066 | 58.32 | 0.00 |
| 3 | 3.651 | 2.977 | 19.63 | 8.12 |
| 4 | 2.684 | 2.126 | 8.38 | 27.51 |
| 5 | 1.957 | 1.565 | 4.78 | 55.90 |
| 6 | 1.469 | 1.292 | 3.64 | 77.92 |
| 7 | 1.253 | 1.134 | 3.11 | 93.70 |
| 9 | 1.149 | 1.098 | 3.00 | 97.60 |
| 13 | — | 1.038 | 2.82 | 99.25 |
| 20 | — | 1.025 | 2.79 | 98.55 |

两个值得注意的现象：

* 前两个 epoch BLEU 是 **0**，但 loss 已经在明显下降 —— 因为 BLEU-4 要求
  1~4 元语法都有命中，句子还说不通时它一定是 0。**看趋势要看 loss，不要只看 BLEU。**
* 逐词准确率对 BLEU 非常敏感：准确率 95% 的句子读起来仍然很别扭，
  但 BLEU 可能已经 90+。这也是 BLEU 常被吐槽的原因。

### 5.3 贪心 vs 束搜索

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --compare-decoders --limit 300
```

在**已经收敛**的模型上（300 句测试集子集）：

| 解码方式 | BLEU | 耗时 | 速度 |
| --- | ---: | ---: | ---: |
| 贪心 beam=1 | 99.91 | 1.2s | 257.6 句/秒 |
| 束搜索 beam=2 | 99.91 | 4.0s | 74.9 句/秒 |
| 束搜索 beam=4 | 99.91 | 8.4s | 35.8 句/秒 |
| 束搜索 beam=8 | 99.91 | 17.1s | 17.5 句/秒 |

结论：**束搜索不是万能的**。当模型对每个位置都很有信心时，它不会带来任何收益，
却要付出 15 倍的时间。束搜索的价值体现在"模型不确定"的场景——想亲眼看到，
就在只训练了 1~3 个 epoch 的 checkpoint 上再跑一次这条命令。

### 5.4 换个分布：人工口语句子（probe）

`data/natural/probe.tsv` 里的 80 句日常口语**没有参与训练**：

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --split probe --beam-size 4
```

| 数据 | BLEU-4 |
| --- | ---: |
| 模板测试集（同分布） | **99.69** |
| 人工口语（不同分布） | **4.56** |

模型会把没见过的句子硬塞进它学过的模板里：

| 源句 | 模型译文 | 参考译文 |
| --- | --- | --- |
| 你能帮我一个忙吗？ | do you read a help ? | can you do me a favor ? |
| 这家餐厅的菜很好吃。 | you are very ticket at home . | the food in this restaurant is delicious . |
| 请问怎么连无线网？ | how does she like the problem online ? | excuse me , how do i connect to the wifi ? |
| 我会尽快回复你。 | i i need the meeting . | i will reply to you as soon as possible . |

**这是整个项目最重要的一张表**：同一个模型、同一套代码，BLEU 从 99.7 掉到 4.6，
差别只来自数据分布。模型能力的天花板由数据决定，而不是由网络结构决定。

---

## 6. 这个项目与论文原版的差异（以及为什么）

| 项目 | 论文原版 | 本项目 | 原因 |
| --- | --- | --- | --- |
| 词表 | 32k BPE | 几百（由语料决定） | 合成语料本身词汇量就小 |
| 模型规模 | d=512, h=8, 6+6 层 | d=192, h=4, 2+2 层 | 笔记本 4GB 显存也能几分钟跑完 |
| 位置编码 | 正弦 | 正弦（同论文） | 便于对照论文 |
| 学习率 | warmup 4000 | warmup 800（可配） | 小语料用不了那么多步 |
| 优化器 | Adam(0.9, 0.98) | 同论文 | — |
| 束搜索 | beam 4, α=0.6 | 同论文，且支持 KV cache | 顺便理解大模型推理加速 |
| 数据 | WMT 千万句 | 模板生成 2.4 万句 | 保证离线、可复现、几分钟见效 |

---

## 7. 用你自己的数据训练

只要两列 TSV（源语言 TAB 目标语言），就能复用整条流水线：

```bash
# my_data/train.tsv  里每行形如：  我喜欢这本书<TAB>i like this book
python - <<'PY'
from src.bpe import BPE
from src.data import read_tsv, write_tsv
from src.synth import split_pairs

pairs = read_tsv("my_data/train.tsv")
train, dev, test = split_pairs(pairs, dev_size=2000, test_size=2000, seed=0)
write_tsv(train, "my_data/raw_train.tsv")
write_tsv(dev,   "my_data/raw_dev.tsv")
write_tsv(test,  "my_data/raw_test.tsv")
PY

python -m src.build_data --include-natural ...   # 或直接改用你自己的 build 流程
python -m src.train --data-dir my_data/processed --epochs 30 --vocab-size 32000
```

如果换成英德、日英等语言对，只需注意两处：

1. `src/bpe.py::pre_tokenize` 是否按你的语言正确切分（例如日语按字符切、泰语需要词典分词）；
2. `src/build_data.py` 里 `split_pairs` 的列顺序（第一列是源语言）。

---

## 8. 常见问题

**Q：终端中文乱码？**
Windows 下 Python 默认用 GBK 输出。执行 `$env:PYTHONUTF8=1` 或 `chcp 65001` 后重试。

**Q：导入 torch 时打印一堆 "A module that was compiled using NumPy 1.x …"？**
这是当前环境里 numpy 2.x 与 PyTorch 2.2 的兼容性提示，**不影响本项目**（本项目完全不使用 numpy）。
想彻底消除可以 `pip install "numpy<2"`，但那可能影响你环境里的其它包，请自行权衡。

**Q：显存不够（CUDA out of memory）？**
依次尝试：`--batch-size 32`、`--max-tokens 2048`、`--d-model 128 --d-ff 256`、`--enc-layers 2 --dec-layers 2`。

**Q：BLEU 一直是 0？**
前几个 epoch BLEU=0 是正常的：4-gram 精确率只要有一阶为 0，BLEU 就是 0。
看 `val_loss` 和逐词准确率是否在下降，以及终端打印的样例是否从"空串"变成"像样的句子"。

**Q：为什么验证时用贪心而不是束搜索？**
速度考虑。训练中每轮要快速拿到一个趋势信号，最后评估时才值得上束搜索。

**Q：`--include-natural` 有什么用？**
把 80 句人工口语混进训练集。默认不混，是为了让你看到"模板外的真实句子"上模型会露馅——
这本身就是重要的一课（数据分布决定模型能力边界）。

---

## 9. 练习题

**入门（改参数 + 看现象）**

1. `--vocab-size 200 / 1000 / 4000` 各训一次，词表大小如何影响 BLEU 与 `<unk>` 率？
2. `--warmup-steps 50 / 800 / 5000` 对比：去掉 warmup 会怎样？
3. `--label-smoothing 0 / 0.1 / 0.3` 对比验证 BLEU。
4. `--n-heads 1 / 4 / 8`（`d_model` 要能被头数整除）：单头会差多少？
5. 训练到 5/10/20 个 epoch 分别评估，画出 BLEU-epoch 曲线。

**进阶（改代码 + 写测试）**

6. 把 `--norm-first` 的 pre-norm 结果与 post-norm 对比，解释为什么大模型都用 pre-norm。
7. 把 FFN 的激活换成 `gelu` / `swish`（`--activation`），比较收敛；再实现 SwiGLU。
8. 把位置编码换成**可学习**的（`nn.Embedding(max_len, d_model)`），它能外推到更长序列吗？
9. 给训练加**梯度累积**（`--accum-steps`），用小显存模拟大 batch，验证效果是否接近。
10. 关闭 KV cache（把 `_step` 换成每步全量前向），对比生成 100 句的耗时，算出加速比。
11. 实现 `beam_size` 之外的解码策略：温度采样 / top-k 采样 / 长度惩罚调参。
12. 用 `token_accuracy` 之外的方式衡量质量：BLEU-1/2/3 分别多少？哪一阶拖后腿？
13. 实现 **reverse source**（训练时把中文反序输入）这个经典 trick，看 BLEU 变化。

**挑战（补全/扩展模块）**

14. 手写 `torch.autograd.Function` 版本的 `scaled_dot_product_attention`，与 `matmul` 版对拍。
15. 用 `torch.nn.utils.` 的 `weight_norm` 或自实现 **RoPE** 替换正弦位置编码。
16. 把编码器改成 BERT 式（双向、无解码器），做中文掩码语言模型预训练。
17. 实现**知识蒸馏**：用大模型（比如自己训的 512 维版本）教小模型。
18. 加入 **checkpoint averaging**（对最后 N 个 checkpoint 权重求平均），验证是否更稳。
19. 把 `noisy channel`/`back-translation` 的简化版本实现一遍：用模型把单语英文翻成中文，扩充训练集。
20. 用 `torch.compile` 或混合精度（`torch.cuda.amp`）加速训练，并验证数值一致性。

---

## 10. 延伸阅读

* 论文：*Attention Is All You Need* (Vaswani et al., 2017)
* 注释版实现：The Annotated Transformer（Harvard NLP）
* 子词分词：*Neural Machine Translation of Rare Words with Subword Units* (Sennrich et al., 2016)
* 束搜索与长度惩罚：*Google's Neural Machine Translation System* (Wu et al., 2016)
* 现代变体：Pre-LN、RMSNorm、RoPE、SwiGLU、GQA/MQA、FlashAttention

---

写完这些，你已经把 Transformer 的每个零件都亲手装过一遍。接下来最值得做的一件事是：
**打开 `attention_report.html`，找一句你感兴趣的句子，看看交叉注意力的对齐图**——
那一刻抽象公式会变成具体的东西。
