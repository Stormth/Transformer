# Transformer：从零实现英德机器翻译

一份**逐行可读、能跑、能验证**的 Transformer 教学工程。
所有模型代码从零手写（不调用 `nn.Transformer`），只用 PyTorch：
分词器（BPE）、BLEU、注意力可视化全部用标准库实现。

任务：**英译德**（论文《Attention Is All You Need》用的就是这个语言对）。
数据：**真实新闻平行语料**（OPUS 上的 News-Commentary + WMT 官方测试集）。

## 为什么选英译德

1. **数据量足够**。News-Commentary 的英德方向有 **29.4 万句对**，比小语种方向多得多；
   不够还可以加 Europarl（近 200 万句对）。
2. **分词器能正常工作**。德语是"空格分词 + 拉丁字母"，BPE 能把复合词
   （Wirtschaftswachstum → Wirtschaft + Wachstum）拆成有意义的子词，
   句子长度稳定在 30 个 token 左右。**注意力的开销是 O(T²)，
   序列长度减半意味着训练成本降到四分之一。**
3. **分数可以和论文直接比**。测试集用 WMT 的 newstest2014 ——
   论文报的 27.3 BLEU 就是在它上面算的。

## 这个项目跟别的"手写 Transformer"有什么不同

1. **真实数据，真实分数**。用 WMT 官方测试集，BLEU 可以和论文、开源模型摆在一起比。
   很多教学项目用模板生成的合成语料，验证 BLEU 能到 99 —— 那个数字没有任何意义。
2. **能证明自己写对了**。`nmt/verify.py` 把权重搬进官方 `nn.Transformer`，逐层对比输出；
   `tests/` 里有 64 个不变量测试（因果性、KV cache 一致性、掩码、BLEU……）。
3. **能看见模型在看哪里**。`nmt/inspect.py` 输出张量形状追踪和注意力热力图（终端 + HTML）。
4. **工程该有的都有**：混合精度、梯度累积、断点续训、长度分桶动态 batching、
   束搜索 + KV cache、TensorBoard（可选）。

## 快速开始

```bash
pip install -r requirements.txt          # 只需要 torch（pytest 可选）

# 0) 先看一遍数据流（不用数据、不用训练，随机权重就能跑）
python examples/tour.py

# 1) 下载并清洗语料，训练 BPE 分词器（一次性；32k 词表大约 1 小时）
#    服务器上想先快速跑通，可以加 --vocab-size 16000 把这一步缩短一半
python -m nmt.corpus

# 2) 训练（4090 单卡，base 配置，几小时量级；支持断点续训）
python -m nmt.train --preset base

# 3) 翻译
python -m nmt.translate --checkpoint checkpoints/best.pt --beam-size 4 \
    --text "The committee postponed the meeting until next Monday."

# 4) 在 WMT 官方测试集上评测（newstest2014 就是论文用的那份）
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --beam-size 4
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --compare-decoders

# 5) 打开模型内部：形状追踪 + 注意力热力图
python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes \
    --text "The committee postponed the meeting until next Monday." --html attention.html

# 6) 验证实现正确性
python -m nmt.verify --preset tiny --dtype float64 --show-internals
python -m pytest tests -q
```

**笔记本 / 显存小**：用 `--preset small`（d_model=256，3+3 层）或 `--preset tiny`（冒烟测试）。

## 复现论文（8 卡）

如果你想复现《Attention Is All You Need》的英德结果（newstest2014 上 27.3 BLEU），
完整流程写在 [docs/12-reproduce.md](docs/12-reproduce.md)，命令是这三条：

```bash
# 1) 论文设置的数据：Europarl v7 + Common Crawl + News Commentary v9，456 万句对（1~2 小时）
python -m nmt.corpus --recipe wmt14

# 2) 8 卡训练（论文 base：100k 步、50k token 批、warmup 4000；8×4090 约 3~6 小时）
torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --max-steps 100000

# 3) 论文口径的评测：最后 5 个 checkpoint 平均 + beam 4 + 区分大小写 BLEU
python -m nmt.average --checkpoint-dir checkpoints --num 5
python -m nmt.evaluate --checkpoint checkpoints/averaged.pt --split test2014 --beam-size 4 --case-sensitive
```

`--preset paper-big` 是论文的大模型（d_model 1024、16 头、2.13 亿参数，28.4 BLEU，8×4090 约 10~20 小时）。

## 数据集划分

| split | 来源 | 句数 | 用途 |
| --- | --- | --- | --- |
| train | News-Commentary v16 (de-en) | 294,498（清洗后会少一些） | 训练 |
| dev | newstest2013 | 3000 | 调参、早停 |
| test2014 | newstest2014 | 3003 | **和论文对比的主力测试集** |
| test2017 | newstest2017-ende | 3004 | 域内稳定性 |
| test2018 | newstest2018-ende | 2998 | 域内稳定性 |
| test2019 | newstest2019-ende | 1997 | 域内稳定性 |

## 训练结果

这份代码还没有在 4090 上跑过完整实验，下面这张表留给你填（把 `—` 换成实测值即可）：

| 配置 | 训练数据 | dev BLEU | test2014 BLEU | chrF | 长度比 | 训练时长 |
| --- | --- | --- | --- | --- | --- | --- |
| small（d_model=256，3+3 层） | NC 全量 | — | — | — | — | — |
| base（d_model=512，6+6 层） | NC 全量 | — | — | — | — | — |
| base + beam 4 | NC 全量 | — | — | — | — | — |

训练时每轮的 dev BLEU 都会写进 `checkpoints/train_log.csv`，可以直接拿去画学习曲线。

> 参考量级（**估算，不是实测**）：论文的 base 模型在 newstest2014 上是 27.3 BLEU，
> 但那是 450 万句对 + 8 卡训练。单卡 4090 + 29 万句对跑几小时，合理预期在 12~20 之间；
> 关键看训练轮数够不够 —— 如果曲线还在涨，说明还能再训。

## 学习路线

按这个顺序读代码，每一章都配了"自己动手"的实验：

| 章节 | 主题 | 代码 |
| --- | --- | --- |
| [第 0 章](docs/00-overview.md) | 整个工程在做什么 | 全局地图 |
| [第 1 章](docs/01-data.md) | 平行语料、数据清洗 | `corpus.py` |
| [第 2 章](docs/02-tokenization.md) | 手写 BPE、长度分桶 | `bpe.py`、`dataset.py` |
| [第 3 章](docs/03-embedding.md) | 词嵌入、位置编码 | `embedding.py` |
| [第 4 章](docs/04-attention.md) | 注意力、多头、KV cache | `attention.py` |
| [第 5 章](docs/05-encoder-decoder.md) | 编码器/解码器/残差 | `encoder.py`、`decoder.py`、`layers.py` |
| [第 6 章](docs/06-masks.md) | 掩码与"怎么证明写对了" | `masks.py`、`verify.py` |
| [第 7 章](docs/07-training.md) | 损失、Noam、AMP、续训 | `loss.py`、`scheduler.py`、`train.py` |
| [第 8 章](docs/08-decoding.md) | 贪心、束搜索、长度惩罚 | `decoding.py` |
| [第 9 章](docs/09-evaluation.md) | BLEU / chrF 与诚实的评测 | `bleu.py`、`evaluate.py` |
| [第 10 章](docs/10-experiments.md) | 可以动手的实验清单 | — |
| [第 11 章](docs/11-faq.md) | 常见坑与排错 | — |
| [第 12 章](docs/12-reproduce.md) | **复现论文的英德结果（8 卡）** | `average.py`、DDP |

## 项目结构

```
.
├── nmt/                    模型与工具（每个文件对应一章）
│   ├── bpe.py              手写 BPE 分词器
│   ├── corpus.py           下载 / 清洗 / 切分 / 编码
│   ├── dataset.py          长度分桶 + 动态 padding
│   ├── masks.py            三种掩码
│   ├── attention.py        缩放点积注意力 + 多头 + KV cache
│   ├── layers.py           残差 / LayerNorm / 前馈网络
│   ├── embedding.py        词嵌入 + 正弦位置编码
│   ├── encoder.py          编码器
│   ├── decoder.py          解码器（三个子层）
│   ├── model.py            组装 + 掩码工具
│   ├── loss.py             标签平滑交叉熵
│   ├── scheduler.py        Noam 学习率
│   ├── decoding.py         贪心 / 束搜索
│   ├── bleu.py             BLEU-4 / chrF
│   ├── checkpoint.py       保存与恢复
│   ├── average.py          最后 N 个 checkpoint 权重平均（论文的做法）
│   ├── train.py            训练循环
│   ├── evaluate.py         评测
│   ├── translate.py        命令行翻译
│   ├── inspect.py          形状追踪 + 注意力可视化
│   ├── verify.py           与官方 nn.Transformer 逐层对齐
│   └── synth.py            合成语料（只用于测试，不要用来评测）
├── docs/                   0~11 章讲解
├── examples/tour.py        不需要数据和训练的形状导览
├── tests/                  64 个不变量测试
├── data/                   raw/（下载的语料）ready/（处理后的张量）—— 不入库
└── checkpoints/            训练产物 —— 不入库
```

## 模型与训练配置

| | tiny | small | base |
| --- | --- | --- | --- |
| d_model | 64 | 256 | 512 |
| 注意力头数 | 4 | 4 | 8 |
| 编码器/解码器层数 | 2 / 2 | 3 / 3 | 6 / 6 |
| d_ff | 128 | 1024 | 2048 |
| 参数量（词表 32k） | 约 2.2M | 约 14M | 约 60M |
| 适用场景 | CPU 冒烟测试 | 笔记本 GPU | 单卡 4090 |

训练默认：标签平滑 0.1、Noam 学习率（warmup 4000）、混合精度、动态 batching、长度分桶、
每轮验证 dev BLEU 并保存 `best.pt`。

## 常用命令速查

```bash
# 数据
python -m nmt.corpus                                  # 默认：News-Commentary + WMT，32k 词表
python -m nmt.corpus --vocab-size 16000               # 数据准备时间减半
python -m nmt.corpus --extra europarl                 # 加 30 万句欧洲议会语料
python -m nmt.corpus --extra ted2013 --extra multiun  # 换领域做对比实验
python -m nmt.corpus --skip-download                  # 语料已下载好时跳过下载

# 训练
python -m nmt.train --preset base
python -m nmt.train --preset small --max-steps 500    # 限时冒烟
python -m nmt.train --preset base --resume auto       # 断点续训（要用和原来相同的 --preset）
python -m nmt.train --preset base --override train.lr_scale=0.5
python -m nmt.train --preset base --override train.tensorboard=true

# 翻译与评测
python -m nmt.translate --checkpoint checkpoints/best.pt --beam-size 4 --text "..."
python -m nmt.translate --checkpoint checkpoints/best.pt --file news.en --out news.de
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --compare-decoders
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2014 --show 8 --save-pred out.txt

# 观察模型内部 / 验证实现
python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes --html attention.html
python -m nmt.verify --preset tiny --dtype float64 --show-internals
python -m pytest tests -q
```

## 数据来源与许可

* 训练语料：[News-Commentary v16](https://opus.nlpl.eu/News-Commentary/)（OPUS，英德方向）
* 验证/测试：[WMT-News v2019](https://opus.nlpl.eu/WMT-News/)（含 newstest2008~2019）

两份语料都来自 OPUS 开放平台，各自带有原始许可（通常是 CC-BY / 允许研究使用）。
本项目只做教学演示；商用请自行核对原始语料的许可条款。

## 环境要求

* Python 3.9+
* PyTorch 2.0+（有 CUDA 会快很多，纯 CPU 也能跑 `--preset tiny`）
* 可选：`pytest`（测试）、`tensorboard`（曲线）

本项目**刻意不依赖 numpy / sacrebleu / tokenizers**：分词、BLEU、可视化都自己写，
一来没有版本兼容问题，二来你能看清每一步到底做了什么。

## 许可证

MIT
