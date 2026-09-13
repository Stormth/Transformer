# Transformer：从零实现英译中机器翻译

一份**逐行可读、能跑、能验证**的 Transformer 教学工程。
所有模型代码从零手写（不调用 `nn.Transformer`），只用 PyTorch：
分词器（BPE）、BLEU、注意力可视化全部用标准库实现。

任务：**英译中**。数据：**真实新闻平行语料**（OPUS 上的 News-Commentary + WMT 官方测试集）。

## 这个项目跟别的"手写 Transformer"有什么不同

1. **真实数据，真实分数**。用 WMT 官方测试集，BLEU 可以和论文、开源模型摆在一起比。
   很多教学项目用模板生成的合成语料，验证 BLEU 能到 99 —— 那个数字没有任何意义。
2. **能证明自己写对了**。`nmt/verify.py` 把权重搬进官方 `nn.Transformer`，逐层对比输出；
   `tests/` 里有 62 个不变量测试（因果性、KV cache 一致性、掩码、BLEU……）。
3. **能看见模型在看哪里**。`nmt/inspect.py` 输出张量形状追踪和注意力热力图（终端 + HTML）。
4. **工程该有的都有**：混合精度、梯度累积、断点续训、长度分桶动态 batching、
   束搜索 + KV cache、TensorBoard（可选）。

## 快速开始

```bash
pip install -r requirements.txt          # 只需要 torch（pytest 可选）

# 0) 先看一遍数据流（不用数据、不用训练，随机权重就能跑）
python examples/tour.py

# 1) 下载并清洗语料，训练 BPE 分词器（约 10 分钟，一次性）
python -m nmt.corpus --vocab-size 16000

# 2) 训练（4090 单卡，base 配置，几小时量级）
python -m nmt.train --preset base

# 3) 翻译
python -m nmt.translate --checkpoint checkpoints/best.pt --beam-size 4 \
    --text "The committee postponed the meeting until next Monday."

# 4) 在 WMT 官方测试集上评测
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --beam-size 4
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --compare-decoders

# 5) 打开模型内部：形状追踪 + 注意力热力图
python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes \
    --text "The committee postponed the meeting until next Monday." --html attention.html

# 6) 验证实现正确性
python -m nmt.verify --preset tiny --dtype float64 --show-internals
python -m pytest tests -q
```

**笔记本 / 显存小**：用 `--preset small`（d_model=256，3+3 层）或 `--preset tiny`（冒烟测试）。

## 训练结果

下面这份是**本机实测**（RTX 3050 Laptop 4 GB，`--preset small`，9.63M 参数，
News-Commentary 98k 句对，12 轮，45 分 48 秒）：

| 指标 | 数值 |
| --- | --- |
| dev（newsdev2017）BLEU | **3.60** |
| dev chrF | 5.72 |
| test2019 BLEU（贪心，全部 1980 句） | **3.27** |
| test2019 chrF | 5.04 |
| 长度比（译文/参考） | 1.02 |
| 评测耗时（test2019 全量，贪心） | 47 秒 |

学习曲线（dev BLEU）：

```
轮次   1     2     3     4     5     6     7     8     9    10    11    12
BLEU 0.00  0.13  0.32  0.36  0.88  1.00  1.90  2.22  2.12  2.99  3.23  3.60
```

曲线**还在上升就停了** —— 12 轮远远不够（这个配置在 4090 上跑 30 轮大约 1.5~2 小时，
base 配置（60M 参数）能到多少分留给你跑出来填在这张表里）。

解码策略对比（test2019 前 200 句）：

| 解码 | BLEU | chrF | 长度比 | 耗时 |
| --- | --- | --- | --- | --- |
| 贪心 | 3.58 | 5.32 | 0.95 | 4.5s |
| 束搜索 K=2 | **3.73** | 5.29 | 0.88 | 13.0s |
| 束搜索 K=4 | 3.62 | 5.10 | 0.82 | 23.1s |
| 束搜索 K=8 | 3.35 | 5.08 | 0.80 | 42.1s |

这个结果很值得琢磨：**束搜索没有救回一个没训够的模型**，K 越大反而越差，
而且译文越来越短（长度比从 0.95 掉到 0.80）。
原因是束搜索只是"更彻底地搜索模型自己给出的概率"，模型本身不靠谱时，
它会更加自信地走向错误的答案。想看到束搜索真正 +1~2 BLEU，得先把模型训到收敛。

> 参考量级：论文里的 base 模型在 WMT 英德上拿 27+ BLEU，用的是 450 万句对 + 8 卡训练。
> 本项目只有 10 万句对、单卡几小时，**目的是让你看懂每个环节，不是刷分**。

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
| [第 10 章](docs/10-experiments.md) | 22 个可以动手的实验 | — |
| [第 11 章](docs/11-faq.md) | 常见坑与排错 | — |

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
│   ├── train.py            训练循环
│   ├── evaluate.py         评测
│   ├── translate.py        命令行翻译
│   ├── inspect.py          形状追踪 + 注意力可视化
│   ├── verify.py           与官方 nn.Transformer 逐层对齐
│   └── synth.py            合成语料（只用于测试，不要用来评测）
├── docs/                   1~11 章讲解
├── tests/                  62 个不变量测试
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
| 参数量（词表 16k） | 约 0.2M | 约 8M | 约 60M |
| 适用场景 | CPU 冒烟测试 | 笔记本 GPU | 单卡 4090 |

训练默认：标签平滑 0.1、Noam 学习率（warmup 4000）、混合精度、动态 batching（8192 token/batch）、
长度分桶、每轮验证 dev BLEU 并保存 `best.pt`。

## 常用命令速查

```bash
# 数据
python -m nmt.corpus --vocab-size 16000                # 默认：News-Commentary + WMT
python -m nmt.corpus --extra un --extra ted2013        # 加不同领域的语料
python -m nmt.corpus --skip-download                   # 语料已下载好时跳过下载

# 训练
python -m nmt.train --preset base
python -m nmt.train --preset small --max-steps 500     # 限时冒烟
python -m nmt.train --preset base --resume auto        # 断点续训
python -m nmt.train --preset base --override train.lr_scale=0.5
python -m nmt.train --preset base --override train.tensorboard=true

# 翻译与评测
python -m nmt.translate --checkpoint checkpoints/best.pt --beam-size 4 --text "..."
python -m nmt.translate --checkpoint checkpoints/best.pt --file news.en --out news.zh
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --compare-decoders
python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --show 8 --save-pred out.txt

# 观察模型内部 / 验证实现
python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes --html attention.html
python -m nmt.verify --preset tiny --dtype float64 --show-internals
python -m pytest tests -q
```

## 数据来源与许可

* 训练语料：[News-Commentary v16](https://opus.nlpl.eu/News-Commentary/)（OPUS，en-zh）
* 验证/测试：[WMT-News v2019](https://opus.nlpl.eu/WMT-News/)（含 newsdev2017、newstest2017/2018/2019）

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
