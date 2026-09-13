# 第 0 章：整个工程在做什么

## 一句话

把一句英文变成一句德语。中间要经过这些步骤，每一步都有对应的文件和章节：

```
原始语料（OPUS 上的平行句对）
   │  ① 下载 + 清洗           corpus.py           → 第 1 章
   ▼
干净的句对（英文一句、德文一句，一一对应）
   │  ② 训练分词器 + 编码      bpe.py / corpus.py  → 第 2 章
   ▼
整数 id 序列（"Der Ingenieur …" → [1, 306, 88, ...]）
   │  ③ 组 batch（分桶 + 动态 padding）dataset.py → 第 2 章
   ▼
模型输入张量 src [B,S] / tgt [B,T]
   │  ④ 前向                        model.py       → 第 3~6 章
   │     词嵌入 + 位置编码           embedding.py
   │     编码器（英文 → memory）      encoder.py
│     解码器（一边看 memory 一边写德文）decoder.py
   │     掩码（哪些位置不许看）        masks.py
   ▼
logits [B, T, 词表大小]
   │  ⑤ 算损失、反向传播、更新参数    loss.py / train.py → 第 7 章
   ▼
训练好的权重（checkpoints/best.pt）
   │  ⑥ 解码：一个词一个词生成译文    decoding.py    → 第 8 章
   ▼
德语译文
   │  ⑦ 评测：BLEU / chrF            bleu.py        → 第 9 章
   ▼
一个数字，告诉你好不好
```

## 为什么用"英译德 + 新闻域"

* **英译德**：这是论文《Attention Is All You Need》用的语言对，你算出来的 BLEU 可以直接和它报的 27.3 对齐（当然量级会不同，见第 9 章）。
* **数据够多**：News-Commentary 的英德方向有 29.4 万句对，比小语种方向多一个数量级；不够还能加 Europarl 的近 200 万句对。
* **分词器能正常工作**：德语是"空格分词 + 拉丁字母"，BPE 把复合词拆成有意义的子词，句子长度稳定在 30 个 token 左右。序列长度直接决定训练成本（注意力是 O(T²)）。
* **新闻域**：WMT 官方每年都发布 newstest 测试集，你的 BLEU 可以和论文、开源模型的公开分数对比。自己随手切一个测试集，分数再高也不知道算高还是低。

## 代码地图

| 文件 | 行数级别 | 干什么 | 对应章节 |
| --- | --- | --- | --- |
| `nmt/bpe.py` | 300 | 手写 BPE 分词器 | 2 |
| `nmt/corpus.py` | 450 | 下载、清洗、切分、编码 | 1 |
| `nmt/dataset.py` | 200 | 长度分桶 + 动态 padding | 2 |
| `nmt/masks.py` | 90 | 三种掩码 | 6 |
| `nmt/attention.py` | 180 | 缩放点积注意力 + 多头 + KV cache | 4 |
| `nmt/layers.py` | 90 | 残差 / LayerNorm / 前馈网络 | 5 |
| `nmt/embedding.py` | 80 | 词嵌入 + 正弦位置编码 | 3 |
| `nmt/encoder.py` | 80 | 编码器层与堆叠 | 5 |
| `nmt/decoder.py` | 120 | 解码器层与堆叠（三个子层） | 5 |
| `nmt/model.py` | 230 | 组装 + 掩码工具 | 5 |
| `nmt/loss.py` | 90 | 标签平滑交叉熵 | 7 |
| `nmt/scheduler.py` | 90 | Noam 学习率 | 7 |
| `nmt/decoding.py` | 180 | 贪心 / 束搜索（带 KV cache） | 8 |
| `nmt/bleu.py` | 200 | BLEU-4 与 chrF | 9 |
| `nmt/checkpoint.py` | 130 | 保存 / 恢复 | 7 |
| `nmt/train.py` | 380 | 训练循环（AMP / 梯度累积 / 续训） | 7 |
| `nmt/inference.py` | 150 | 推理入口 | 8 |
| `nmt/evaluate.py` | 160 | 评测脚本 | 9 |
| `nmt/translate.py` | 120 | 命令行翻译 | 8 |
| `nmt/inspect.py` | 260 | 形状追踪 + 注意力热力图 | 4 |
| `nmt/verify.py` | 300 | 和官方实现逐层对齐 | 6 |

## 建议的阅读顺序

1. **先跑通再看代码**。按 README 的"五分钟跑通"跑一遍，心里有底。
2. 按章节顺序读：数据 → 分词 → 嵌入 → 注意力 → 编解码器 → 掩码 → 训练 → 解码 → 评测。
3. 每一章都配了"自己动手"的小实验，改一个数字、看一个现象。**只读不跑，是学不会的。**
4. 卡住的时候看 `docs/11-faq.md`，里面收集了这类项目最常见的坑。

## 一个心态上的提醒

手写 Transformer 最大的价值不是"能跑"，而是**你能指着每一行说出它在算什么**。
所以本项目刻意做了两件看起来多余的事：

* `nmt/verify.py`：和 PyTorch 官方实现逐层对比，证明自己没写错；
* `nmt/inspect.py`：把注意力画出来，看看模型到底在看哪儿。

这两件事把"我觉得应该是对的"变成"我能证明它是对的"。
