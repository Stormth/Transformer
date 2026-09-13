"""Transformer 从零实现：英译中机器翻译教学工程。

阅读顺序（也是这个包的模块顺序）：

    1. bpe.py        分词：把句子变成 id 序列
    2. corpus.py     语料：下载、清洗、切分（真实数据是脏的）
    3. dataset.py    批处理：分桶 + 动态 padding
    4. masks.py      屏蔽：哪些位置不许看
    5. attention.py  注意力：整个 Transformer 的心脏
    6. layers.py     残差 + LayerNorm + 前馈网络
    7. embedding.py  词嵌入 + 正弦位置编码
    8. encoder.py    编码器：把英文读成向量
    9. decoder.py    解码器：一边看英文，一边写中文
    10. model.py     组装 + 掩码工具
    11. loss.py      标签平滑交叉熵
    12. scheduler.py Noam 学习率
    13. decoding.py  贪心 / 束搜索
    14. bleu.py      BLEU / chrF 评测
    15. checkpoint.py 保存与恢复
    16. train.py     训练循环（AMP / 梯度累积 / 断点续训）
    17. evaluate.py  评测
    18. translate.py 命令行翻译
    19. inspect.py   打开模型内部看形状和注意力
    20. verify.py    和 PyTorch 官方 nn.Transformer 逐层对齐验证

只依赖 PyTorch：分词器、BLEU、可视化全部用标准库手写。
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
