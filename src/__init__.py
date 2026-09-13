"""从零实现 Transformer（用于机器翻译）的教学项目。

模块分层（建议按此顺序阅读源码）：

    bpe.py        分词器：纯 Python 实现的 BPE
    data.py       数据集 / 动态 padding / 长度分桶采样
    masks.py      掩码：padding mask 与 causal mask
    attention.py  缩放点积注意力 + 多头注意力
    layers.py     前馈网络 / 残差连接 / LayerNorm
    embedding.py  词嵌入 + 正弦位置编码
    encoder.py    编码器层与编码器
    decoder.py    解码器层与解码器
    model.py      组装 Transformer + 掩码工具 + 权重初始化
    loss.py       标签平滑交叉熵
    scheduler.py  Noam 学习率调度器
    bleu.py       纯 Python 实现的 BLEU
    train.py      训练循环
    translate.py  推理：贪心 / 束搜索（含 KV cache）
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
