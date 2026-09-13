"""不用训练、不用数据，随机权重就能跑一遍完整前向 —— 看一眼每一步的形状。

    python examples/tour.py

它做的事：

  1. 现场训一个小 BPE，把英德各一句话变成 id；
  2. 建一个 tiny 模型（随机权重）；
  3. 手工走一遍前向，把每一步的张量形状打出来；
  4. 用贪心解码生成几个 token（随机模型，输出当然不通顺）；
  5. 打印注意力矩阵的一小块，看看"谁在看谁"。

想看点真的？训一个模型再看：

    python -m nmt.corpus --vocab-size 8000
    python -m nmt.train --preset small --max-steps 500
    python -m nmt.inspect --checkpoint checkpoints/best.pt --shapes --html attention.html
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接 `python examples/tour.py` 时，Python 只会把 examples/ 加进模块搜索路径，
# 找不到仓库根目录下的 nmt 包。这行把它补上（自己写脚本时经常要处理这件事）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from nmt.bpe import BPE
from nmt.config import tiny_config
from nmt.decoding import greedy_decode
from nmt.masks import make_decoder_self_attn_mask, make_encoder_attn_mask
from nmt.model import Transformer
from nmt.synth import toy_pairs


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(0)

    # ---------------------------------------------------------------- 1. 分词
    rule("第 1 步：分词（现场训一个小 BPE，词表 600）")
    pairs = toy_pairs(200)
    tokenizer = BPE.train(
        [text for pair in pairs for text in pair], vocab_size=600, min_frequency=1
    )
    print(tokenizer.summary())

    source = "The engineer will discuss the problem tomorrow at school."
    reference = "Der Ingenieur wird das Problem morgen in der Schule besprechen."
    src_ids = tokenizer.encode(source, add_eos=True)
    tgt_ids = tokenizer.encode(reference, add_bos=True, add_eos=True)
    print(f"\n英文：{source}")
    print(f"  分词：{tokenizer.tokenize(source)}")
    print(f"  id  ：{src_ids}")
    print(f"德语：{reference}")
    print(f"  分词：{tokenizer.tokenize(reference)}")
    print(f"  id  ：{tgt_ids}")
    print(f"\n解码回来：{tokenizer.decode(src_ids)} / {tokenizer.decode(tgt_ids)}")

    # ---------------------------------------------------------------- 2. 建模型
    rule("第 2 步：建一个 tiny 模型（随机权重）")
    config = tiny_config(vocab_size=len(tokenizer))
    model = Transformer(config.model).eval()
    print(model.describe())

    # ---------------------------------------------------------------- 3. 前向
    rule("第 3 步：一个 batch 里装了什么")
    src = torch.tensor([src_ids])
    tgt = torch.tensor([tgt_ids])
    src_mask = make_encoder_attn_mask(src, tokenizer.pad_id)
    decoder_input, labels = tgt[:, :-1], tgt[:, 1:]
    # 注意：解码器自注意力的掩码要按 **解码器输入**（长度 T-1）来造，
    # 而不是按完整目标句（长度 T）—— 行列分别对应 query 和 key 的位置。
    tgt_mask = make_decoder_self_attn_mask(decoder_input, tokenizer.pad_id)
    print(f"src            {tuple(src.shape)}   英文 id（含 <eos>）")
    print(f"tgt            {tuple(tgt.shape)}   德文 id（<bos> ... <eos>）")
    print(f"解码器输入      {tuple(decoder_input.shape)}   = tgt[:, :-1]")
    print(f"labels         {tuple(labels.shape)}   = tgt[:, 1:]")
    print(f"src_mask       {tuple(src_mask.shape)}   True = 不许看（padding）")
    print(f"tgt_mask       {tuple(tgt_mask.shape)}   True = padding 或未来")

    rule("第 4 步：逐层形状变化")
    with torch.no_grad():
        embedded = model.src_embedding(src)
        print(f"词嵌入         {tuple(src.shape)} -> {tuple(embedded.shape)}")
        print(f"  （乘了 sqrt(d_model)={config.model.d_model ** 0.5:.1f}，和位置编码量级对齐）")
        positioned = model.src_pos(embedded)
        print(f"位置编码       {tuple(positioned.shape)} -> {tuple(positioned.shape)}（相加，形状不变）")

        memory, _ = model.encode(src, src_mask)
        print(f"编码器         {tuple(positioned.shape)} -> {tuple(memory.shape)}")

        hidden, _ = model.decode(decoder_input, memory, tgt_mask, src_mask)
        print(f"解码器         {tuple(decoder_input.shape)} -> {tuple(hidden.shape)}")

        logits = model.generator(hidden)
        print(f"输出层         {tuple(hidden.shape)} -> {tuple(logits.shape)}")
        print(f"              （每一行是一个位置对全词表的打分，共 {config.model.tgt_vocab_size} 个词）")

    # ---------------------------------------------------------------- 4. 注意力
    rule("第 5 步：注意力权重（随机模型，看形状和归一化就够）")
    with torch.no_grad():
        _, encoder_weights = model.encode(src, src_mask, return_weights=True)
        _, decoder_weights = model.decode(
            decoder_input, memory, tgt_mask, src_mask, return_weights=True
        )
    first_layer = encoder_weights[0]
    print(f"编码器第 1 层自注意力：{tuple(first_layer.shape)}  = [batch, heads, 查询位置, 被看位置]")
    print(f"  每一行加起来 = {first_layer.sum(-1).flatten()[0].item():.4f}（softmax 之后必为 1）")
    cross = decoder_weights[0]["cross"]
    print(f"解码器第 1 层交叉注意力：{tuple(cross.shape)}  = [batch, heads, 德文位置, 英文位置]")
    print("  它的形状就是一张「翻译对齐表」：第 i 行 = 写第 i 个德语词时看英文的权重分布")

    # ---------------------------------------------------------------- 5. 解码
    rule("第 6 步：贪心解码（随机权重，输出必然不通顺）")
    with torch.no_grad():
        sequences = greedy_decode(
            model, src, src_mask,
            bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id,
            max_len=20,
        )
    print(f"生成的 id：{sequences[0]}")
    print(f"解码成文字：{tokenizer.decode(sequences[0])}")
    print("（想让它是人话，就得训练 —— 见 README 的快速开始）")

    rule("接下来")
    print("1. python -m nmt.corpus --vocab-size 8000        准备真实语料")
    print("2. python -m nmt.train --preset small --max-steps 500")
    print("3. python -m nmt.inspect  --checkpoint checkpoints/best.pt --shapes --html attention.html")
    print("4. python -m nmt.verify   --preset tiny --dtype float64   # 和官方实现对齐")
    print("5. python -m pytest tests -q                      # 64 个不变量测试")


if __name__ == "__main__":
    main()
