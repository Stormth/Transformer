"""第 5 步：把模型"打开"看内部——张量形状追踪 + 注意力可视化。

    python -m src.inspect_attention --checkpoint checkpoints/best.pt \
        --text "他明天在学校讨论这个问题吗？"

这个脚本做三件事（学习源码时最有用）：
    1. 打印一次前向传播中每一步的张量形状，帮你把公式和代码对齐；
    2. 在终端用 ASCII 热力图打印注意力矩阵；
    3. 生成一个 HTML 报告（attention_report.html），用颜色展示注意力，
       直接用浏览器打开即可，不需要 matplotlib / numpy。

读图提示：
    * 编码器自注意力是**双向**的，矩阵对称位置都有颜色；
    * 解码器自注意力是**下三角**的（生成第 t 个词时看不到第 t+1 个词）；
    * 交叉注意力的每一行是"生成这个英文词时，看了哪些中文词"，
      这是理解翻译对齐最直观的一张图。
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import List, Sequence

import torch

from .inference import load_model, translate_texts

RAMP = " .:-=+*#%@"


# ---------------------------------------------------------------------- #
# 终端的 ASCII 热力图
# ---------------------------------------------------------------------- #
def ascii_heatmap(matrix: torch.Tensor, row_labels: List[str], col_labels: List[str], width: int = 6) -> str:
    """matrix: [Tq, Tk]（已经取好 batch / head）。"""
    m = matrix.tolist()
    max_val = max((max(r) for r in m), default=1.0) or 1.0

    def short(label: str) -> str:
        # 去掉 </w> 标记后再截断，让中文单字能完整显示
        cleaned = label.replace("</w>", "")
        return (cleaned[: width - 1] if len(cleaned) > width - 1 else cleaned).ljust(width)

    header = " " * (width + 2) + "".join(short(c)[:width] for c in col_labels)
    lines = [header]
    for label, row in zip(row_labels, m):
        cells = []
        for v in row:
            idx = min(len(RAMP) - 1, int(v / max_val * (len(RAMP) - 1)))
            cells.append(RAMP[idx].center(width))
        lines.append(short(label) + " " + "".join(cells))
    return "\n".join(lines)


def pick_head(attn: torch.Tensor, head: int | None) -> torch.Tensor:
    """attn: [B, h, Tq, Tk] -> [Tq, Tk]（对 batch 取第 0 个，head=None 表示对所有头取平均）。"""
    a = attn[0]  # [h, Tq, Tk]
    return a.mean(dim=0) if head is None else a[head]


# ---------------------------------------------------------------------- #
# HTML 报告
# ---------------------------------------------------------------------- #
def _html_matrix(matrix: torch.Tensor, row_labels: List[str], col_labels: List[str], title: str) -> str:
    m = matrix.tolist()
    max_val = max((max(r) for r in m), default=1.0) or 1.0
    head_cells = "".join(
        f"<th class='col'>{html.escape(c)}</th>" for c in col_labels
    )
    rows = []
    for label, row in zip(row_labels, m):
        cells = []
        for v in row:
            alpha = min(1.0, (v / max_val) ** 0.7)
            cells.append(f"<td style='background:rgba(37,99,235,{alpha:.3f})'></td>")
        rows.append(f"<tr><th class='row'>{html.escape(label)}</th>{''.join(cells)}</tr>")
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<table class='heat'><thead><tr><th></th>{head_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def write_html_report(
    path: str | Path,
    source_tokens: List[str],
    target_tokens: List[str],
    encoder_maps: List[torch.Tensor],
    decoder_self_map: torch.Tensor,
    cross_maps: List[torch.Tensor],
    header_lines: Sequence[str],
) -> Path:
    path = Path(path)
    sections = []
    for i, m in enumerate(encoder_maps):
        sections.append(_html_matrix(m, source_tokens, source_tokens, f"编码器 第 {i + 1} 层 自注意力"))
    sections.append(_html_matrix(decoder_self_map, target_tokens, target_tokens, "解码器 最后一层 自注意力（因果）"))
    for i, m in enumerate(cross_maps):
        sections.append(_html_matrix(m, target_tokens, source_tokens, f"解码器 第 {i + 1} 层 交叉注意力（英文 -> 中文）"))

    doc = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Transformer 注意力可视化</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; color: #111827; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin: 24px 0 8px; }}
pre {{ background: #f3f4f6; padding: 12px; border-radius: 8px; overflow-x: auto; }}
table.heat {{ border-collapse: collapse; }}
table.heat td {{ width: 22px; height: 22px; border: 1px solid #e5e7eb; }}
table.heat th {{ font-weight: 500; color: #374151; font-size: 12px; }}
table.heat th.col {{ writing-mode: vertical-rl; transform: rotate(180deg); padding-bottom: 4px; }}
table.heat th.row {{ text-align: right; padding-right: 8px; white-space: nowrap; }}
section {{ margin-bottom: 28px; }}
</style></head><body>
<h1>Transformer 注意力可视化</h1>
<pre>{html.escape(chr(10).join(header_lines))}</pre>
{''.join(sections)}
</body></html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
@torch.no_grad()
def inspect(
    checkpoint: str,
    text: str,
    reference: str | None = None,
    beam_size: int = 1,
    head: int | None = None,
    report: str | Path = "attention_report.html",
    device: str = "auto",
) -> dict:
    model, tokenizer, cfg, dev = load_model(checkpoint, None, device)
    mc = cfg.model

    # ---- 1. 编码阶段 -------------------------------------------------- #
    src_ids = tokenizer.encode(text)
    src = torch.tensor([src_ids], dtype=torch.long, device=dev)
    src_mask = model.make_src_mask(src)
    src_tokens = [tokenizer.vocab[i] for i in src_ids]

    print("=" * 78)
    print("阶段 1：编码器")
    print("=" * 78)
    print(f"源句            : {text}")
    print(f"BPE token       : {src_tokens}")
    print(f"src id          : [B=1, Ts={src.size(1)}]")
    emb = model.src_embed(src)
    print(f"词嵌入          : {tuple(emb.shape)}   （乘了 sqrt(d_model)={mc.d_model ** 0.5:.2f}）")
    pe = model.positional(emb)
    print(f"+ 位置编码      : {tuple(pe.shape)}")
    memory = model.encoder(pe, src_mask)
    print(f"memory(编码输出): {tuple(memory.shape)}")

    encoder_maps = [
        pick_head(layer.self_attn.last_attn, head) for layer in model.encoder.layers
    ]

    # ---- 2. 解码阶段 -------------------------------------------------- #
    print()
    print("=" * 78)
    print("阶段 2：解码器")
    print("=" * 78)
    if reference is None:
        hyp, _ = translate_texts(
            model, tokenizer, [text], device=dev, beam_size=beam_size, max_len=64
        )[0]
        print(f"生成译文        : {hyp}")
        target_text = hyp
    else:
        target_text = reference
        print(f"使用给定参考译文: {reference}")

    tgt_ids = tokenizer.encode(target_text, add_eos=True)
    tgt_in = model.shift_right(torch.tensor([tgt_ids], dtype=torch.long, device=dev), model.bos_id)
    tgt_tokens = [tokenizer.vocab[i] for i in tgt_in[0].tolist()]
    logits = model(src, tgt_in)
    probs = logits.softmax(dim=-1)
    print(f"解码器输入 id   : {tuple(tgt_in.shape)}  {tgt_tokens}")
    print(f"logits          : {tuple(logits.shape)}   （[B, Tt, vocab]）")
    print(f"最大概率预测    : {[tokenizer.vocab[int(i)] for i in probs.argmax(-1)[0].tolist()]}")

    decoder_self_map = pick_head(model.decoder.layers[-1].self_attn.last_attn, head)
    cross_maps = [pick_head(layer.cross_attn.last_attn, head) for layer in model.decoder.layers]

    # ---- 3. 终端热力图 ------------------------------------------------ #
    label = "所有头平均" if head is None else f"第 {head} 个头"
    print()
    print(f"编码器 第 1 层 自注意力（{label}，行=Query，列=Key）")
    print(ascii_heatmap(encoder_maps[0], src_tokens, src_tokens))
    print()
    print(f"解码器 最后一层 交叉注意力（{label}，行=英文词，列=中文词）")
    print(ascii_heatmap(cross_maps[-1], tgt_tokens, src_tokens))

    # ---- 4. HTML 报告 ------------------------------------------------- #
    header_lines = [
        f"源句        : {text}",
        f"源 token    : {' '.join(src_tokens)}",
        f"目标 token  : {' '.join(tgt_tokens)}",
        f"模型        : d_model={mc.d_model}, heads={mc.n_heads}, "
        f"enc={mc.num_encoder_layers}, dec={mc.num_decoder_layers}",
        f"显示        : {label}",
    ]
    report_path = write_html_report(
        report, src_tokens, tgt_tokens, encoder_maps, decoder_self_map, cross_maps, header_lines
    )
    print()
    print(f"HTML 报告已生成：{report_path.resolve()}")
    print("（用浏览器打开可以看到所有层、所有头的注意力热力图）")

    return {
        "source_tokens": src_tokens,
        "target_tokens": tgt_tokens,
        "encoder_attn": encoder_maps,
        "decoder_self_attn": decoder_self_map,
        "cross_attn": cross_maps,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="检视 Transformer 内部：形状 + 注意力")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--text", default="他明天在学校讨论这个问题吗？")
    parser.add_argument("--reference", default=None, help="给定参考译文（不填则用模型自己生成的）")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--head", type=int, default=None, help="只看某个头；不填则对所有头取平均")
    parser.add_argument("--report", default="attention_report.html")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    inspect(
        args.checkpoint, args.text, args.reference, args.beam_size,
        args.head, args.report, args.device,
    )


if __name__ == "__main__":
    main()
