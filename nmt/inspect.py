"""打开模型内部：张量形状追踪 + 注意力热力图。

这是这个项目里最"教学"的一个脚本。两件事：

1. **形状追踪**：给每个子模块挂 forward hook，把数据流过时每一步的形状打出来。
   论文里的图看得再多，也不如亲眼看着 [2, 9] 怎么变成 [2, 9, 512]，
   再切成 [2, 8, 9, 64] 来得实在。

2. **注意力热力图**：把"哪个位置在看哪个位置"画出来。
   编码器自注意力能看到模型把哪些词圈在一起，
   解码器交叉注意力能看到"写这个中文字时在看英文的哪个词" —— 翻译对齐关系一目了然。

输出：
    * 终端里的字符热力图（点号到 @ 的深浅）
    * 可选的 HTML 报告，包含每层每头的完整热力图
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .inference import Translator
from .masks import make_incremental_self_attn_mask, make_encoder_attn_mask, make_decoder_self_attn_mask
from .utils import ensure_dir, get_logger, setup_console

logger = get_logger()

RAMP = " .:-=+*#%@"


def display_token(token: str) -> str:
    """把 token 里的词尾标记去掉再显示 —— 热力图旁边只需要看内容。"""

    from .bpe import WORD_END

    return token[: -len(WORD_END)] if token.endswith(WORD_END) else token


# --------------------------------------------------------------------------
# 形状追踪
# --------------------------------------------------------------------------
def trace_shapes(
    translator: Translator,
    source: str,
    target: str,
    limit: int = 40,
) -> List[str]:
    """跑一次前向，记录每个子模块的输入输出形状。"""

    model = translator.model
    tokenizer = translator.tokenizer
    device = translator.device

    src_ids = tokenizer.encode(source, add_eos=True)
    tgt_ids = tokenizer.encode(target, add_bos=True, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    src_mask, tgt_mask = model.make_masks(src, tgt, tokenizer.pad_id)

    records: List[Tuple[str, str, str]] = []
    handles = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs, output):
            def shape_of(value):
                if isinstance(value, torch.Tensor):
                    return str(tuple(value.shape))
                return type(value).__name__

            in_shapes = ", ".join(shape_of(v) for v in inputs)
            out_shapes = ", ".join(shape_of(v) for v in (output if isinstance(output, (tuple, list)) else (output,)))
            records.append((name, in_shapes, out_shapes))
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):  # 只看叶子模块，避免刷屏
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(src, tgt, src_mask, tgt_mask)
    for handle in handles:
        handle.remove()

    lines = [
        f"输入：{source!r} -> {len(src_ids)} 个 token（含 <eos>）",
        f"      {tgt_ids}  <- 中文 {len(tgt_ids)} 个 token（<bos> ... <eos>）",
        "",
        f"{'模块':<46}{'输入形状':<28}{'输出形状'}",
        "-" * 104,
    ]
    for name, in_shapes, out_shapes in records[:limit]:
        lines.append(f"{name:<46}{in_shapes:<28}{out_shapes}")
    if len(records) > limit:
        lines.append(f"... 还有 {len(records) - limit} 个模块（用 --limit 调整显示数量）")
    return lines


# --------------------------------------------------------------------------
# 注意力
# --------------------------------------------------------------------------
def collect_attention(
    translator: Translator,
    source: str,
) -> Dict[str, object]:
    """先用模型自己翻一遍，再把"原文 + 生成的译文"整段喂回去，取注意力权重。"""

    model = translator.model
    tokenizer = translator.tokenizer
    device = translator.device

    translation = translator.translate_one(source)
    src_ids = tokenizer.encode(source, add_eos=True)
    tgt_ids = [tokenizer.bos_id] + tokenizer.encode(translation) + [tokenizer.eos_id]

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    src_mask = make_encoder_attn_mask(src, tokenizer.pad_id)
    tgt_mask = make_decoder_self_attn_mask(tgt, tokenizer.pad_id)

    model.eval()
    with torch.no_grad():
        memory, encoder_weights = model.encode(src, src_mask, return_weights=True)
        _, decoder_weights = model.decode(
            tgt, memory, tgt_mask, src_mask, return_weights=True
        )

    return {
        "source": source,
        "translation": translation,
        "src_tokens": [display_token(tokenizer.id_to_token[i]) for i in src_ids],
        "tgt_tokens": [display_token(tokenizer.id_to_token[i]) for i in tgt_ids],
        "encoder_self": [w[0].cpu() for w in encoder_weights],      # 每层 [heads, S, S]
        "decoder_self": [w["self"][0].cpu() for w in decoder_weights],
        "decoder_cross": [w["cross"][0].cpu() for w in decoder_weights],
    }


def ascii_heatmap(matrix: torch.Tensor, row_labels: Sequence[str], col_labels: Sequence[str], width: int = 74) -> str:
    """把 [rows, cols] 的权重画成字符热力图。"""

    rows, cols = matrix.shape
    # 如果列太多，就按列聚合（取平均），保证终端能显示
    if cols > width:
        factor = (cols + width - 1) // width
        padded = torch.nn.functional.pad(matrix, (0, factor * width - cols))
        matrix = padded.view(rows, -1, factor).mean(-1)
        cols = matrix.shape[1]

    lines = []
    header = " " * 22 + "".join(f"{i % 10}" for i in range(cols))
    lines.append(header)
    for r in range(rows):
        row = matrix[r]
        peak = row.max().item() or 1.0
        line = "".join(RAMP[min(len(RAMP) - 1, int(v.item() / peak * (len(RAMP) - 1) + 0.5))] for v in row)
        label = row_labels[r] if r < len(row_labels) else ""
        lines.append(f"{label[:18]:<20}|{line}")
    lines.append("")
    lines.append("  列（被看的位置）：" + " ".join(f"{i}:{t[:8]}" for i, t in enumerate(col_labels[:16])))
    if len(col_labels) > 16:
        lines.append(f"  ... 共 {len(col_labels)} 列")
    return "\n".join(lines)


def html_report(data: Dict[str, object], output: Path) -> None:
    """把所有层所有头的热力图写成一个自包含的 HTML。"""

    src_tokens: List[str] = data["src_tokens"]  # type: ignore[assignment]
    tgt_tokens: List[str] = data["tgt_tokens"]  # type: ignore[assignment]

    def grid(matrix: torch.Tensor, col_tokens: Sequence[str]) -> str:
        rows, cols = matrix.shape
        peak = matrix.max().item() or 1.0
        cells = []
        header = "".join(
            f'<th class="lbl">{"&nbsp;" if not token.strip() else token[:3]}</th>' for token in col_tokens
        )
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                value = matrix[r, c].item() / peak
                # 亮黄 -> 深红，数值越大越显眼
                color = f"rgba(220, 40, 40, {value:.3f})"
                text = f"{matrix[r, c].item():.2f}" if value > 0.25 else ""
                row_cells.append(f'<td style="background:{color}" title="{matrix[r, c].item():.4f}">{text}</td>')
            label = tgt_tokens[r] if r < len(tgt_tokens) else ""
            cells.append(f'<tr><th class="lbl">{label}</th>{"".join(row_cells)}</tr>')
        return f'<table><tr><th></th>{header}</tr>{"".join(cells)}</table>'

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Transformer 注意力可视化</title>",
        "<style>",
        "body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;background:#fafafa;color:#222}",
        "h1{font-size:20px} h2{margin-top:32px;font-size:16px} h3{margin:16px 0 6px;font-size:13px;color:#555}",
        "table{border-collapse:collapse;margin-bottom:18px}",
        "td{width:22px;height:22px;border:1px solid #eee;font-size:9px;text-align:center;color:#fff}",
        "th.lbl{font-size:10px;font-weight:400;padding:0 4px;color:#444;max-width:52px;overflow:hidden;white-space:nowrap}",
        ".box{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px;margin-bottom:18px}",
        "</style></head><body>",
        "<h1>Transformer 注意力可视化</h1>",
        f"<div class='box'><b>英文原文</b>：{data['source']}<br><b>模型译文</b>：{data['translation']}</div>",
    ]

    # 说明每种图在看什么
    parts.append("<h2>编码器自注意力（英文看英文）</h2>")
    parts.append("<p>行是当前位置，列是被看的位置。能看到模型把哪些词当成一个整体。</p>")
    for layer, weights in enumerate(data["encoder_self"]):  # type: ignore[arg-type]
        parts.append(f"<h3>第 {layer + 1} 层</h3>")
        for head in range(weights.shape[0]):
            parts.append(f"<b style='font-size:12px'>head {head}</b>")
            parts.append(grid(weights[head], src_tokens))

    parts.append("<h2>解码器交叉注意力（写中文时看英文的哪里）</h2>")
    parts.append("<p>这就是翻译对齐关系的可视化：写第几个中文字时，注意力落在英文的哪些词上。</p>")
    for layer, weights in enumerate(data["decoder_cross"]):  # type: ignore[arg-type]
        parts.append(f"<h3>第 {layer + 1} 层</h3>")
        for head in range(weights.shape[0]):
            parts.append(f"<b style='font-size:12px'>head {head}</b>")
            parts.append(grid(weights[head], src_tokens))

    parts.append("</body></html>")
    ensure_dir(output.parent)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description="看模型内部：形状追踪 + 注意力热力图")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--text", default="The committee postponed the meeting until next Monday.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layer", type=int, default=0, help="看第几层（从 0 开始）")
    parser.add_argument("--head", type=int, default=0, help="看第几个头")
    parser.add_argument("--shapes", action="store_true", help="打印张量形状追踪表")
    parser.add_argument("--html", default=None, help="输出 HTML 报告的路径")
    parser.add_argument("--limit", type=int, default=40, help="形状表最多显示多少行")
    args = parser.parse_args()

    translator = Translator(args.checkpoint, device=args.device)
    data = collect_attention(translator, args.text)

    logger.info("=" * 74)
    logger.info(f"英文原文：{data['source']}")
    logger.info(f"模型译文：{data['translation']}")
    logger.info(f"英文 token：{' '.join(str(t) for t in data['src_tokens'][:24])}")
    logger.info("=" * 74)

    if args.shapes:
        for line in trace_shapes(translator, args.text, str(data["translation"]), limit=args.limit):
            print(line)
        print()

    layer = min(args.layer, len(data["encoder_self"]) - 1)  # type: ignore[arg-type]
    head = min(args.head, data["encoder_self"][layer].shape[0] - 1)  # type: ignore[index,arg-type]

    logger.info(f"编码器自注意力（第 {layer + 1} 层，head {head}）")
    print(ascii_heatmap(data["encoder_self"][layer][head], list(data["src_tokens"]), list(data["src_tokens"])))  # type: ignore[index,arg-type]
    logger.info(f"解码器交叉注意力（第 {layer + 1} 层，head {head}）：写中文时看英文的哪里")
    print(ascii_heatmap(data["decoder_cross"][layer][head], list(data["tgt_tokens"]), list(data["src_tokens"])))  # type: ignore[index,arg-type]

    if args.html:
        html_report(data, Path(args.html))
        logger.info(f"完整热力图已写入 {args.html}（用浏览器打开）")


if __name__ == "__main__":
    main()
