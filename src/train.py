"""第 2 步：训练。

    python -m src.train                    # 用默认配置训练
    python -m src.train --epochs 30 --lr-scale 2 --warmup-steps 1500
    python -m src.train --smoke            # 1 分钟级冒烟测试（小模型 + 小数据）

训练循环的每个 epoch：
    1. 按长度分桶组 batch（src / tgt 都已编码、padding 到 batch 内最长）
    2. 目标句右移一位作为解码器输入，原目标句作为标签
    3. 前向 -> 标签平滑交叉熵 -> 反向 -> 梯度裁剪 -> 优化器/调度器更新
    4. 定期打印 loss / 学习率 / 吞吐量
    5. 每 N 个 epoch 在验证集上算 loss 与 BLEU，BLEU 更好就存 best.pt

关于 teacher forcing：
    训练时把**真实的**前一个词喂给解码器（而不是模型自己生成的词），
    这样所有时间步可以并行计算，收敛也快得多。
    代价是"训练/推理不一致"（exposure bias）——推理时模型看到的是自己生成的词，
    这也是 beam search 等方法存在的原因之一。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from .bleu import corpus_bleu
from .bpe import BPE
from .config import Config, tiny_config
from .data import TranslationDataset, create_dataloader, read_tsv
from .inference import translate_batch_with_references
from .loss import build_criterion, token_accuracy
from .model import Transformer
from .scheduler import NoamLR, build_optimizer
from .utils import (
    Timer,
    describe_environment,
    human_count,
    resolve_device,
    save_checkpoint,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------- #
# 评估
# ---------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loss(
    model: Transformer,
    loader,
    criterion,
    device: torch.device,
) -> float:
    """验证集上的平均 loss（衡量模型的词级预测质量）。"""
    model.eval()
    total, n_tokens = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in = model.shift_right(tgt, model.bos_id)
        logits = model(src, tgt_in)
        mask = (tgt != model.pad_id).sum().item()
        total += criterion(logits, tgt).item() * mask
        n_tokens += mask
    return total / max(1, n_tokens)


def evaluate_bleu(
    model: Transformer,
    tokenizer: BPE,
    pairs: Sequence,
    device: torch.device,
    max_len: int,
    beam_size: int = 1,
    limit: int = 200,
):
    """贪心/束搜索解码后算语料级 BLEU，并返回若干样例。"""
    subset = list(pairs[:limit]) if limit else list(pairs)
    results = translate_batch_with_references(
        model, tokenizer, subset, device=device, beam_size=beam_size, max_len=max_len
    )
    hyps = [r.hypothesis for r in results]
    refs = [r.reference or "" for r in results]
    bleu = corpus_bleu(hyps, refs)
    return bleu, results


# ---------------------------------------------------------------------- #
# 训练
# ---------------------------------------------------------------------- #
def train(
    cfg: Config,
    out_dir: str | Path = "checkpoints",
    resume: Optional[str | Path] = None,
    max_train_sentences: Optional[int] = None,
    log_every_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "train.log")
    tc = cfg.train
    mc = cfg.model

    set_seed(tc.seed)
    device = resolve_device(tc.device)
    if verbose:
        logger.info(f"环境: {describe_environment()}")
        logger.info(f"设备: {device}")

    # ---- 数据 ------------------------------------------------------ #
    data_dir = Path(tc.data_dir)
    tokenizer = BPE.load(data_dir / "tokenizer.json")
    train_pairs = read_tsv(data_dir / "train.tsv")
    dev_pairs = read_tsv(data_dir / "dev.tsv")
    if max_train_sentences:
        train_pairs = train_pairs[:max_train_sentences]

    train_ds = TranslationDataset(train_pairs, tokenizer, tc.max_src_len, tc.max_tgt_len)
    dev_ds = TranslationDataset(dev_pairs, tokenizer, tc.max_src_len, tc.max_tgt_len)
    train_loader, sampler = create_dataloader(
        train_ds,
        batch_size=tc.batch_size,
        max_tokens=tc.max_tokens,
        shuffle=True,
        num_workers=tc.num_workers,
        pad_id=tokenizer.pad_id,
        use_bucket_sampler=tc.use_bucket_sampler,
        seed=tc.seed,
    )
    dev_loader, _ = create_dataloader(
        dev_ds,
        batch_size=tc.batch_size,
        max_tokens=tc.max_tokens,
        shuffle=False,
        num_workers=tc.num_workers,
        pad_id=tokenizer.pad_id,
        use_bucket_sampler=False,
    )

    # 词表大小以实际分词器为准（build_data 可能因基础字符表而略作调整）
    mc.src_vocab_size = len(tokenizer)
    mc.tgt_vocab_size = len(tokenizer)
    cfg.save(out_dir / "config.json")
    tokenizer.save(out_dir / "tokenizer.json")

    # ---- 模型 ------------------------------------------------------ #
    model = Transformer(mc, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    model.to(device)
    criterion = build_criterion(tc.label_smoothing, tokenizer.pad_id)
    optimizer = build_optimizer(model, tc)
    scheduler = NoamLR(optimizer, d_model=mc.d_model, warmup_steps=tc.warmup_steps, scale=tc.lr_scale)

    if verbose:
        logger.info(
            f"模型: {human_count(model.num_parameters())} 参数 | "
            f"d_model={mc.d_model} heads={mc.n_heads} "
            f"enc={mc.num_encoder_layers} dec={mc.num_decoder_layers} d_ff={mc.d_ff}"
        )
        logger.info(
            f"数据: train={len(train_ds)} dev={len(dev_ds)} | "
            f"batch={tc.batch_size} max_tokens={tc.max_tokens} | "
            f"每个 epoch {len(train_loader)} 个 batch"
        )

    # ---- 断点续训 --------------------------------------------------- #
    start_epoch, global_step, best_bleu, best_loss = 0, 0, -1.0, float("inf")
    if resume:
        from .utils import load_checkpoint

        payload = load_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        start_epoch = payload.get("epoch", 0) + 1
        global_step = payload.get("step", 0)
        best_bleu = payload.get("best_bleu", -1.0)
        best_loss = payload.get("best_loss", float("inf"))
        if verbose:
            logger.info(f"已从 {resume} 恢复：epoch={start_epoch} step={global_step} best_bleu={best_bleu:.2f}")

    # ---- CSV 日志 ---------------------------------------------------- #
    csv_path = out_dir / "train_log.csv"
    csv_file = csv_path.open("a" if csv_path.exists() and resume else "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    if not resume:
        csv_writer.writerow(["epoch", "step", "train_loss", "grad_norm", "lr", "val_loss", "val_bleu", "elapsed_s"])

    log_every = log_every_steps or tc.log_every_steps
    max_decode_len = min(mc.max_len, tc.max_tgt_len)
    timer = Timer()
    history: Dict[str, float] = {}
    tokens_seen = 0

    try:
        for epoch in range(start_epoch, tc.epochs):
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
            running_loss, running_acc, n_batches = 0.0, 0.0, 0
            last_grad_norm = 0.0
            timer.reset()

            for batch in train_loader:
                src = batch["src"].to(device, non_blocking=True)
                tgt = batch["tgt"].to(device, non_blocking=True)
                tgt_in = model.shift_right(tgt, model.bos_id)

                logits = model(src, tgt_in)          # [B, Tt, V]
                loss = criterion(logits, tgt)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 梯度裁剪：Transformer 偶发梯度爆炸，裁剪让训练更稳
                last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad))
                optimizer.step()
                scheduler.step()

                acc, n_tok = token_accuracy(logits.detach(), tgt, tokenizer.pad_id)
                running_loss += loss.item()
                running_acc += acc
                n_batches += 1
                global_step += 1
                tokens_seen += n_tok

                if global_step % log_every == 0:
                    avg_loss = running_loss / n_batches
                    avg_acc = running_acc / n_batches
                    ppl = float(torch.exp(torch.tensor(avg_loss)))
                    logger.info(
                        f"epoch {epoch + 1}/{tc.epochs} step {global_step} "
                        f"loss {avg_loss:.3f} ppl {ppl:6.2f} acc {avg_acc:.3f} "
                        f"|g| {last_grad_norm:.2f} lr {scheduler.current_lr():.2e} "
                        f"{tokens_seen / max(timer.elapsed, 1e-6):.0f} tok/s"
                    )

            train_loss = running_loss / max(1, n_batches)
            history["train_loss"] = train_loss
            history["train_acc"] = running_acc / max(1, n_batches)

            # ---- 验证 ------------------------------------------------- #
            do_eval = (epoch + 1) % max(1, tc.eval_every_epochs) == 0 or epoch == tc.epochs - 1
            val_loss, val_bleu = float("nan"), float("nan")
            if do_eval:
                val_loss = evaluate_loss(model, dev_loader, criterion, device)
                val_bleu, samples = evaluate_bleu(
                    model, tokenizer, dev_pairs, device,
                    max_len=max_decode_len, beam_size=1, limit=tc.eval_max_sentences,
                )
                history["val_loss"] = val_loss
                history["val_bleu"] = val_bleu
                logger.info(
                    f"[验证] epoch {epoch + 1} val_loss {val_loss:.3f} "
                    f"val_perplexity {float(torch.exp(torch.tensor(val_loss))):.2f} "
                    f"val_bleu(greedy) {val_bleu:.2f} | 耗时 {timer.format()}"
                )
                for r in samples[:3]:
                    logger.info(f"    源: {r.source}")
                    logger.info(f"    译: {r.hypothesis}")
                    logger.info(f"    参: {r.reference}")

                # 更好的模型：优先看 BLEU；BLEU 都为 0 时用 loss 兜底
                improved = (val_bleu > best_bleu) or (
                    val_bleu == 0.0 and best_bleu <= 0.0 and val_loss < best_loss
                )
                if improved:
                    best_bleu, best_loss = max(val_bleu, 0.0), val_loss
                    # best.pt 只存权重（不含优化器状态）：文件小一半以上，
                    # 推理时 load_model 也只需要权重。last.pt 才是"续训用"的完整状态。
                    save_checkpoint(
                        out_dir / "best.pt", model, None, None,
                        epoch=epoch, step=global_step, val_loss=val_loss, val_bleu=val_bleu,
                        best_bleu=best_bleu, best_loss=best_loss, config=cfg.to_dict(),
                    )
                    logger.info(f"    ✓ 保存最佳模型 best.pt (bleu {val_bleu:.2f}, loss {val_loss:.3f})")

            csv_writer.writerow(
                [epoch + 1, global_step, f"{train_loss:.4f}", f"{last_grad_norm:.4f}",
                 f"{scheduler.current_lr():.6e}", f"{val_loss:.4f}", f"{val_bleu:.4f}",
                 f"{timer.elapsed:.1f}"]
            )
            csv_file.flush()

            save_checkpoint(
                out_dir / "last.pt", model, optimizer, scheduler,
                epoch=epoch, step=global_step, val_loss=val_loss, val_bleu=val_bleu,
                best_bleu=best_bleu, best_loss=best_loss, config=cfg.to_dict(),
            )

    finally:
        csv_file.close()

    history["best_bleu"] = best_bleu
    history["best_loss"] = best_loss
    logger.info(f"训练结束：best_bleu={best_bleu:.2f} best_val_loss={best_loss:.3f}")
    logger.info(f"权重保存在 {out_dir.resolve()}（best.pt / last.pt）")
    return history


# ---------------------------------------------------------------------- #
# 命令行入口
# ---------------------------------------------------------------------- #
def _override(cfg, args) -> None:
    """把命令行参数覆盖到配置对象上（None 表示不改）。"""
    pairs = [
        ("epochs", "epochs"), ("batch_size", "batch_size"), ("max_tokens", "max_tokens"),
        ("lr_scale", "lr_scale"), ("warmup_steps", "warmup_steps"),
        ("label_smoothing", "label_smoothing"), ("seed", "seed"),
        ("data_dir", "data_dir"), ("eval_max_sentences", "eval_max_sentences"),
        ("eval_every_epochs", "eval_every_epochs"), ("max_src_len", "max_src_len"),
        ("max_tgt_len", "max_tgt_len"),
    ]
    for attr, arg_name in pairs:
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg.train, attr, value)

    model_pairs = [
        ("d_model", "d_model"), ("n_heads", "n_heads"),
        ("num_encoder_layers", "enc_layers"), ("num_decoder_layers", "dec_layers"),
        ("d_ff", "d_ff"), ("dropout", "dropout"), ("activation", "activation"),
    ]
    for attr, arg_name in model_pairs:
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg.model, attr, value)

    if getattr(args, "norm_first", False):
        cfg.model.norm_first = True
    if getattr(args, "tie_embeddings", False):
        cfg.model.tie_embeddings = True
    if getattr(args, "no_bucket", False):
        cfg.train.use_bucket_sampler = False
    if getattr(args, "device", None):
        cfg.train.device = args.device


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练 Transformer 翻译模型")
    parser.add_argument("--data-dir", default=None, help="build_data 产生的目录")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="动态 batching 的 token 预算")
    parser.add_argument("--lr-scale", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--enc-layers", type=int, default=None)
    parser.add_argument("--dec-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", default=None, choices=["relu", "gelu", "swish"])
    parser.add_argument("--norm-first", action="store_true", help="改用 pre-norm 结构")
    parser.add_argument("--tie-embeddings", action="store_true", help="输出层与目标词嵌入共享权重")
    parser.add_argument("--no-bucket", action="store_true", help="关掉长度分桶采样")
    parser.add_argument("--max-src-len", type=int, default=None)
    parser.add_argument("--max-tgt-len", type=int, default=None)
    parser.add_argument("--eval-every-epochs", type=int, default=None)
    parser.add_argument("--eval-max-sentences", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto / cpu / cuda")
    parser.add_argument("--resume", default=None, help="从 last.pt 继续训练")
    parser.add_argument("--max-train-sentences", type=int, default=None, help="只用前 N 句训练")
    parser.add_argument("--smoke", action="store_true", help="小模型 + 小数据快速跑通，验证代码正确性")
    parser.add_argument("--config", default=None, help="从 JSON 加载配置")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    _override(cfg, args)

    max_train = args.max_train_sentences
    if args.smoke:
        cfg.model = tiny_config()
        cfg.train.epochs = args.epochs or 2
        cfg.train.batch_size = 32
        cfg.train.max_tokens = 2048
        cfg.train.warmup_steps = 100
        cfg.train.lr_scale = 3.0
        cfg.train.eval_max_sentences = 100
        max_train = max_train or 3000

    train(
        cfg,
        out_dir=args.out_dir,
        resume=args.resume,
        max_train_sentences=max_train,
        log_every_steps=args.log_every_steps,
    )


if __name__ == "__main__":
    main()
