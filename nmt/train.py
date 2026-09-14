"""训练循环：把前面所有零件串起来，并且把"服务器上该有的东西"配齐。

这个文件里有四件事值得逐行读：

1. **一个 batch 里到底喂了什么**
       src      [B, S]   英文 id（含 <eos>）
       tgt      [B, T]   德文 id（含 <bos> 和 <eos>）
       decoder 输入 = tgt[:, :-1]，标签 = tgt[:, 1:]
       —— 错开一位，就是"老师强制"（teacher forcing）：每一步都在用正确的前文预测下一个字。

2. **混合精度（AMP）**
   float16 算得快、省显存，但动态范围窄，梯度容易下溢成 0，
   所以要配 GradScaler 把 loss 放大再反传，更新前再缩回来。
   bfloat16 动态范围和 float32 一样，不需要缩放，4090 上可以试试。

3. **梯度累积**
   显存装不下大 batch 时，把 N 个小 batch 的梯度加起来再更新一次，
   数学上近似等于一个大 batch（分母会略有差别，实践中无感）。
   代价是训练变慢，换来的是等效 batch size。

4. **断点续训**
   优化器状态、学习率步数、混合精度缩放因子、epoch/step 全都要存。
   只存模型权重的话，续训出来会"看着在跑，其实在退步"。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .bpe import BPE
from .bleu import evaluate_all
from .checkpoint import load_checkpoint, move_optimizer_state_to_device, restore_rng, save_checkpoint
from .config import PRESETS, Config, build_config
from .corpus import DATA_PAIR, build_dataset
from .dataset import ParallelTextDataset, build_dataloader, collate_batch, truncate_pair
from .inference import select_eval_indices, translate_dataset
from .loss import build_criterion
from .model import Transformer
from .scheduler import NoamScheduler
from .utils import (
    Timer,
    autocast_context,
    count_parameters,
    describe_device,
    get_logger,
    human_time,
    load_json,
    make_grad_scaler,
    need_grad_scaler,
    read_lines,
    resolve_device,
    set_seed,
    setup_console,
)

logger = get_logger()


# --------------------------------------------------------------------------
# 多卡（DDP）辅助
# --------------------------------------------------------------------------
def is_distributed() -> bool:
    """torchrun 会往环境变量里写 WORLD_SIZE，单卡跑就没有这个变量。"""

    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(prefer_cuda: bool = True) -> Tuple[int, int, int]:
    """初始化进程组，返回 (rank, world_size, local_rank)。单卡时返回 (0, 1, 0)。

    单机多卡用 NCCL 后端最快，torchrun 会自动把 RANK / WORLD_SIZE / LOCAL_RANK
    塞进环境变量，我们只需要按 local_rank 绑好显卡，别让 8 个进程都往 0 号卡上挤。

    prefer_cuda=False 时用 gloo + CPU：这条路是给"只有一张卡还要验证 DDP 逻辑"
    准备的 —— 用 torchrun --nproc_per_node=2 就能在本机把多进程流程跑一遍。
    """

    if not is_distributed():
        return 0, 1, 0

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if (prefer_cuda and torch.cuda.is_available()) else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def barrier() -> None:
    """所有 rank 对齐一次。只在分布式模式下有意义。"""

    if is_distributed():
        import torch.distributed as dist

        dist.barrier()


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def evaluate_on_split(
    model: Transformer,
    dataset: ParallelTextDataset,
    tokenizer: BPE,
    device: torch.device,
    split: str,
    data_dir: Path,
    max_sentences: int,
    beam_size: int,
    max_tokens: int = 4096,
    bleu_lowercase: bool = True,
) -> Dict[str, float]:
    """在验证集上跑一遍解码，返回 BLEU / chrF / 长度比。"""

    indices = select_eval_indices(len(dataset), max_sentences)
    hypotheses = translate_dataset(
        model, dataset, tokenizer, device,
        indices=indices, beam_size=beam_size, max_tokens=max_tokens,
    )
    reference_path = data_dir / f"{split}.ref.de"
    references = read_lines(reference_path)
    if len(references) != len(dataset):
        raise ValueError(
            f"{reference_path} 有 {len(references)} 行，但 {split}.pt 有 {len(dataset)} 句，"
            "两者应该一一对应（重新跑一次 python -m nmt.corpus 就能修好）"
        )
    references = [references[index] for index in indices]
    metrics = evaluate_all(hypotheses, references, lowercase=bleu_lowercase)

    # 顺手把 loss 也算出来：BLEU 只看最终译文的 n-gram，
    # loss 能告诉你"逐位置的预测概率"有没有在改善，两者一起看最稳。
    model.eval()
    # 这里用普通交叉熵（不平滑），因为它的数值可以和"每个 token 的平均负对数似然"直接对应
    criterion = build_criterion(tokenizer.pad_id, len(tokenizer), 0.0)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(indices), 32):
            chunk = indices[start : start + 32]
            # 同样要防超长：位置编码的上限比数据准备的长度上限更硬
            limit = int(model.config.max_len)
            batch = collate_batch(
                [truncate_pair(*dataset[i], limit) for i in chunk], tokenizer.pad_id
            )
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_input, labels = tgt[:, :-1], tgt[:, 1:]
            src_mask, tgt_mask = model.make_masks(src, decoder_input, tokenizer.pad_id)
            logits = model(src, decoder_input, src_mask, tgt_mask)
            loss = criterion(logits, labels)
            n_tokens = int((labels != tokenizer.pad_id).sum())
            total_loss += float(loss) * n_tokens
            total_tokens += n_tokens
    metrics["loss"] = total_loss / max(1, total_tokens)
    model.train()
    return metrics


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def train(
    config: Config,
    data_dir: Path,
    save_dir: Path,
    resume: Optional[str] = None,
    max_steps: int = 0,
    prepare_data: bool = False,
) -> Dict[str, float]:
    train_cfg, model_cfg = config.train, config.model

    # --- 多卡：先握手，再决定每张卡怎么分数据、谁来记日志 ---
    # device 显式写成 "cpu" 时走 gloo + CPU（本地验证多进程流程用），否则用 NCCL + 对应显卡
    use_cpu = str(train_cfg.device).startswith("cpu") or not torch.cuda.is_available()
    rank, world_size, local_rank = setup_distributed(prefer_cuda=not use_cpu)
    is_main = rank == 0
    if world_size > 1:
        # 非主进程只保留 warning 以上级别，否则 8 张卡的日志会互相刷屏
        logger.setLevel(logging.WARNING)
        device = torch.device("cpu") if use_cpu else torch.device("cuda", local_rank)
    else:
        device = resolve_device(train_cfg.device)
    # 每个进程的随机种子错开，保证 dropout 的随机性不重复（数据分片由 sampler 负责）
    set_seed(train_cfg.seed + rank)
    if is_main:
        logger.info(
            f"进程数 {world_size}"
            + (f"（本进程 rank={rank}，使用 {device}）" if world_size > 1 else "")
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    # --- 0. 没数据就顺手准备一份 ---
    if prepare_data or not (data_dir / "train.pt").exists():
        # 只能有一个进程去建数据，否则 8 个进程会同时写同一批文件
        if is_main:
            logger.info("没找到处理好的数据，先跑一遍数据准备")
            build_dataset(data_dir=data_dir.parent, out_dir=data_dir, vocab_size=model_cfg.src_vocab_size)
        barrier()

    tokenizer = BPE.load(data_dir / "vocab.json")
    # 校验数据是不是英译德的：服务器上很可能残留着上一次英译中的产物，
    # 那套数据自洽（词表和张量是配套的），不校验的话会静默训出一个中文模型。
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        logger.warning(
            f"{meta_path} 不存在，无法确认数据是不是英德语料。"
            "如果这份数据是别的语言对留下的，请先跑 python -m nmt.corpus"
        )
    else:
        meta = load_json(meta_path)
        if meta.get("pair") != DATA_PAIR:
            raise SystemExit(
                f"{data_dir} 里的数据是 {meta.get('pair')!r} 语言对的，"
                f"而当前代码需要 {DATA_PAIR!r}。\n"
                "请先重新生成数据：python -m nmt.corpus\n"
                "（旧产物可以直接删掉：data/ready 和 data/raw 下的非 *.de-en 目录）"
            )
    # 词表大小必须和模型一致；这里以实际训出来的词表为准，而不是预设里的数字
    model_cfg.src_vocab_size = len(tokenizer)
    model_cfg.tgt_vocab_size = len(tokenizer)

    # 位置编码的长度上限是硬约束：超过它 forward 会直接报错。
    # 数据准备默认放到 192，而 tiny 预设的 max_len 只有 128，
    # 所以这里把训练用的长度上限压到模型能承受的范围（超长的句子会被 sampler 跳过）。
    length_limit = model_cfg.max_len - 1
    if train_cfg.max_src_len > length_limit or train_cfg.max_tgt_len > length_limit:
        logger.warning(
            f"max_src_len/max_tgt_len 超过位置编码上限 {model_cfg.max_len}，"
            f"自动压到 {length_limit}（超长句子会被跳过）"
        )
        train_cfg.max_src_len = min(train_cfg.max_src_len, length_limit)
        train_cfg.max_tgt_len = min(train_cfg.max_tgt_len, length_limit)

    train_set = ParallelTextDataset(data_dir / "train.pt")
    dev_set = ParallelTextDataset(data_dir / "dev.pt")
    logger.info(tokenizer.summary())
    logger.info(
        f"训练集 {len(train_set):,} 句 / 验证集 {len(dev_set):,} 句 | "
        f"设备 {describe_device(device)}"
    )

    # --- 1. 模型 / 优化器 / 学习率 / 精度 ---
    raw_model = Transformer(model_cfg).to(device)
    model: torch.nn.Module = raw_model
    if world_size > 1:
        # DDP 会在构造时把 rank 0 的参数广播给所有进程，所以各进程的初始化必须一致
        # find_unused_parameters=False：这个模型每个参数都会被用到，开启它只会白费性能
        ddp_kwargs = {} if device.type == "cpu" else {"device_ids": [local_rank], "output_device": local_rank}
        model = torch.nn.parallel.DistributedDataParallel(
            raw_model, find_unused_parameters=False, **ddp_kwargs
        )
    criterion = build_criterion(tokenizer.pad_id, model_cfg.tgt_vocab_size, train_cfg.label_smoothing)
    optimizer = torch.optim.Adam(
        raw_model.parameters(),
        lr=1.0,  # 真实学习率由 Noam 调度器每步设置
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = NoamScheduler(optimizer, model_cfg.d_model, train_cfg.warmup_steps, train_cfg.lr_scale)

    amp_dtype = torch.bfloat16 if train_cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = need_grad_scaler(amp_dtype)
    scaler = make_grad_scaler(device, train_cfg.amp and use_scaler)

    if is_main:
        logger.info(raw_model.describe())
        logger.info(
            f"超参 | 学习率峰值 {scheduler.lr_at(train_cfg.warmup_steps):.2e} | warmup {train_cfg.warmup_steps} 步 | "
            f"标签平滑 {train_cfg.label_smoothing} | 梯度累积 {train_cfg.accum_steps} | "
            f"混合精度 {'关' if not train_cfg.amp else train_cfg.amp_dtype}"
            + ("（配 GradScaler）" if scaler is not None else "")
        )

    # --- 2. 续训 ---
    start_epoch = 0
    global_step = 0
    best_score = float("-inf")
    history: List[Dict[str, float]] = []

    if resume:
        resume_path = save_dir / "last.pt" if resume == "auto" else Path(resume)
        if resume_path.exists():
            # 注意这里一定是 map_location="cpu"：checkpoint 里还存着随机数状态
            # （CPU ByteTensor），按 cuda 读会被搬到显卡上，后面 restore_rng 直接报错。
            checkpoint = load_checkpoint(resume_path, map_location="cpu")
            raw_model.load_state_dict(checkpoint.model)
            if checkpoint.optimizer:
                optimizer.load_state_dict(checkpoint.optimizer)
                move_optimizer_state_to_device(optimizer, device)
            if checkpoint.scheduler:
                scheduler.load_state_dict(checkpoint.scheduler)
            if checkpoint.scaler and scaler is not None:
                scaler.load_state_dict(checkpoint.scaler)
            start_epoch = checkpoint.epoch
            global_step = checkpoint.step
            best_score = checkpoint.best_score
            history = list(checkpoint.history)
            restore_rng(checkpoint.rng)
            logger.info(
                f"从 {resume_path} 续训：第 {start_epoch} 轮，第 {global_step} 步，"
                f"历史最优 {checkpoint.best_metric} {best_score:.2f}"
            )
        else:
            logger.info(f"{resume_path} 不存在，从头开始训练")

    # --- 3. 数据加载器 ---
    # 多卡时 max_tokens 是**全局**预算，按卡数均摊；否则 8 张卡会各跑一个满额批，
    # 等效批大小变成论文的 8 倍，学习率就对不上了
    per_rank_tokens = max(1024, train_cfg.max_tokens // world_size)
    per_rank_sentences = max(1, train_cfg.max_sentences // world_size)
    loader, sampler = build_dataloader(
        train_set,
        pad_id=tokenizer.pad_id,
        max_tokens=per_rank_tokens,
        max_sentences=per_rank_sentences,
        bucket_size=train_cfg.bucket_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        seed=train_cfg.seed,
        max_src_len=train_cfg.max_src_len,
        max_tgt_len=train_cfg.max_tgt_len,
        rank=rank,
        world_size=world_size,
    )
    if is_main:
        logger.info(
            f"训练集共 {sampler.num_pairs:,} 句 | 每卡 {len(sampler):,} 个 batch（token 预算 {per_rank_tokens:,}）"
            + (f" | {world_size} 卡合计等效批 {per_rank_tokens * world_size:,} token" if world_size > 1 else "")
        )

    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,step,train_loss,grad_norm,lr,val_loss,val_bleu,val_chrf,"
            "length_ratio,elapsed_s,skipped_updates\n",
            encoding="utf-8",
        )

    writer = None
    if train_cfg.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
            logger.info(f"TensorBoard 日志写到 {save_dir / 'tensorboard'}")
        except ImportError:
            logger.warning("没装 tensorboard，跳过（需要的话 pip install tensorboard）")

    epochs = train_cfg.epochs
    accum = max(1, train_cfg.accum_steps)
    train_start = time.perf_counter()
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_grad_norm = 0.0
        update_count = 0
        norm_count = 0
        skipped = 0
        micro = 0
        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(loader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_input, labels = tgt[:, :-1], tgt[:, 1:]
            # 注意用 raw_model 调自定义方法：DDP 包装只转发子模块/参数/buffer，
            # 不转发 make_masks 这类普通方法（这是本地两进程测试抓出来的坑）
            src_mask, tgt_mask = raw_model.make_masks(src, decoder_input, tokenizer.pad_id)

            with autocast_context(device, train_cfg.amp, amp_dtype):
                logits = model(src, decoder_input, src_mask, tgt_mask)
                loss = criterion(logits, labels)

            micro += 1
            n_tokens = int((labels != tokenizer.pad_id).sum())
            epoch_loss += float(loss) * n_tokens
            epoch_tokens += n_tokens

            is_update = (micro % accum == 0) or (batch_index == len(loader) - 1)
            scaled = loss / accum
            # 梯度累积时，只有最后一次反传需要跨卡同步。
            # 中间几步套上 no_sync()，否则每步都做一次 all-reduce，通信量翻 accum 倍。
            sync_context = (
                model.no_sync()
                if (world_size > 1 and not is_update)
                else contextlib.nullcontext()
            )
            with sync_context:
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            if not is_update:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler 发现梯度里有 inf/nan 时会跳过这一步更新并调低缩放因子。
                # 这属于正常的自保护，但要让你看得见 —— 跳过率高说明数值不稳（通常是学习率太大）。
                if scaler.get_scale() < scale_before:
                    skipped += 1
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            lr = scheduler.step()
            global_step += 1
            update_count += 1
            norm_value = float(grad_norm)
            # inf/nan 不参与平均，否则整个 epoch 的 grad_norm 统计会永远变成 inf
            if math.isfinite(norm_value):
                epoch_grad_norm += norm_value
                norm_count += 1

            if global_step % train_cfg.log_every_steps == 0 or global_step == 1:
                elapsed = time.perf_counter() - epoch_start
                logger.info(
                    f"epoch {epoch + 1}/{epochs} | step {global_step} | "
                    f"loss {epoch_loss / max(1, epoch_tokens):.4f} | "
                    f"lr {lr:.2e} | grad {float(grad_norm):.2f} | "
                    f"tok/s {epoch_tokens / max(1e-6, elapsed):,.0f}"
                )

            if max_steps and global_step >= max_steps:
                logger.info(f"到达 --max-steps={max_steps}，提前结束本轮")
                break

        train_loss = epoch_loss / max(1, epoch_tokens)
        avg_grad = epoch_grad_norm / max(1, norm_count)
        elapsed = time.perf_counter() - epoch_start
        if skipped:
            logger.info(f"  本轮有 {skipped}/{update_count} 次更新因梯度非有限被跳过（GradScaler 自保护）")

        # --- 每轮结束：验证 + 保存 ---
        metrics: Dict[str, float] = {}
        # 多卡时只在主进程上解码验证：8 张卡各跑一遍验证是纯浪费
        if is_main and (epoch + 1) % max(1, train_cfg.eval_every_epochs) == 0:
            with Timer() as eval_timer:
                metrics = evaluate_on_split(
                    raw_model, dev_set, tokenizer, device, "dev", data_dir,
                    max_sentences=train_cfg.eval_max_sentences,
                    beam_size=train_cfg.eval_beam_size,
                    bleu_lowercase=train_cfg.bleu_lowercase,
                )
            logger.info(
                f"epoch {epoch + 1} 验证：loss {metrics['loss']:.4f} | "
                f"BLEU {metrics['bleu']:.2f} | chrF {metrics['chrf']:.2f} | "
                f"长度比 {metrics['length_ratio']:.2f} | 耗时 {eval_timer.elapsed:.1f}s"
            )

        # 把主进程算出的分数广播给所有 rank：早停、max_steps 这些控制流
        # 必须是所有 rank 一致的决定，否则会各自 break，DDP 直接卡死在 all-reduce
        score = metrics.get("bleu", float("-inf"))
        if world_size > 1:
            import torch.distributed as dist

            score_tensor = torch.tensor([score], dtype=torch.float32, device=device)
            dist.broadcast(score_tensor, src=0)
            score = float(score_tensor.item())

        improved = score > best_score
        if improved:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1

        if not is_main:
            # 记日志、写 CSV、存 checkpoint 都只由主进程做
            barrier()
            if max_steps and global_step >= max_steps:
                break
            if train_cfg.patience and no_improve >= train_cfg.patience:
                break
            continue

        record = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": train_loss,
            "grad_norm": avg_grad,
            "lr": scheduler.current_lr,
            "val_loss": metrics.get("loss", float("nan")),
            "val_bleu": metrics.get("bleu", float("nan")),
            "val_chrf": metrics.get("chrf", float("nan")),
            "length_ratio": metrics.get("length_ratio", float("nan")),
            "elapsed_s": elapsed,
            "skipped_updates": skipped,
        }
        history.append(record)
        with open(log_path, "a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(list(record.values()))
        if writer is not None:
            for key, value in record.items():
                if key not in {"epoch", "step"}:
                    writer.add_scalar(f"train/{key}", value, epoch + 1)
            writer.add_scalar("dev/bleu", metrics.get("bleu", float("nan")), epoch + 1)
            writer.add_scalar("dev/chrf", metrics.get("chrf", float("nan")), epoch + 1)

        checkpoint_kwargs = dict(
            model=raw_model, config=config, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, step=global_step, best_score=best_score,
            history=history, vocab_path=str(data_dir / "vocab.json"),
        )
        save_checkpoint(save_dir / "last.pt", epoch=epoch + 1, **checkpoint_kwargs)
        if improved:
            save_checkpoint(save_dir / "best.pt", epoch=epoch + 1, **checkpoint_kwargs)
            logger.info(f"  验证 BLEU 提升到 {best_score:.2f}，已保存 best.pt")

        # 论文的分数是最后 5 个 checkpoint 的平均，所以训练时要按轮存快照。
        # 快照不带优化器状态（小 2/3），而且只保留最近 N 个，免得磁盘被吃满。
        if train_cfg.keep_last_checkpoints > 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch + 1,
                save_optimizer=False,
                **checkpoint_kwargs,
            )
            snapshots = sorted(save_dir.glob("epoch_*.pt"))
            for old in snapshots[: -train_cfg.keep_last_checkpoints]:
                old.unlink()

        if max_steps and global_step >= max_steps:
            logger.info("到达 --max-steps，训练结束")
            barrier()
            break
        if train_cfg.patience and no_improve >= train_cfg.patience:
            logger.info(f"验证 BLEU 连续 {no_improve} 轮没有提升，早停")
            barrier()
            break
        barrier()   # 主进程可能要花几十秒解码验证，等等其他进程，别让它们先跑进下一轮

    if is_main:
        # 词表和配置各复制一份到 checkpoint 目录，让这个目录可以独立搬走
        (save_dir / "vocab.json").write_text(
            (data_dir / "vocab.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        config.save(save_dir / "config.json")

    total_time = time.perf_counter() - train_start
    if is_main:
        if writer is not None:
            writer.close()
        logger.info("=" * 68)
        logger.info(f"训练结束，总耗时 {human_time(total_time)}，最优验证 BLEU {best_score:.2f}")
        logger.info(f"checkpoint：{save_dir / 'best.pt'}")
        logger.info(f"日志：{log_path}")
        if train_cfg.keep_last_checkpoints > 0:
            logger.info(
                "下一步可以做 checkpoint 平均（论文报的分数就是最后 5 个平均）：\n"
                f"  python -m nmt.average --checkpoint-dir {save_dir} --num 5 "
                f"--output {save_dir / 'averaged.pt'}"
            )
    cleanup_distributed()
    return {"best_bleu": best_score, "elapsed_s": total_time, "steps": global_step}


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(
        description="训练英德 Transformer",
        epilog=(
            "单机 8 卡：torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --max-steps 100000\n"
            "（本项目只做英译德，语言对不一致的数据会在启动时被拒绝）"
        ),
    )
    parser.add_argument("--preset", default="base", choices=sorted(PRESETS))
    parser.add_argument("--data-dir", default="data/ready")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="续训：checkpoint 路径，或 auto（用 save-dir/last.pt）")
    parser.add_argument("--max-steps", type=int, default=0, help="限制总步数，用来限时训练或冒烟测试")
    parser.add_argument("--override", action="append", default=[], help="覆盖配置，如 --override train.epochs=5")
    parser.add_argument("--prepare-data", action="store_true", help="训练前先准备数据")
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        if "=" not in item:
            raise SystemExit(f"--override 要写成 key=value，收到的是 {item!r}")
        key, value = item.split("=", 1)
        # 自动推断类型，这样 --override train.epochs=5 能写成整数
        try:
            parsed: object = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = {"true": True, "false": False}.get(value.lower(), value)
        overrides[key] = parsed

    config = build_config(args.preset, overrides=overrides)
    if args.max_steps:
        config.train.max_steps = args.max_steps
    train(config, Path(args.data_dir), Path(args.save_dir), resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
