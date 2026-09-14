# 第 12 章：复现论文的英德翻译结果

目标：用**一台 8 卡的 4090 机器**复现《Attention Is All You Need》的英德翻译成绩
——base 模型在 newstest2014 上 **27.3 BLEU**。

## 0. 论文设置 vs 本项目

| | 论文 | 本项目 |
| --- | --- | --- |
| 训练数据 | WMT14 英德 456 万句对 | 同左（Europarl v7 + Common Crawl + NC v9） |
| 词表 | 共享 BPE，约 37k | 共享 BPE，32k（可调） |
| 分词 | Moses tokenizer + subword-nmt | 手写预切分 + 手写 BPE |
| 模型（base） | d_model 512、8 头、6+6 层、d_ff 2048 | 同左 |
| dropout / 标签平滑 | 0.1 / 0.1 | 同左 |
| 优化器 | Adam(0.9, 0.98, 1e-9) + Noam，warmup 4000 | 同左 |
| 批大小 | 约 25k 源 token + 25k 目标 token | 同左（`max_tokens=50000`） |
| 步数 / 时间 | 100k 步 / 8×P100 12 小时 | 100k 步 / 8×4090 约 3~6 小时 |
| 解码 | beam 4、长度惩罚 0.6 | 同左 |
| 报告分数 | 最后 5 个 checkpoint 平均 | 同左（`--keep-last-checkpoints 5` + `nmt.average`） |

## 1. 准备数据（约 1~2 小时）

```bash
python -m nmt.corpus --recipe wmt14
```

会依次做四件事：下载 WMT 官方 tar 包 → 清洗 → 训 32k 的联合 BPE → 编码落盘。

几个预期值（用来判断有没有跑偏）：

| 阶段 | 正常范围 |
| --- | --- |
| 下载 | 约 0.7~1.5 GB（Europarl + Common Crawl + NC v9） |
| 清洗保留率 | 90% 以上（Common Crawl 最脏，会多丢一些） |
| 训练集句对 | 约 420~440 万（原始 456 万） |
| 英文平均 token 数 | 28~32（用 32k 词表时） |
| BPE 训练 | 约 1 小时（32k 合并规则、10 万句对子集） |

> 磁盘要留 **10 GB**（数据 6~7 GB + checkpoint 2~3 GB），内存建议 **16 GB 以上**。
> 嫌 BPE 慢就加 `--vocab-size 16000`，时间大致减半。

## 2. 8 卡冒烟测试（10 分钟）

**先跑通再加长**，这一步的目的是确认 DDP 和数据都对：

```bash
torchrun --nproc_per_node=8 -m nmt.train \
    --preset paper-base \
    --save-dir checkpoints/smoke \
    --max-steps 2000
```

该看到的：

* 日志里 `进程数 8`，只有 rank 0 在打信息级日志（其余进程只报 warning，避免刷屏）；
* `每卡 N 个 batch（token 预算 6,250）| 8 卡合计等效批 50,000 token`；
* loss 从 10 左右快速往下掉；
* 日志里的 `tok/s` 就是每卡的吞吐，**乘 8 就是全机速度**，用它反推总时间：
  `总时间 ≈ 100000 步 × 50000 token ÷ (8 × 单卡 tok/s)`；
* `checkpoints/smoke/` 下有 `last.pt` / `best.pt`。

## 3. 正式训练（3~6 小时）

```bash
# 用 nohup 挂后台，避免 SSH 断开把训练带走
nohup torchrun --nproc_per_node=8 -m nmt.train \
    --preset paper-base \
    --max-steps 100000 \
    --save-dir checkpoints \
    > train_console.log 2>&1 &

tail -f train_console.log      # 看进度
```

两个常用的续训命令：

```bash
# 被中断了：从 last.pt 接着训（--preset 必须和原来一致）
torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --resume auto --max-steps 100000

# 想再训久一点：先改大 warmup/lr_scale，否则学习率已经衰减到底，加步数没什么用
torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --resume auto \
    --max-steps 200000 --override train.warmup_steps=8000 --override train.lr_scale=1.4
```

> 为什么加步数要同时调学习率？Noam 的学习率按 `step^-0.5` 衰减，
> 10 万步时已经降到 1.4e-4；再训 10 万步全在很低的步长下爬，收益很小。

## 4. checkpoint 平均 + 评测

论文报的 27.3 是**最后 5 个 checkpoint 权重的平均**（`--preset paper-base` 已经默认每轮存快照）：

```bash
python -m nmt.average --checkpoint-dir checkpoints --num 5 --output checkpoints/averaged.pt

# 论文口径：test2014 + beam 4 + 区分大小写的 BLEU
python -m nmt.evaluate --checkpoint checkpoints/averaged.pt \
    --split test2014 --beam-size 4 --case-sensitive --show 5
```

顺手对比一下"平均前 vs 平均后"和"贪心 vs 束搜索"，这两个对比本身就是很好的实验记录：

```bash
python -m nmt.evaluate --checkpoint checkpoints/best.pt     --split test2014 --beam-size 4 --case-sensitive
python -m nmt.evaluate --checkpoint checkpoints/averaged.pt --split test2014 --compare-decoders --case-sensitive
```

## 5. 和论文的差异（诚实清单）

这些会造成**几十个 BLEU 小数点后一位**的偏差，也可能让最终数字比 27.3 低一点：

1. **BPE 在 10 万句对的子集上训练**（论文用全量）。想完全对齐就 `--bpe-train-pairs 4000000`，
   代价是这一步从 1 小时变成十几小时。
2. **分词器不同**：论文用 Moses tokenizer + subword-nmt；本项目是手写预切分 + 手写 BPE。
3. **清洗规则不同**：我们会丢掉长度比例异常、功能词判语言可疑的句子（约 5%）。
4. **测试集版本**：本项目用的 newstest2014 是 3003 句；论文报的 27.3 用的那份过滤后是 2737 句。
5. **BLEU 实现**：本项目是手写的简化 13a，和 sacrebleu 会有小数点后的差异。
6. **big 模型的批大小**：受 24 GB 显存限制用了 16k + 2 次梯度累积（等效 32k），论文是 50k。

**所以合理的预期是 24~27.5，而不是精确的 27.3。** 如果你只关心"有没有真正复现"，
看三个信号：训练集能正常收敛、验证 BLEU 持续上升到 20+、test2014 落在 24~27.5。

## 6. 时间预算

| 配置 | 步数 | 8×P100（论文） | 8×4090（估算） |
| --- | --- | --- | --- |
| base | 100k | 12 小时 | **3~6 小时** |
| big | 300k | 3.5 天 | **10~20 小时** |

跑完冒烟测试后，用日志里的 `tok/s` 算一下就能得到自己机器上的准确数字。

## 7. 排错

### NCCL 卡住不动

* 最常见的原因是**各卡步数不一致**：本项目已经强制每卡 batch 数相同（除不尽的末尾 batch 丢掉），
  如果你改过采样器要留意这一点；
* 其次是端口冲突：换 `--master_port`；
* 只有一张卡、想验证多进程流程是否正常：`--override train.device=cpu` 配合
  `torchrun --nproc_per_node=2`，会走 gloo + CPU，把 rank 分片、no_sync、barrier 全跑一遍
  （本项目的 DDP 就是这么在单卡机器上验证的）。

### CUDA out of memory

**先分清是"真的不够"还是"卡被别人占着"**。8 卡跑 `paper-base` 时每卡只需要 5~6 GB，
如果你看到某张卡显示 20 GB+ 已用，那基本不是你的模型吃的：

```bash
nvidia-smi                                          # 看每张卡的占用和进程
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 之前跑挂/被 Ctrl+C 的训练，worker 进程常常不会全退干净，会一直占着显存
pkill -f "nmt.train"
sleep 5 && nvidia-smi                               # 确认卡空了再重跑
```

确认是显存真不够，再按这个顺序压：

```bash
--override train.max_tokens=32768          # 全局批减半
--override train.accum_steps=2             # 用梯度累积补回等效批大小
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 动态 batching 的形状变化多，缓解碎片
```

训练时每轮会打印**各卡峰值显存**。如果这个数字随轮数缓慢上涨 → 是碎片或泄漏；
如果某一轮突然跳上去 → 是某个 batch 异常大，看 `train_log.csv` 里那一轮的长度分布。

### 报错说"序列长度 N 超过了位置编码支持的最大长度"

数据准备的长度上限（`--max-src-len`）和模型的位置编码上限（`model.max_len`）是两个旋钮。
本项目在训练/评测时会自动截断并提示，但如果你手改过配置，注意让 `max_len ≥ 训练长度上限`。

### 评测时报 "token id 最大是 X，但模型词表只有 Y"

数据和模型用的不是同一套词表 —— 大概率是 `--data-dir` 指错了目录
（比如模型在 `data/tiny` 上训的，评测时用了 `data/ready`）。

## 8. 如果结果不理想，按这个顺序调

1. **先看曲线**：`checkpoints/train_log.csv` 里 `val_bleu` 到 100k 步还在涨吗？还在涨就继续训（记得同时调 warmup/lr_scale）。
2. **看长度比**：稳定在 1.0 附近才正常；明显偏离就调 `--length-penalty`。
3. **看数据**：`--extra-pairs` 加到 Europarl 全量（196 万）能再多一点数据。
4. **上 big**：`--preset paper-big`，论文 28.4 BLEU。
5. **读译文**：`--show 8`，BLEU 差不代表译文一定差，反过来也一样。
