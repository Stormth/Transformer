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

**进度显示有两种形态**（`--progress auto` 自动选）：

* 在终端里跑（TTY）→ 一条原地刷新的进度条，带百分数、`step/s` 和 **ETA**：

  ```
  epoch 1/1000 ████████············ | 320/2000 | 16.0% | 2.41 step/s | ETA 11m36s | loss 7.8123 | lr 1.20e-04 | 41.2K tok/s
  ```

* 输出被重定向到文件（`> log` 或 `| tee`）→ 自动退化成每 `log_every_steps` 步一行日志，
  免得日志文件里被 `\r` 弄成一堆互相覆盖的乱码。想强制开进度条用 `--progress on`。

## 3. 正式训练（3~6 小时）

### 先挂到后台再启动（重要）

**不要直接在前台跑。** SSH 断线、frp / 跳板机抖动、笔记本合盖、家里网断了——
任何一个都会把进程带走，几小时的训练直接报废。用 `tmux` / `screen` / `zellij` / `nohup`
任意一种把它扔到后台：

```bash
# 方式一：tmux（推荐，随时能回去看实时日志）
tmux new -s train
torchrun --nproc_per_node=8 -m nmt.train \
    --preset paper-base --max-steps 100000 --save-dir checkpoints \
    --override train.save_every_steps=2000 \
    2>&1 | tee train_console.log
# 按 Ctrl+B 再按 D 脱离；之后 tmux attach -t train 回到这个会话
# 查看所有会话：tmux ls

# 方式二：screen（几乎每台服务器都自带）
screen -S train
torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --max-steps 100000 \
    2>&1 | tee train_console.log
# 按 Ctrl+A 再按 D 脱离；screen -r train 回来

# 方式三：nohup（最轻量，不需要装任何东西）
nohup torchrun --nproc_per_node=8 -m nmt.train \
    --preset paper-base \
    --max-steps 100000 \
    --save-dir checkpoints \
    > train_console.log 2>&1 &

tail -f train_console.log      # 随时看进度
```

三个要点：

1. **一定要用 `tee` 或 `>` 留一份日志文件**。终端会话丢了还能从文件看进度，
   而且出问题时可以直接把这个文件发出来。
2. **`--override train.save_every_steps=2000`**：万一真的被杀掉，最多只丢 2000 步。
3. **别用 `kill -9` 直接杀 torchrun**。8 个 worker 常常不会跟着退出，
   残留进程会一直占着显存 —— 下次启动就会莫名其妙 OOM（本项目的首跑就是这么栽的：
   一张 24 GB 的卡显示只剩 33 MB 空闲，而模型本身只需要 5~6 GB）。
   正确做法是 `pkill -f "nmt.train"`，等 5 秒，再用 `nvidia-smi` 确认显存真的释放了。

### 万一还是断了

```bash
# 看它还活着没有
gpustat                      # 或者 nvidia-smi
ps aux | grep nmt.train
tail -20 train_console.log

# 从最近的 checkpoint 接着训（最多丢 save_every_steps 步）
torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --resume auto --max-steps 100000
```

续训要求 checkpoint 目录里同时有 `last.pt` 和 `vocab.json`（训练脚本会自动把词表复制过去）。

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
```

训练时每轮会打印**各卡峰值显存**。如果这个数字随轮数缓慢上涨 → 是碎片或泄漏；
如果某一轮突然跳上去 → 是某个 batch 异常大，看 `train_log.csv` 里那一轮的长度分布。

### 显存"越跑越多"：先想到碎片

**动态 batching 是碎片的温床**：每一步的 batch 形状都不一样，缓存块对不上号，
碎片越攒越多，跑几小时后 `nvidia-smi` 上的占用会比刚启动时高一倍。
本项目的 8 卡首跑就是这样：21:37 每卡 9~16 GB，23:56 变成 25.2 GB 只剩 31 MB，
再过两小时另一张卡也爆一次 —— 但 step 的显存需求其实一直没变。

**根治办法是把分配器换成可扩展段**（PyTorch 2.1+）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 -m nmt.train --preset paper-base --max-steps 100000
```

想确认是不是碎片，用这条盯着看（正常情况下应该平掉，一直涨就是碎片或泄漏）：

```bash
watch -n 30 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

配套的代码侧还有两件事已经做掉了：损失函数**分块计算**（不再一次性持有
`[token 数, 词表大小]` 的巨块张量，峰值只跟 chunk 大小有关），以及每步的
GPU 同步全部去掉（同步会把流水线排空，既慢又推高峰值占用）。

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
