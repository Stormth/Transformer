# 第 7 章：训练

对应代码：`nmt/loss.py`、`nmt/scheduler.py`、`nmt/checkpoint.py`、`nmt/train.py`

## 一个 batch 里到底喂了什么

```
src             [B, S]     英文 id，以 <eos> 结尾
tgt             [B, T]     德文 id，以 <bos> 开头、<eos> 结尾
decoder 输入    tgt[:, :-1]     <bos> 我 明天 去 学校
labels          tgt[:, 1:]      我 明天 去 学校 <eos>
```

错开一位，就是**teacher forcing**：每一步都在用正确的前文预测下一个字。
注意最后一个位置的 label 是 `<eos>`，模型要学会"什么时候该停"。

## 损失：标签平滑交叉熵

对应代码：`nmt/loss.py`

普通交叉熵只关心"正确答案那一项"，逼模型把正确词的概率推到 1。
翻译这种"一句话有多种正确译法"的任务上，这会让模型过度自信、更容易过拟合。

标签平滑（论文 5.4 节，eps=0.1）把正确答案的置信度从 1 降到 0.9，
剩下的 0.1 平摊给其它词：

```
目标分布 = [0.9 给正确词, 0.1/(V-1) 给其它每个词]
```

它在说："这一个词是对的，但别的词也不是完全没道理。"

两个必须知道的后果：

1. **loss 不可能降到 0**。看到 loss 卡在 1.x 别慌，这是设计如此；
2. 它和 BLEU 是两件事：loss 衡量"逐位置的预测概率"，BLEU 衡量"整句的 n-gram 重合"。
   训练日志里两个都要看。

`<pad>` 位置不参与 loss（`ignore_index`）—— 它们只是凑长度用的。

## 学习率：Noam 调度

对应代码：`nmt/scheduler.py`

```
lr = scale × d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```

画出来是"先线性上升、到 `warmup` 步达到峰值、之后按 `step^-0.5` 衰减"。

* **为什么要 warmup**：训练最初几步，模型权重是随机的，梯度方向很不可靠。
  先用小步长试探，再逐步加大，可以避免一开始就把参数带进坏区域。
  post-norm 结构对这个特别敏感 —— 没有 warmup，base 模型几乎必然训崩。
* **为什么要衰减**：接近最优点时步长要越来越小，否则会在最优点附近来回跳。

### 一个容易踩的坑：峰值学习率不是常数

```
峰值 lr = scale × d_model^(-0.5) × warmup^(-0.5)
```

注意 `warmup` 在分母里。论文的 base 配置（d_model=512, warmup=4000）峰值约 `7e-4`。
但如果你把模型改小到 d_model=64、同时把 warmup 从 4000 调到 30，峰值会变成 `1.4e-2` —— **大了 20 倍**，训练会直接震荡。

本项目在 `tiny` / `small` 预设里显式降低了 `lr_scale` 来补偿这一点，
`train.py` 启动时也会把峰值打印出来。改小模型或改短 warmup 时，先看这行日志。

## 混合精度（AMP）

float16 算得快、省显存，但动态范围窄（最大 65504），梯度容易下溢成 0。
标准做法是**梯度缩放**：

1. 前向算出的 loss 乘以一个很大的缩放因子（比如 65536）再反传；
2. 更新前把梯度除回来（`unscale_`）；
3. 如果发现梯度里有 inf/nan，就跳过这一步更新并调低缩放因子。

`torch.amp.GradScaler` 把这三件事都做了。

bfloat16 的动态范围和 float32 一样宽，**不需要**缩放，在 4090 上可以直接用：

```bash
python -m nmt.train --preset base --override train.amp_dtype=bfloat16
```

注意顺序：`unscale_` → 梯度裁剪 → `step` → `update`。顺序错了裁剪的就是被放大过的梯度。

## 梯度累积

显存装不下大 batch 时，把 N 个小 batch 的梯度加起来，累积到 N 次再更新一次参数。

数学上近似等于一个大 batch。为什么要"近似"？因为每个小 batch 的 loss 是各自平均的，
`(L1+L2)/2` 和大 batch 的 `sum/N` 在样本数不等时会有细微差别。实践中无感，但你应该知道这一点。

```bash
python -m nmt.train --preset base --override train.accum_steps=4    # 等效 batch ×4
```

## 梯度裁剪

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

把所有参数的梯度当成一个长向量，如果它的 L2 范数超过 1.0 就整体缩回去（方向不变）。
作用是防止个别 batch 产生爆炸性梯度把模型带飞。

日志里的 `grad` 就是这个范数。健康的训练里它应该在 0.1~10 之间波动，
如果长期贴着裁剪阈值，说明学习率太大或者数据有问题。

## 验证与 checkpoint

每个 epoch 结束后：

1. 在 dev（newstest2013）上解码一批句子（默认 400 句，贪心解码，快）；
2. 算 BLEU / chrF / 长度比，写进 `train_log.csv`；
3. 保存 `last.pt`；BLEU 超过历史最优时同时保存 `best.pt`。

**一个"能续训"的 checkpoint 里必须有什么？**

| 内容 | 少了会怎样 |
| --- | --- |
| 模型权重 | 显然 |
| 优化器状态 | Adam 的一阶/二阶矩丢失，续训等于重新开始，曲线会突然变差 |
| 学习率步数 | warmup 从头再来一遍，或学习率从峰值重新开始 |
| GradScaler 缩放因子 | 混合精度需要重新预热，前几百步效率低 |
| epoch / step / 历史指标 | 日志断档，无法判断是否收敛 |
| 配置（结构超参） | 读的时候可能和写的时候结构不一致，报错或静默错误 |
| 随机数状态 | 只有想"续训后的数据顺序也完全一致"时才需要 |

本项目把这些全存进 `checkpoints/last.pt`，`--resume auto` 就能接着训：

```bash
python -m nmt.train --preset base --resume auto
```

另外 checkpoint 是**先写临时文件再原子替换**的。这样训练中途被杀不会留下半个损坏的文件，
这在动辄几小时的服务器训练里很重要。

## 训练日志怎么看

```
epoch 3/30 | step 1200 | loss 4.2137 | lr 6.82e-04 | grad 0.84 | tok/s 41,203
epoch 3 验证：loss 4.0521 | BLEU 8.31 | chrF 31.2 | 长度比 1.04 | 耗时 38.4s
```

几个"体检指标"：

| 现象 | 可能原因 |
| --- | --- |
| loss 一直不降 | 学习率太大/太小、数据有问题、掩码写错 |
| loss 降但 BLEU 不涨 | 数据噪声大、模型在背训练集（看验证集） |
| 长度比远小于 1 | 模型倾向于早停，试试调长度惩罚或看 `<eos>` 的预测 |
| 长度比远大于 1 | 模型不会停，检查训练目标里 `<eos>` 的监督信号 |
| grad 范数长期 > 10 | 学习率过大，或者有坏 batch |
| 验证 BLEU 强烈震荡 | 学习率过大、验证集太小、batch 太小 |

## 自己动手

```bash
# 1. 限时训练：只跑 300 步看曲线形状
python -m nmt.train --preset small --max-steps 300

# 2. 学习率扫描（这是最划算的调参）
python -m nmt.train --preset small --override train.lr_scale=0.3 --max-steps 500
python -m nmt.train --preset small --override train.lr_scale=1.0 --max-steps 500
python -m nmt.train --preset small --override train.lr_scale=3.0 --max-steps 500

# 3. 关掉标签平滑，比较 loss 的下界和最终 BLEU
python -m nmt.train --preset small --override train.label_smoothing=0.0

# 4. 故意把 warmup 设得很小，看 loss 曲线怎么抖
python -m nmt.train --preset small --override train.warmup_steps=50

# 5. 看日志（Excel 直接能打开 train_log.csv）
python -m nmt.train --preset base --override train.tensorboard=true
tensorboard --logdir checkpoints/tensorboard
```
