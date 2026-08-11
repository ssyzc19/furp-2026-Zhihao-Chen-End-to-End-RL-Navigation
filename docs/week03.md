# 实验链路完整陈述

## 起点：Baseline 三 seed
用标准 HM3D PointNav PPO 训练三个 seed（100/200/300），得到 val SR = 0.755 / 0.885 / 0.895，均值 0.845。对 200 个 episode 做失败分析，结论：

lost（迷路/超时）占 62%，平均 DTG 7.5m
near_miss 占 32%，DTG 仅 0.22m
失败分散在各场景，不是难场景问题，是导航能力不足
训练步数只有10m，需要到50m才够

## Stop-Aware Reward vs baseline
失败分析显示 32% 是 near_miss——agent 走到了目标旁边但没有在正确位置停下。于是给停止动作加了显式奖励：正确停止 +2.0，错误停止 -1.0。

训练三个 seed 后 val SR 提升到 0.875 / 0.895 / 0.900，均值 0.890（+0.045）。

但关键发现不是 SR 提升，而是 seed 间方差从 ±0.075 降到 ±0.013。Stop-aware 让训练更稳定——这在 TensorBoard 上也能看到。

## LR 线性衰减 vs baseline
观察到 baseline seed100 的 TensorBoard 参数波动显著大于 stop_aware seed100。分析波动来源时，梯度噪声（学习率固定、无衰减）是最直接的怀疑对象。

实验 A（use_linear_lr_decay=True）结果：训练 SR 从 0.805 → 0.873（+8.4%），但 grad_norm std 反而从 0.36 变差到 0.42，max 从 6.6 升到 11.6，value_loss 纹丝不动。

结论：LR decay 治标不治本，action_loss 负值比例只从 40.8% 降到 36.3%。

## exp-reward-norm（P1）vs baseline
对六个实验做了系统性 TensorBoard 分析，找到真正的根因链：


advantage 估计方差过高
    → value function 跟不上（stop_aware value_loss 上升 78%）
        → 梯度持续爬升
            → 熵崩溃（stop_s200 曾崩溃到 0.389）
                → PPO clip 失效
action_loss 负值比例 36-41% 是 advantage 方差高的直接症状——advantage 归一化（use_normalized_advantage）加上更平滑的 GAE（tau=0.99）被认为能从根本上压制方差。

但 P1 把两个变量同时改了，结果过激：action_loss 负值比例飙到 98.7%，grad_norm max 到 18.75，value_loss 从 0.49 涨到 1.37。两个变量叠加导致每次更新极度激进，PPO clip 以 84.7% 的比例拼命压制策略变化。

## P1a 和 P1b 单独测试
P1 失败的诊断是：normalized_advantage 和 tau=0.99 各自有效果，但组合放大了彼此的副作用。拆开单独测才能知道哪个有效、哪个有害。

P1a：只改 tau=0.99，验证 GAE 更平滑是否有帮助
P1b：只开 use_normalized_advantage=True，验证归一化单独的效果
两个实验目前在两台 GPU 上并行运行。

## Nav2与SLAM建图

# 下一步方向

1）sim-to-real
2）继续优化RL：让PPO训练更稳定

 xp-critic vs baseline

 exp-entropy vs baseline

# 困难与挑战
1）两台GPU操作起来困难，没有规范的流程
2）消融实验路径长，提高稳定性太过大。需要再回去找文章读看看有没有更好的实验方法，另外真机部署更能帮助PPO的训练优化可惜没有相机。
3）仿真策略用 depth image，TurtleBot3 只有 LiDAR，无法直接部署 RL 策略。相机未到货，这个 gap 短期内无法桥接。
4）Nav2+LiDAR 可以跑 PointGoal，但这和仿真训练的 RL 策略是两套完全独立的系统，对比意义有限——不能说明 RL 策略在真机上表现如何。
5）Nav2 需要先建静态地图，实验室环境的坐标系和目标点设定还没确定。
