# PointNav PPO → TurtleBot3 Deployment

将 Habitat 训练的 PointNav PPO 策略部署到真实 TurtleBot3（ROS2 Humble / OrangePi 5 Pro）。

## 状态

| 模块 | 状态 | 说明 |
|------|------|------|
| odom → PointGoal 计算 | ✅ 完成 | 从 `/odom` 算相对目标极坐标 `[rho, phi]` |
| 动作 → `/cmd_vel` 转换 | ✅ 完成 | 离散动作 → 定时 Twist（开环） |
| Episode 主循环 | ✅ 完成 | 到达目标 / STOP 自动停车 |
| 策略加载 + 推理 | ✅ 完成 | CPU 推理，加载 `ckpt.49.pth` |
| **相机输入 (RGB+Depth)** | ⏳ **待相机到位** | `_rgb_cb` / `_depth_cb` 是 TODO |

**当前可测**：没有相机也能启动节点，验证 odom→pointgoal 和底盘运动正确性（策略推理部分会被 camera gate 挡住并打印提示）。

## 文件

```
pointnav_deploy/
├── pointnav_deploy_node.py   # ROS2 主节点
├── policy_wrapper.py          # 策略加载 + PointGoal 计算
├── action_mapper.py           # 离散动作 → cmd_vel（含速度标定参数）
└── README.md
```

## 部署到 OrangePi

```bash
# 1. 复制到机器人
scp -r pointnav_deploy orangepi@<robot_ip>:/home/orangepi/furp_code/gaiouka/

# 2. 复制 checkpoint
scp ckpt.49.pth orangepi@<robot_ip>:/home/orangepi/furp_code/gaiouka/
```

## 前置条件（OrangePi 上）

策略推理需要 `torch` + `habitat_baselines`。OrangePi 是 ARM，装 CPU 版 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# habitat 包较重，若装不上见下方"轻量导出"方案
```

**轻量导出方案（推荐用于 ARM）**：在训练机上把策略导出为 TorchScript，机器人端只需 `torch`，无需完整 habitat：

```python
# 在训练机上运行（后续提供导出脚本）
traced = torch.jit.trace(actor_critic, example_obs)
traced.save("pointnav_policy.pt")
```

## 关键标定步骤 ⚠️

`action_mapper.py` 里的 `LINEAR_SPEED` / `ANGULAR_SPEED` 必须用真机实测，否则"前进0.25m/转10°"会不准。

**直线速度标定**：
```bash
# 发布固定速度，用卷尺量 t 秒走的距离 d，真实速度 = d/t
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"
# 用 /odom 也可以：记录前后 x 坐标差
```

**转向速度标定**：
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}"
# 量 t 秒转过的角度，或读 /odom 的 yaw 变化
```

把实测值填回 `action_mapper.py`。

## 运行

```bash
# 1. 启动机器人底盘
ros2 launch turtlebot3_bringup robot.launch.py

# 2. 新终端启动部署节点（目标点在 odom 前方 2m）
cd /home/orangepi/furp_code/gaiouka/pointnav_deploy
python3 pointnav_deploy_node.py --ros-args \
    -p ckpt_path:=/home/orangepi/furp_code/gaiouka/ckpt.49.pth \
    -p goal_x:=2.0 -p goal_y:=0.0
```

## 相机到位后要做的 3 件事

1. 确认相机话题名，填入 `-p rgb_topic:=... -p depth_topic:=...`
2. 取消注释 `_rgb_cb` / `_depth_cb`，实现图像 resize 到 256×256、depth 归一化到 0..1
3. 确认坐标系约定：Habitat 相机朝向 vs 真机相机安装方向

## sim-to-real 注意事项

- **视觉域差距**：Habitat 渲染图 vs 真实相机图，可能需要 domain randomization 重训
- **PointGoal 漂移**：`/odom` 长距离会累积误差，可用外部定位（如 AMCL + 地图）校正
- **开环动作误差**：定时 Twist 有惯性/滑移误差，可改为闭环（用 odom 反馈判断是否走够距离）
