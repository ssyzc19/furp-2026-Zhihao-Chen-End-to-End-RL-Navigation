(habitat) user0724@DESKTOP-4KJ59TJ:~/habitat-lab$ head -120 examples/shortest_path_follower_example.py
#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numpy as np

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower（走最短路径）
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def shortest_path_example():
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
        ],
    )

    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.habitat.simulator.forward_step_size
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        print("Environment creation successful")
        for episode in range(3):
            env.reset()
            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                im = observations["rgb"]
                top_down_map = draw_top_down_map(info, im.shape[0])
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
            images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def main():
    shortest_path_example()


if __name__ == "__main__":
    main()
(habitat) user0724@DESKTOP-4KJ59TJ:~/habitat-lab$


这份代码非常重要。

因为这是你看到的第一个真正的 **PointNav 示例**。

我不打算逐行讲代码，而是帮你建立一个研究者视角。

---

# 先看最关键的一行

```python
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
```

这意味着：

```text
这个机器人不是随机行动
```

而是：

```text
知道如何走最短路
```

回忆一下前面的 `example.py`：

```python
env.action_space.sample()
```

意思：

```text
随机前进
随机左转
随机右转
```

而这里：

```python
best_action = follower.get_next_action(...)
```

意思：

```text
计算当前最好的动作
```

---

# 这才是真正的 PointNav

假设地图如下：

```text
S = 起点
G = 终点

#########
#S      #
#   ### #
#     G #
#########
```

随机机器人：

```text
左
右
右
左
前
左
...
```

乱走。

---

ShortestPathFollower：

```text
前
前
前
右转
前
前
```

直接走向目标。

---

# 再看配置文件

这一行特别重要：

```python
config = habitat.get_config(
    config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
)
```

这里出现了：

```text
pointnav
```

说明：

```text
这是 PointNav 任务
```

---

PointNav 本质：

```text
输入：
目标坐标

输出：
移动动作
```

例如：

```text
当前位置:
(1,2)

目标:
(8,5)
```

机器人要自己过去。

---

# 你应该关注的对象

这里：

```python
with SimpleRLEnv(config=config) as env:
```

创建环境。

然后：

```python
env.reset()
```

开始一局。

然后：

```python
while not env.habitat_env.episode_over:
```

不断循环。

---

这和 CartPole 完全一样：

```text
reset()

↓

while not done

↓

step()

↓

done
```

---

# 最关键的一行

```python
best_action = follower.get_next_action(
    env.habitat_env.current_episode.goals[0].position
)
```

翻译成人话：

```text
机器人知道目标在哪
↓
计算下一步怎么走
↓
返回动作
```

动作可能是：

```text
前进
左转
右转
停止
```

---

然后：

```python
observations, reward, done, info = env.step(best_action)
```

执行动作。

---

# 你现在应该理解什么？

不要记代码。

记下面这张图：

```text
PointNav

目标坐标
      ↓
ShortestPathFollower
      ↓
最佳动作
      ↓
env.step()
      ↓
新的观察
      ↓
继续导航
```

---

# 为什么这个例子重要？

因为它代表：

```text
导航领域的上界(Baseline)
```

即：

```text
如果机器人知道地图
知道目标
知道最短路

能做到什么程度？
```

---

# 接下来你应该做什么

不要继续看代码。

先回答下面三个问题：

### 问题1

`ShortestPathFollower` 和

```python
env.action_space.sample()
```

最大的区别是什么？

---

### 问题2

PointNav 的输入是什么？

提示：

```python
current_episode.goals[0].position
```

---

### 问题3

为什么这个例子比 `example.py` 更接近真正的机器人导航？

你试着用自己的话回答。

如果能回答出来，你就已经真正进入 Habitat PointNav 了，而不只是“安装好了 Habitat”。
