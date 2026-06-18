1. Gym 环境测试代码

import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()

print(obs)

作用：
验证 Gymnasium 安装成功
验证 CartPole 环境可创建
查看初始 State（obs）

2. PPO 训练代码

import gymnasium as gym
from stable_baselines3 import PPO

# 创建环境
env = gym.make("CartPole-v1")

# 初始化 PPO 智能体
model = PPO("MlpPolicy", env, verbose=1)

# 训练智能体
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_cartpole")

作用
训练 PPO 智能体
学习 CartPole 平衡策略
保存模型

输出
ppo_cartpole.zip

3. PPO 测试代码

obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

作用
加载训练好的策略
让智能体自主玩 CartPole

4. Episode Reward 统计代码

total_reward = 0

obs, info = env.reset()

for _ in range(1000):

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:

        print("Episode total reward:", total_reward)

        total_reward = 0

        obs, info = env.reset()

结果
Episode total reward: 500
Episode total reward: 500

说明
PPO 已达到 CartPole 满分水平

Linux基础命令

查看当前位置

pwd

查看文件

ls

切换目录

cd ~/habitat-lab

创建目录

mkdir test

递归创建

mkdir -p scene_datasets/habitat-test-scenes

查找文件

find ~/habitat-lab/data -iname "*.glb"

find ~/habitat-lab/data -name "train.json.gz"

搜索内容

grep -R "habitat-test-scenes" habitat-lab/habitat/config

Conda环境

创建环境

conda create -n habitat39 python=3.9 -y

激活环境

conda activate habitat39

查看环境

conda env list

查看python

python --version

查看包

conda list

系统检查

查看GPU

nvidia-smi

查看系统

cat /etc/os-release

查看Git

git --version

5. Habitat-Sim安装验证

查看版本

python -c "
import habitat_sim
print('version:', habitat_sim.__version__)
print('cuda_enabled:', habitat_sim.cuda_enabled)
"

输出

version: 0.3.3
cuda_enabled: False

Habitat-Lab安装

下载源码

由于 git clone 失败

git clone https://github.com/facebookresearch/habitat-lab.git

报错

curl 28
443 timeout

改为

wget https://github.com/facebookresearch/habitat-lab/archive/refs/heads/main.zip

解压

unzip main.zip

进入目录

cd habitat-lab

安装

pip install -e habitat-lab

验证

python -c "
import habitat
print('habitat-lab ok')
"

成功输出

habitat-lab ok

数据集操作

WSL查看场景

find ~/habitat-lab/data -iname "*.glb" | head

输出：

apartment_1.glb
skokloster-castle.glb
van-gogh-room.glb

打包场景

cd ~/habitat-lab/data

tar czvf habitat_test_scenes.tar.gz versioned_data/habitat_test_scenes

打包PointNav数据集

cd ~/habitat-lab/data

tar czvf pointnav_dataset.tar.gz versioned_data/habitat_test_pointnav_dataset_1.0

AutoDL解压数据

解压场景

cd ~/habitat-lab/data

tar -xzvf ~/habitat_test_scenes.tar.gz

解压PointNav

cd ~/habitat-lab/data

tar -xzvf ~/pointnav_dataset.tar.gz

修复Habitat目录结构

创建场景目录

mkdir -p scene_datasets/habitat-test-scenes

复制glb

cp versioned_data/habitat_test_scenes/*.glb \
scene_datasets/habitat-test-scenes/

检查

ls scene_datasets/habitat-test-scenes

Habitat配置检查

查看配置

cat habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml

查看dataset配置

cat habitat-lab/habitat/config/habitat/dataset/pointnav/habitat_test.yaml

运行PointNav示例（最终成功）

官方Shortest Path Example

cd ~/habitat-lab

python examples/shortest_path_follower_example.py

最终成功输出：

Environment creation successful

Agent stepping around inside environment.

Episode finished

Video created:
examples/images/shortest_path_example/00/trajectory.mp4

查看生成结果

生成目录：

~/habitat-lab/examples/images/shortest_path_example/