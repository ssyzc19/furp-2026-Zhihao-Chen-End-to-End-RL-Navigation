Q：PPO 到底学到了什么？

A：通过不断与环境交互，

根据Reward调整神经网络参数，

让Agent在相似State下

更倾向于选择高Reward的Action。

Q：神经网络是在每一个 step 后学习，还是失败后学习？

A：PPO既不是每个Step更新，

也不是等失败才更新。

它会先收集一批Step产生的经验，

然后利用这一整批经验统一更新神经网络参数。

Q:为什么 ppo_cartpole.py 可以删掉 model 变量后重新运行，而 ppo_cartpole.zip 还能保留训练结果？

A:运行 load_model.py 两次不会自动变强，
因为没有训练。

但如果加载旧模型后继续调用 learn()，
就能在已有参数基础上继续学习。

因此模型文件保存的是训练成果，
代码文件保存的是训练方法。
