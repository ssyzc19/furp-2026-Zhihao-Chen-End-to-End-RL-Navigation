# Day 9 — obs / action 来源与 PPO 调用链

## obs 从哪来
来自 Habitat Environment。`PPOTrainer._init_envs()` 通过 `env_factory.construct_envs()` 创建环境，再用 `envs.step()` 拿到 obs。

## action 从哪来
由 Agent 里的 Policy Network 产生。`_create_agent()` 按配置 `rl.agent.type` 从 Registry 取 Agent 类实例化。

## 训练调用链
```
trainer.train()
 ├── _create_agent()                 → SingleAgentAccessMgr(actor_critic + rollouts + PPO updater)
 └── while training:
      ├── _collect_rollout_step()    → actor_critic.act() → envs.step() → rollouts.insert()
      ├── compute_returns()          → GAE
      └── _update_agent()            → updater.update(rollouts)
     save_checkpoint()
```

## 1000 步 smoke test 日志（验证管线）
```
update 10  reward -0.057
update 20  reward -0.038
update 30  reward -0.039
```
说明：管线跑通（rollout→update→log），1000 步太短，success/spl 仍为 0，属正常。
