# Day 13 — Policy 架构（PointNavResNetPolicy）

## 视觉编码
- RGB → ResNet18
- Depth → 复制成 3 通道 → ResNet18
- 两路特征 → Global Avg Pool → **视觉特征 (512维)**

## PointGoal 编码
- PointGoal(GPS+Compass)：(distance, angle) → (distance, cosθ, sinθ) → Linear → **Goal 特征 (32维)**

## 上一步动作编码
- PrevAction → Embedding → **Action 特征 (32维)**

## forward 总流程
```
Visual(512) + Goal(32) + Action(32)
        │ torch.cat → 576维
        ▼
       GRU → 512维隐藏状态（记忆）
        │
   ┌────┴────┐
 Actor     Critic
   │          │
 Action     Value
```

> 对比 CartPole 的 MlpPolicy（4维→MLP）：这里多了**视觉编码(ResNet)**和**记忆(GRU)**。
