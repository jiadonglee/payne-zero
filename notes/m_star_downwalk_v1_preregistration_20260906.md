# M-star downwalk v1 — preregistration (2026-09-06)

## 目标

complete-v1 关闭了 3600/[M/H]−1.0 的"从 3800 向上"入口（25/12.5/6.25 K
三档全发散）。尚有一条界内路线未试：**从已认证的 4000/−1.0 产品沿轨
向下**走 16 个 25 K 航点至 3600 K。本轮预注册该下行 walk。

## 方法

- 链种子：scaleout v1 已认证的 `dwarf_g+4.50_m-1.00_t4000` cap60
  primary 产品（(m,T) only，认证重建路径）。
- 步长：25 K，失败减半至 12.5 K；低于下限失败即关闭该 walk。
- 航点验收：solver 形式收敛 + 六场有限（`survives_solver`），production
  物理、cap 60，无 flux gate（航点非训练格）。
- **目标格 3600 认证**：primary 用相位感知停止
  （`require_improving_flux_residual`，cap 120）+ 严格自重启（同策略）
  + 冻结 flux gate 双腿 + path consistency + 双腿
  `flux_residual_improving_at_stop == True`。与 complete-v1 认证标准
  完全一致。

## 判读

1. walk 走通且目标认证通过 → 3600/−1.0 certified，记录每步迭代数与
   残差（下行与上行失败位置的对比是走廊几何的直接证据）。
2. 任一步在下限步长失败 → 记录发散签名（来源温度、步长、迭代数、
   末轮残差），walk 关闭；该节点在"物理与冻结 gate 不动"的前提下
   判为无界内路线，**交由用户裁决**：放宽该节点门槛、或改动物理
   （single-layer parity 之后的决定）。
3. 本轮不引入任何新的求解器策略。
