# M-star pipeline complete v1 — preregistration (2026-09-05)

## 目标

scaleout v1 矩阵中 5 个未认证节点 + 1 个被撞名跳过的 cap 扫描，全部
重跑至认证。物理与冻结 gate 不动；允许的仅有两个已实现的认证策略
工具：`require_improving_flux_residual` 停止策略（solver 持续迭代直至
"ΔT 达标且 p95 残差非恶化"的停止点，cap 放宽至 120）与 continuation
种子路由（从最近已收敛节点细步长进入）。

## 节点与策略

| 节点 | v1 失败模式 | 本轮策略 |
| --- | --- | --- |
| giant 4000/2.5 | restart 60 轮不停（flux max 3.1，非发散） | restart 相位感知 + cap 120 |
| dwarf 3800/[M/H]−1.0 | primary 停在恶化相位（restart 干净） | 双腿相位感知 + cap 120 |
| dwarf 4000/[M/H]0 | 双腿停在恶化相位（残差 0.4%） | 双腿相位感知 + cap 120 |
| dwarf 3600/[M/H]−1.0 | MARCS 直跳发散 | continuation：从已收敛 3800/−1.0 产品 25→12.5→6.25 K 走到 3600，目标认证相位感知 |
| dwarf 3500/[M/H]0 | MARCS 直跳发散（corridor 已知） | continuation：从已过门 3600 产品作种子直达 3500，相位感知认证 |
| giant 3750/2.5 (cap 扫描) | node_id 撞名被跳过 | 唯一 node_id，cap 30/120 补扫描（60 已有） |

## 认证

与 scaleout 完全一致：双腿 survives + 冻结 flux gate + path
consistency + 双腿 `flux_residual_improving_at_stop == True`。
cap 上限 120 仅用于给相位感知停止留预算；结论若依赖 cap=120 而非
60，须在 closeout 中明确标注。

## 判读

1. 逐节点报告 certified 与否、迭代数、三指标、双腿守卫。
2. 相位感知停止的代价（多花多少轮）单独记录。
3. 若 continuation 入口仍不能使 3600/−1.0 或 3500/0 收敛，记录发散
   签名并判为该入口下关闭（不反复试错）。
