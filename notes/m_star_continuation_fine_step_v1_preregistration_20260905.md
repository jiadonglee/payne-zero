# M-star continuation fine step v1 — preregistration (2026-09-05)

## 目标

3200 K（轨 A：logg 4.5、[M/H] 0）是 continuation 网络上最后一个关闭
格。旧 walk 的 50 K 步形式收敛但 gate 失败（16.4%），25 K 减半步从
3400 直接发散。本轮用两条夹击臂 + 认证相位守卫重试，求解器物理与
production 完全一致。

## 臂

- **down**：链种子 = tomography 已过门 `…_t3400` 产品；25 K 步长
  （3375 → 3350 → 3325 → 3300 → 3275 → 3250 → 3225 → 3200），失败减半，
  下限 12.5 K。
- **up**：链种子 = tomography 已过门 `…_t3300` 产品；25 K 步长
  （3275 → 3250 → 3225 → 3200），失败减半，下限 12.5 K。

航点验收：solver 形式收敛 + 六场有限（复用 `survives_solver`），不设
flux gate（航点不是训练格）。任一步在下限步长仍失败，该臂关闭。

## 目标格认证（3200）

primary + 同路径严格自重启（仅 (m,T)、新 carry、production）+ 冻结
flux gate 双腿 + path consistency + **认证相位守卫**：双腿的
`flux_residual_improving_at_stop` 必须为 True（求解器本轮新增的观测
量——记录停止那一轮 p95 flux 误差相对上一轮非恶化；tomography 已证
明同格点的 gate 生死取决于残差相位，故认证必须排除相位运气）。
该量缺失（未收敛）即自动不合格。

## 判读

1. 任一臂 certified 3200 即成功，记录走通的角度与步长史。
2. 双臂均关闭则 3200 记为"production+continuation 下关闭"，分歧点
   （发散发生的步长与来源温度）写入 closeout。
3. 相位守卫在双腿的判定结果分别报告；若出现 primary 过门但守卫
   False 的情况，该格不 certified——这正是本守卫要排除的情形。
