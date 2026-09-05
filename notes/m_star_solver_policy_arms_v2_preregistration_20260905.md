# M-star solver policy arms v2 — preregistration (2026-09-05)

在 arms v1 的 post-mortem（`m_star_solver_policy_arms_v1_garching_closeout_20260905.md`）
基础上，残差引导步长换用修订调度，其余协议逐字沿用 v1（同 5 case、同
种子、同臂划分 S1/S2/S2S、cap 60、strict 门、冻结 flux gate、同臂严格
自重启、path consistency；S0 仍以 tomography production 为参照）。

## 调度修订（三条，全部来自 v1 的 A 发散解剖）

1. **武装延迟**：前 `FLUX_RESIDUAL_ARMING_ITERATIONS = 15` 轮步长恒为 1，
   早期瞬态（production 正常的大尖峰）不被干预——v1 中 A 正是在第 3 轮
   被改变轨迹的。
2. **深部分层缩放**：缩放只作用于 log τ ≥ 0.5 的深部超绝热带
   （`FLUX_RESIDUAL_SCALED_LOG_TAU_MIN = 0.5`；tomography 定位的残差
   居住区），浅层保持 production 步长。
3. **保守恢复**：连续 3 轮非恶化才恢复，且 ×1.25（原 ×1.5/2 轮在振荡
   相位产生调度锯齿）。

## 判读规则（与 v1 相同，预注册锁定）

1. **A/C 不回退**（v1 的 S2 死在这里）：两臂下 A/C 必须保持 eligible。
2. **B/D 改善**：S2/S2S 是否使 B 或 D eligible（D 在 v1 已被救活，
   本轮检验修订后是否保持；B 是未决问题）。
3. **E**：发散是否被驯服（形式收敛即可记录，eligible 分开报告）。
4. **采纳标准**：仅当 (1) 满足且 (2) 不劣于 v1 时，才考虑升级；
   升级前须在更大样本上另行预注册。
5. S1 臂本轮不再必要（v1 已否定），但保留作同环境对照。
