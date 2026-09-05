# M-star solver policy arms v1 — Garching closeout (2026-09-05)

Preregistration: `m_star_solver_policy_arms_v1_preregistration_20260905.md`.
Driver: `experiments/reduced_state_emulator/m_star_solver_policy_arms_v1.py`.
Results: Garching（已回传本地同名目录）5 case × 3 arms，全部 15 个主解
+ 幸存者同臂严格自重启，wall 2378 s + E 补跑 1302 s（E 首轮因脚本
`CONTINUATION_CASE` 漏 `seed_source` 键报错，修复后单独重跑）。

## 矩阵（p95 = primary flux p95；S0 = tomography production 参照）

| case | S0 | S1 (阻尼0.5) | S2 (残差引导) | S2S (S2+停止伴随) |
| --- | --- | --- | --- | --- |
| A 3500 | ✓ 7.58 (38轮) | ✗ 16.88 (42) | ✗ **发散** 2095 (60) | ✗ 同 S2 |
| B 3400 | ✗ 9.15* (30) | ✗ 13.98 (44) | ✗ 16.89 (39) | ✗ 16.89 (39) |
| C 3300 | ✓ 8.61 (28) | ✗ 17.33 (49) | ✓ 9.23 (32)，重启 8.37 | ✓ 同 S2 |
| D 3600mp | ✗ 11.56 (25) | ✗ 11.44 (55)，重启爆炸 8e4 | ✓ **7.56 (31)，重启 5.36** | ✓ 同 S2 |
| E 3200 | 发散 (walk) | 形式收敛但 90.2 (40) | ✗ 发散 7767 (60) | ✗ 同 S2 |

*B 本轮 primary 过门但 restart gate 失败（刀尖点，见 tomography closeout）。

## 判读

1. **S1 全线劣于 production**：A/C 从过门变失败，B/D 无改善，D 的
   restart 爆炸。固定全局阻尼被否定。
2. **S2 双刃，按预注册采纳标准判负**：它救活了设计目标 D——production
   下唯一的稳定 gate-failer 变成 eligible（7.56/5.36，D 迄今最好结果），
   并保住 C；但把稳健的 A 踢进了发散盆（A 的 S2S 与 S2 逐位相同），
   B 仍未解决。触犯"A/C 不回退"，**不升级为 production 候选**。
3. S2S ≡ S2（本轮所有点停止伴随条件都不是决定项）。
4. A 的 S2 发散机制（逐轮步长+残差轨迹）：调度器在**第一个正常弹跳**
   （12→76%，production 同相位弹到 365%）就减半，轨迹从此偏离
   production 的"骑过尖峰再下降"路径；第 24 轮第二次弹跳后进入
   全局 α 压不住的发散振荡（α=0.125 仍增长），恢复段呈 0.42↔0.84
   锯齿——调度器自身在振荡。
5. E：全局步长引导驯不住 3200 K 的发散；S1 只能把它压成无意义的
   形式收敛（90%）。

## 下一轮的设计假设（记录，未实现）

- **武装延迟**：残差引导不应在早期瞬态（前 ~15 轮的大尖峰期）动作；
  A 正是在那里被改变轨迹的。
- **深部分层缩放**：振荡只住在 log τ 1.25–3.0 的 4–8 层；全局 α 波及
  无辜表面层。只对该深度带缩放是靶向版本。
- **恢复更保守**：×1.5/2 轮恢复在振荡相位产生调度锯齿。
- B 的出路更可能是盆定向的 continuation（已被 probe A 证明有效），
  而不是全局步长。

## 遗留

- arms 的 `_case_worker` 无 skip-existing 逻辑（重跑即重解）；
  `CONTINUATION_CASE` 漏键已修复。
- 两开关保持默认关闭；production 路径由回归套件（505 通过）与位级
  parity 钉住。
