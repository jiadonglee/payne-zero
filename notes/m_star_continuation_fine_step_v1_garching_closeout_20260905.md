# M-star continuation fine step v1 — Garching closeout (2026-09-05)

Preregistration: `m_star_continuation_fine_step_v1_preregistration_20260905.md`。
Driver: `experiments/reduced_state_emulator/m_star_continuation_fine_step_v1.py`。
wall 1755.7 s（双臂并行）。

## 3200 K：双臂夹击，走廊关闭

- **down**（种子 3400）：3400→3375 过（13 轮，p95 18.5%），3375→3350 过
  （22 轮，25.4%），3350→3325 发散（60 轮，p95 7.8e6）；12.5 K 减半重试
  （3350→3337.5）仍发散（4.0e6）。`minimum_step_failed`。
- **up**（种子 3300）：3300→3275 第一步即发散（60 轮）；12.5 K 重试
  （3300→3287.5）发散。`minimum_step_failed`。

结论：**3275–3325 K 走廊在 production 步进下从两侧动力断连**——12.5 K
步长仍发散（60 轮冲到 ~10⁶），这不是步长噪声敏感性，是盆断连。3200 K
在 production + continuation 框架下记为关闭；若要突破需先回答断连区
的物理/数值出身（single-layer parity，原 P3），或引入非 production 的
步进策略（已有两轮 arms 证据表明全局方案不支配 production）。

## 认证相位守卫：实现、验证、辨别力

求解器新增观测量 `flux_residual_improving_at_stop`（收敛轮的 p95 flux
误差相对上轮非恶化；仅 `enable_convergence_stop` 且收敛时写入
diagnostics，默认 payload 其余不变）。同轮验证：

1. **位级惰性**：新 runner 重跑 3400 K production 直跳，30 轮、
   p95 9.14612741827599 与存档逐位相同，产品 max|ΔT| = 0.0。
2. **辨别力**（同格点 3400 K、同 gate）：
   - production 直跳：停止轮 p95 9.15，守卫 **False**——停止抓在恶化
     相位上，与其重启失败一致；若要求守卫，该路径不可认证。
   - continuation 路径（3500→3450→3400）：primary 15 轮 p95 8.06 守卫
     **True**，restart 3 轮 p95 3.82 守卫 **True**——认证在守卫下成立。
3. 3500→3450 航点守卫 False（20 轮，p95 6.95）不构成问题：航点只要求
   solver 收敛 + 有限状态，不参与认证。

## 冷星稳定求解现状（轨 A：logg 4.5、[M/H] 0）

- **certified 稳定区间：3300–3500 K**（3500/3300 production 直接过门，
  3400 经 continuation 且双腿过相位守卫）。
- **3200 K 及以下：关闭**。3275–3325 走廊双侧动力断连（本轮），
  3100–3200 插值直跳此前也已失败。
- 稳定求解的充产出要素：production 求解器 + continuation 网络 +
  冻结 gate + 相位守卫认证。后续任何新格点按此流程预注册。
