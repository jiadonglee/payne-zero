# M-star solver policy arms v1 — preregistration (2026-09-05)

## 问题

tomography 与续跑实验已把失败形态分好类：

- A（3500）/C（3300）：production 停止判据在残差下降相位提前触发，
  多跑会更好——"停早了"。
- B（3400）/D（3600 mp）：停止后继续迭代，残差以 4 倍振幅振荡
  （10–39%），既不收敛到更低也不发散——**步长/全球化问题**。
- E（3200，从 3400 continuation 一步）：production 步进直接发散
  （p95 ~5×10⁵%）——迭代失稳。

本轮检验两个默认关闭的实验开关能否在**不改物理、不动验收门槛**的
前提下改善这三类：

- **S1**：`temperature_correction_damping=0.5`（全局定阻尼，对照臂）。
- **S2**：`flux_residual_guided_damping=True`（残差引导步长：p95 flux
  误差相对上轮恶化超过 50% 则步长减半（下限 0.125），连续 2 轮非恶化
  则 ×1.5 恢复至 1）。
- **S2S**：S2 + `require_improving_flux_residual=True`（收敛附加条件：
  本轮 p95 不得比上轮差）。

S0 = 既有 tomography production 结果，作为对照不重跑。

## 集与门槛

- Case：A/B/C/D（tomography 同一构建路径的种子，array sha256 已冻结）
  + E（种子 = tomography 已过门 t3400 产品的 (m,T)，目标 3200 K，
  轨 A，即 walk 关闭格）。
- 主解 cap 60、strict all-layer 5e-4，与 tomography 相同；冻结 flux gate
  不变（p95 ≤ 9.557% 等）。
- 每个幸存者跑**同臂严格自重启**（同臂设置、新 carry、仅 (m,T)）+
  path consistency（温度 p95 ≤ 3.0e-3，column mass ≤ 7.7e-3 dex），
  eligibility 判式与 tomography 完全相同。

## 判读规则（预注册）

1. **B/D**：S2 或 S2S 是否让停止时刻的残差落在门内且自重启同侧
   （eligible）？S1 作对照：纯定阻尼是否同样有效。
2. **E**：任何臂能让 3200 K 形式收敛（不发散）即记录；eligible 与否
   分开报告。
3. **A/C 不回退**：若某臂使 A/C 失去 eligible，该臂判负。
4. **采纳标准**：仅当 (3) 满足且 (1) 或 (2) 有改善时，才考虑把臂升级
   为候选 production 策略；升级前须在更大样本上另行预注册。
5. 非目标：不改物理、不改 gate 阈值、不改 production 默认路径
   （两开关默认关闭，全套回归 + 位级 parity 已由
   `tests/test_flux_residual_policy.py` 与回归套件钉住）。
