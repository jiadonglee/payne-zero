# Oracle cross 与 textbook opacity 实验记录

日期：2026-08-26

状态：`FAIL_STOP`。本轮没有把候选物理 ODE 初值接入 production solver，也没有读取或修改 sealed holdout。

## 1. Oracle 归因

两臂使用 Paper-II 已验证的
`reduced_state.reconstruct.reconstruct_full_atmosphere(seed="physical")`，然后用 unchanged 15-iteration solver restart。两臂只交换 `(m,T)` 的来源；压强、电子密度、opacity 和 transfer 均由同一重建链生成。

| corpus index | Teff / log g | grey T + truth m | truth T + grey m | truth T + truth m control |
|---:|---:|---:|---:|---:|
| 2891 | 4398 K / 1.53 | 15 步未收敛，最终非有限 | 15 步未收敛，有限，deep ΔT/T = 1.81e-3 | 3 步收敛，有限 |
| 25948 | 7246 K / 5.03 | 15 步未收敛，最终非有限 | 15 步未收敛，有限，deep ΔT/T = 3.75e-2 | 3 步收敛，有限 |

结论仅限于这两个 diagnostic 点：truth–truth control 排除了重建链本身失效；两个交叉臂均失败，因此不能支持“灰体失败主要由质量通道单独造成”的强归因。`truth T + grey m` 的有限性改善也不能称为 solver 通过。

## 2. Named-constant textbook opacity

实现了 Saha(H、Na、K、Ca、Mg、Fe) + H− bf/ff + H bf + Kramers + electron scattering，以及正值 `dm/dtau = 1/kappa(T,g*m)` log-ODE。该模块只作为离线候选，未接入生产入口。

在排除既有 manifest 的 10,228 个 development validation 星上：

| 指标 | pooled p95 |
|---|---:|
| `kappa` log10 误差 | 5.706 dex |
| 用真值 `P,T` 积分 `m` 的 log10 误差 | 5.727 dex |
| 冷端 `<6000 K` 的 `kappa` 误差 | 6.129 dex |
| 5500–7000 K 的 `kappa` 误差 | 5.034 dex |
| 6000–10000 K 的 `kappa` 误差 | 4.289 dex |

预设离线 gate 为冷端 p95 ≤ 0.30 dex、中段 p95 ≤ 0.50 dex；结果为 `FAIL`。因此问题已在 local opacity 层暴露，不应归咎于 ODE 的定点/积分结构，也不应继续跑 solver funnel。组件分解中 Kramers 项占 pooled 中位总 opacity 的约 0.743，但这只是诊断，不足以证明唯一因果来源；下一轮若重开，需先补低温/低密度 opacity 物理并重新注册常数。

## 3. 可复现实验产物

- [oracle cross runner](../experiments/analytic_initializer/run_oracle_cross.py)
- [textbook opacity implementation](../experiments/analytic_initializer/textbook_opacity.py)
- [offline validation runner](../experiments/analytic_initializer/run_textbook_opacity_offline.py)
- [corrected oracle 2891 JSON](../results/analytic_initializer/oracle_cross_reconstructed_2891.json)
- [corrected oracle 25948 JSON](../results/analytic_initializer/oracle_cross_reconstructed_25948.json)
- [truth control 2891 JSON](../results/analytic_initializer/oracle_control_reconstructed_2891.json)
- [truth control 25948 JSON](../results/analytic_initializer/oracle_control_reconstructed_25948.json)
- [full offline opacity validation](../results/analytic_initializer/textbook_opacity_offline_validation.json)

新增构造测试与既有 analytic-initializer/grey-start 测试合计 `148 passed`。
