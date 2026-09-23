# 内环候选的开发点扫描预注册(2026-09-23)

对象:`notes/m_star_inner_loop_fill_holes_trajectories_closeout_20260923.md` 中在 A、
D 两条单轮链式轨迹上收敛的内环候选。本次改用常规求解(一次运行内多轮迭代、原停
止判据),在 2026-09-18 内环预注册与 S3 v1 驱动(`m_star_convection_inner_loop_v1.py`)
定义的五个开发点上,按冻结通量门与 TiO 双路径检验评估。热力学用 H₂ 配分函数的
pchip 插值包副本(与 2026-09-22/23 各臂相同,`build_overlay('pchip')`);混合长度、
不透明度与生产默认不变。

## 候选

写入梯度修正、试探温度状态重算、对流层修正保持、λ = 0.5、填洞,即
`convection_zone_inner_loop_passes = 8`、`freeze_mask = True`,以及
`correct_written_gradient`、`refresh_state`、`hold_correction`、`relaxation = 0.5`、
`fill_holes` 五个开关。

## 开发点与起点

| 点 | 轨道 | Teff | 起点 |
|---|---|---|---|
| A | g+4.50 m+0.00 | 3500 K | tomography 冻结种子 |
| B | g+4.50 m+0.00 | 3400 K | tomography 冻结种子 |
| C | g+4.50 m+0.00 | 3300 K | tomography 冻结种子 |
| D | g+4.50 m−0.50 | 3600 K | tomography 冻结种子 |
| E | g+4.50 m+0.00 | 3200 K | tomography 3400 K 产品的 (m, T),按 3200 K 标签重建 |

种子由 `m_star_solver_policy_arms_v2._case_seed` 重建,与 S0 tomography、policy
arms v2 与 S3 v1 相同。

## 两臂(成对)

- 候选:见上。
- S0:内环关闭,同一 pchip 包、同一种子与协议,在本次扫描中同时运行,作为成对对照。

## 协议(与 S3 v1 驱动相同)

- 主求解:上限 60 轮,停止判据为全层相对温度变化 ≤ 5e-4。
- 自重启:主求解存活且有产品时,从其 (m, T) 重建后再求解一次,同样上限与判据。
- 冻结通量门(`results/m_star_iteration_tomography_v1/flux_gate.json`,gate_hash
  `ae0d384e…`):p95 ≤ 9.557%、中位数 ≤ 0.234%、最大值 ≤ 23.15%,且求解器存活;主求
  解与自重启分别判定。
- 路径一致性:`m_star_bootstrap_v1._product_consistency`(主求解与自重启产品)。
- TiO 双路径:主求解与自重启产品各合成 665–667 nm、R = 20000、含分子谱线的光谱
  (`emulator_v1_2.gates.compare_spectra._synthesize_one`),归一化通量最大绝对差、
  按连续谱缩放的总通量最大差、连续谱最大相对差三者都 ≤ 5e-3。
- 一点合格 = 主求解存活 ∧ 主求解过通量门 ∧ 自重启过通量门 ∧ 路径一致 ∧ TiO 双路
  径过线。TiO 双路径检验同一臂两条路径的一致性,不是物理准确度。

驱动 `experiments/reduced_state_emulator/m_star_inner_loop_dev_sweep_20260923.py`(复用
S3 v1 的种子、求解、通量门与一致性函数,加 TiO 步骤)。Garching Node-05 单线程,每
臂每点一个进程(共 10 个,预计 2–4 小时),产物在
`results/m_star_inner_loop_dev_sweep_20260923/`。

## 历史参照(线性插值生产 EOS,不重跑)

`notes/m_star_iteration_tomography_v1_garching_closeout_20260905.md`(通量门与路径一
致性,未做 TiO):A 合格(38 轮,p95 7.58%)、B 不合格(自重启 p95 17.54%)、C 合格
(28 轮,p95 8.61%)、D 不合格(主求解 p95 11.56%);E 在 S0 与 policy arms v2 各臂均
发散(`notes/m_star_solver_policy_arms_v2_garching_closeout_20260905.md`)。判据只用本
次成对的 S0-pchip。

## 判据(不因结果调整)

- E1(无退化):S0-pchip 合格的每一点,候选也合格。
- E2(改进):S0-pchip 不合格的点中,候选至少一点合格。
- S0-pchip 五点全部合格时 E2 不适用,只判 E1;全部不合格时 E1 不适用,只判 E2。
- 诊断(不判):各点主求解轮数与 p95;两臂都合格的点上候选与 S0 产品的 TiO 光谱差
  (收敛到同一大气时应在 5e-3 以内);S0-pchip 与历史线性插值 S0 的差异。

## 决策表

| 结果 | 下一步 |
|---|---|
| E1 与 E2 成立 | 候选在开发集上改进且无退化;下一步:safezone v2 的 9 个验证点(通量门 + TiO,另行预注册);开关仍默认关闭 |
| E1 成立,E2 不成立 | 通量门层面无改进;逐点查轮数、p95 尾部与 TiO |
| E1 不成立 | 候选在原本合格的点上退化;先诊断该点,不进入验证点 |

单轮链式轨迹与常规多轮求解的差别(跨轮携带的修正记录、不透明度与状态)是本次
扫描要检验的内容之一;A、D 在两种方式下的差异单独报告。生产默认、质量门与已注册
阈值不因本结果改变。
