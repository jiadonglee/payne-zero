# 候选加写入梯度滤波在开发点与验证点上的重跑预注册(2026-09-24)

对象:`notes/m_star_inner_loop_solver_filter_diagnostic_closeout_20260924.md`。在 g4.75 3850 K 上,
候选加 `convection_zone_inner_loop_filter_written_gradient`(掩码内部去除写入梯度的 (−1)^i 成分)后,
两条续算静止并收敛到同一状态;网格底部 L76–79 仍有通量误差(L79 +5.9%)。本次在开发点扫描与
验证的全部点上,用与原候选相同的协议重跑"候选 + 滤波"(下称候选-F),判断滤波能否进入候选。
开关默认关闭,生产默认不变。

## 臂与对照

- 候选-F:开发点扫描驱动的 `ARMS['candidate']` 加 `convection_zone_inner_loop_filter_written_gradient = True`,
  pchip 包。
- 对照(不重跑):原候选与 S0-pchip 在开发点扫描(`notes/m_star_inner_loop_dev_sweep_closeout_20260923.md`)
  与验证(`notes/m_star_inner_loop_validation_closeout_20260924.md`)中的结果。滤波开关默认关闭,只在
  打开时进入新分支;S0 不经过内环。两者的代码路径与当时相同,因此可作成对对照。

## 点与协议(与原协议相同)

- 开发点 A–E(`notes/m_star_inner_loop_dev_sweep_preregistration_20260923.md`):tomography 种子,主
  求解与自重启,上限 60 轮,全层相对温度变化 ≤ 5e-4,冻结通量门(gate_hash `ae0d384e…`),路径一
  致性,TiO 双路径。
- 验证点(`notes/m_star_inner_loop_validation_preregistration_20260924.md`):冻结热启动
  `results/m_star_inner_loop_validation_20260924/inputs/warm_starts.npz` 的主求解与自重启,参照解产品为
  起点的求解,判定与配对与验证相同;8 个独立点,g4.50 m+0.0 3400 K 单独报告。

## 判据(不因结果调整)

- F1(开发点不退化):原候选合格的开发点(A、B、C、D),候选-F 都合格。
- F2(验证点不退化):原候选合格的独立验证点(m−1.0 3800 K 与 g4.75 的 3750–4000 K 共 7 点),候
  选-F 都合格。
- F3(起点无关):候选-F 合格的每个独立验证点都起点无关(定义同验证的 E3)。
- 同时报告相对 S0-pchip 的 E1、E2(定义同原预注册)。

## 诊断(不判)

- 各点网格底部 L74–79 的逐层通量误差与 L0–73 的最大值(主求解终态)。
- 各产品 ln T 的交替幅度(L45–51、L52–58、L59–65、L66–72、L73–78)。
- 主求解轮数、通量误差 p95;与参照解的 TiO 差(验证点)。
- g4.50 m+0.5 3600 K(深层过冲,不在滤波的作用范围内)与开发点 E(3200 K,自重启 column mass)
  的状态。

## 决策表

| 结果 | 下一步 |
|---|---|
| F1、F2、F3 成立 | 候选-F 取代原候选作为 M 矮星的研究候选(仍默认关闭);下一步为网格底部边界与 3600 K 过冲,之后另行预注册库产品生成 |
| F1、F2 成立,F3 不成立 | 对起点依赖的点做续算检验(与 2026-09-24 相同),区分停止判据的精度与真正的起点依赖;前者进入更严停止判据的预注册 |
| F1 或 F2 不成立 | 滤波在该点退化;先诊断该点,原候选不变 |

## 运行

Garching 单线程;验证点 18 个进程(热启动 9、参照起点 9)与开发点 5 个进程分在两个节点上,按空
闲内存(约 20 GB/进程,占可用内存一半以内)选节点;验证点结束后运行一次配对。产物在
`results/m_star_inner_loop_filter_rerun_20260924/{dev,validation}/`,读数脚本
`experiments/analyze_m_star_inner_loop_filter_rerun_20260924.py`。开发点扫描与验证驱动加可选的
`--preregistration` 参数,使身份记录指向本预注册;其余行为不变。

## 范围

单一 pchip 包、单一混合长度;验证点中 6 点在同一 g4.75 轨道上。候选-F 的停止判据与原候选相同。
生产默认、质量门与已注册阈值不因本结果改变。
