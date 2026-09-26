# M 矮星对流内环修复:进展汇总(截至 2026-09-26)

本文汇总 2026-09-22 至 09-25 的 M 矮星内环工作,细节与原始数字见文末各条记录。所有修复都是默认关闭的
开关,生产默认、质量门与已注册阈值均未改动。

## 现状

- 内环候选(五项修复)在原停止判据下:开发点 5 个中 4 个合格,safezone v2 的 8 个独立验证点中 7 个合格,
  主求解通量误差 p95 0.24–1.17%;成对的 S0-pchip 分别为 2/5 与 2/8,p95 6–11%。
- 原停止判据(单轮全层相对温度变化 < 5e-4 即停)下,自重启常在最少的 3 轮停止,自重启一致性检验只能
  发现快速的不静止。改为连续三轮后,开发点 D 与验证点 3900 K(以及 3850 K)的自重启离开主求解;按更严
  的判据,候选的验证合格数为 5/8,原计数偏高两点(3850 K、3900 K),开发点 D 同样由合格变为不合格。
- 同一更严判据下,候选合格的判定点全部起点无关;此前起点依赖的 m−1.0 3800 K 与 4000 K 两起点温度差 p95
  降到 1.7e-4、1.5e-4。
- 已查明三处数值问题:内环掩码内的奇偶交替模式、网格底层的单侧读回、停止判据的盲区;未查明两处:开发
  点 D 对流区顶部的漂移、开发点 E(3200 K)自重启的 column mass 不一致。g4.50 m+0.5 3600 K 是深层周期 2
  过冲,在所有配置下都不收敛。

## 候选的构成

`ARMS['candidate']`(`experiments/reduced_state_emulator/m_star_inner_loop_dev_sweep_20260923.py`):内环 8 遍、
冻结掩码,加五项开关(`payne_zero_atmosphere/config.py`):

| 开关 | 作用 |
|---|---|
| `convection_zone_inner_loop_correct_written_gradient` | 内环写入的后向差分梯度按中心读回所需的变化平移,消除 H₂ 区读回与写入的格式错配(D-plat 上通量过冲 1.7–1.95 倍) |
| `convection_zone_inner_loop_refresh_state` | 内环物理回调在试探温度下重解压力迭代状态(不透明度仍用本轮输入) |
| `convection_zone_inner_loop_hold_correction` | 全局温度修正不动内环设定的层,避免 column mass 重排 |
| `convection_zone_inner_loop_relaxation = 0.5` | 内环退出时只保留一半改动 |
| `convection_zone_inner_loop_fill_holes` | 对流区内辐射无法承载的亚绝热层并入工作掩码 |

单轮链式轨迹上,D 由 0.4349 降到 0.0084,A 由 0.1698 降到 0.0045(输入态深窗均值 |R_raw|)。另有两个检
验过、未进入候选的设置:`convection_zone_inner_loop_filter_written_gradient`(掩码内部去除写入梯度的
(−1)^i 成分)与 `required_consecutive_converged_iterations = 3`(连续三轮停止)。

## 证据链

| 日期 | 阶段 | 结果 |
|---|---|---|
| 09-22–23 | 平台分解与五项修复 | D、A 两条单轮链式轨迹单调下降并收敛 |
| 09-23 | 开发点扫描(A–E) | 候选合格 A、B、C、D,S0-pchip 合格 A、B;候选 p95 0.30–0.67%,S0 6.0–11.5%;E 两臂都因自重启 column mass 不一致而不合格 |
| 09-24 | safezone v2 验证(8 个独立点) | 候选 7/8、S0 2/8;E1、E2 成立;E3(起点无关)4/7 不成立;m+0.5 3600 K 两个起点都不收敛 |
| 09-24 | 起点依赖续算 | m−1.0 3800 K 与 4000 K 停止后继续收拢(停止判据的精度);3850 K 热启动一条不静止 |
| 09-24 | 不静止诊断(12 个单轮分阶段捕获) | 3600 K 为深层对流区的周期 2 过冲,来自内环;3850 K 为内环掩码内的奇偶交替模式,内环在掩码上部同相放大 |
| 09-24 | 交替模式来源 | 写入梯度修正每遍保留已有交替成分;中心读回对它不敏感;H₂ 区 ∇_ad 同相响应并喂养它 |
| 09-24 | 写入梯度滤波,代理模型三轮 | 都消去了交替成分;第三轮(边缘投影,W5e)五项判据满足四项,16 遍能量平衡在网格底层与掩码顶层超限(0.122 对 0.10),路线按预注册终止 |
| 09-24 | 滤波的求解器诊断(3850 K) | 两条续算静止且收敛到同一状态(差距 T 4.6e-4、Δlog m 9.4e-4 dex,TiO 1.2e-4),交替成分消失;网格底层 L79 通量误差 +5.9% |
| 09-24 | 候选加滤波重跑(全部点) | 交替成分在各产品中消失、3850 K 起点无关;D 与 3900 K 退化,L79 通量误差在 12/13 点升到 +3.5–22.5%;滤波不进入候选 |
| 09-25 | 候选改用连续三轮停止重跑(全部点) | 合格判定点全部起点无关;D 与 3900 K 退化,原因是原判据下自重启 3 轮即停掩盖了不静止;停止判据暂不进入候选协议 |

## 三种配置的逐点结果

候选 = 原候选;候选-F = 候选加写入梯度滤波;候选-S = 候选加连续三轮停止。格子内为是否合格与主求解通量误差
p95(%);验证点另注起点无关性(无关 / 依赖)。

| 点 | 候选 | 候选-F | 候选-S | S0-pchip |
|---|---|---|---|---|
| A g4.50 m+0.0 3500 K | ✓ 0.297 | ✓ 0.442 | ✓ 0.296 | ✓ 8.80 |
| B g4.50 m+0.0 3400 K | ✓ 0.333 | ✓ 0.364 | ✓ 0.314 | ✓ 9.06 |
| C g4.50 m+0.0 3300 K | ✓ 0.349 | ✓ 0.357 | ✓ 0.314 | ✗ 10.62 |
| D g4.50 m−0.5 3600 K | ✓ 0.667 | ✗ 1.699 | ✗ 0.476 | ✗ 11.46 |
| E g4.50 m+0.0 3200 K | ✗ 0.299 | ✗ 1.345 | ✗ 0.303 | ✗ 6.00 |
| g4.50 m−1.0 3800 K | ✓ 0.315 依赖 | ✓ 0.788 依赖 | ✓ 0.248 无关 | ✗ |
| g4.50 m+0.5 3600 K | ✗ | ✗ | ✗ | ✗ |
| g4.75 4000 K | ✓ 0.259 依赖 | ✓ 0.296 依赖 | ✓ 0.268 无关 | ✓ 6.20 |
| g4.75 3950 K | ✓ 0.342 无关 | ✓ 0.454 无关 | ✓ 0.250 无关 | ✗ |
| g4.75 3900 K | ✓ 0.251 无关 | ✗ 0.407 | ✗ 0.231 | ✓ 8.88 |
| g4.75 3850 K | ✓ 1.169 依赖 | ✓ 0.244 无关 | ✗ 0.691 | ✗ |
| g4.75 3800 K | ✓ 0.456 无关 | ✓ 0.556 无关 | ✓ 0.271 无关 | ✗ |
| g4.75 3750 K | ✓ 0.242 无关 | ✓ 0.548 无关 | ✓ 0.322 无关 | ✗ |
| g4.50 m+0.0 3400 K(与 B 重叠) | ✓ 0.952 | ✓ 0.311 | ✓ 0.609 | ✗ |

| 合计 | 候选 | 候选-F | 候选-S | S0-pchip |
|---|---|---|---|---|
| 开发点合格 | 4/5 | 3/5 | 3/5 | 2/5 |
| 独立验证点合格 | 7/8 | 6/8 | 5/8 | 2/8 |
| 合格独立点中起点无关 | 4/7 | 4/6 | 5/5 | — |

## 已查明的机制

1. **内环掩码内的奇偶交替模式。** 写入梯度修正使一遍的 ln T 变化等于"目标 − 读回"的向内累加,已有的
   (−1)^i 成分原样保留;中心读回对它不敏感(L52–72 上读回与写入的交替成分之比 ≤ 0.08);H₂ 离解区 ∇_ad 对它同
   相响应(交替幅度之比 0.29–0.85),每遍注入同相成分。候选各产品的交替幅度为 2e-4–2.6e-3,S0 ≤ 1.4e-4。
   它使 3850 K 与 3900 K 不静止。滤波可以去除它,代价见第 2 条。
2. **网格底层的单侧读回。** `differentiate_on_depth_grid` 在末层用单侧差分;未滤波的写入梯度修正带着一
   个由底部向上衰减的奇偶波纹补偿它。滤波去除这一波纹后,L79 通量误差升到 +3.5–22.5%(开发点 E 为 22.5%,
   距冻结通量门的最大值限 23.15% 只差 0.65)。
3. **停止判据的盲区。** 已记录的候选求解,每一次停止都只由一轮满足阈值触发;改为连续两轮或三轮时,没有一
   个会在原处停止。连续三轮时自重启最早在第 5 轮
   停止;稳定的点都停在第 5 轮(B 为第 6 轮),离开主求解的恰是自重启更久的四点:3900 K(12 轮)、D、
   E、3850 K(各 8 轮)。m−1.0 3800 K 与 4000 K 的起点依赖来自停止判据的精度。
4. **m+0.5 3600 K 的深层周期 2 过冲。** 深层对流区(L48–79)整体过冲,内环一步逐轮换号放大 1.9–3.2 倍,
   饱和后内环 ±2.0e-2、重映射同号再加约一半;候选从 S0 已收敛的参照解出发也会离开。

## 未查明

- **开发点 D 对流区顶部的漂移。** 连续三轮停止下,D 的自重启使 L36–43(紧邻亚绝热段 L42–45,掩码在 L44
  断开,填洞在此作用)降温 0.35–0.38%,p95 由 0.48% 变为 0.65%,交替成分小。
- **开发点 E(3200 K)自重启的 column mass 不一致。** 各配置下都不合格,开发点扫描中两臂失败方式相同。

## 下一步(未开始)

1. 诊断 D 的对流区顶部漂移:从候选-S 的主求解产品续算 10 轮(停止判据关闭),逐轮记录填洞层数、掩码与逐
   层改动,再对漂移中的几轮做分阶段捕获,区分内环、全局修正与重映射。
2. 在 D 的诊断之后,另行预注册"滤波 + 连续三轮停止":两者分别覆盖对方的失效(交替模式;停止过早),但网
   格底层的误差与 D 的漂移不在其中。
3. 网格底层单侧读回的边界处理。
4. 是否把更严的停止判据作为收紧协议采用:它只会使合格更难,但与 S0 的成对比较需要 S0 在同一判据下重跑。

## 代码、测试与运行

- 求解器:`payne_zero_atmosphere/convection_inner_loop.py`、`runner.py`、`temperature_correction.py`、
  `config.py`、`run_setup.py`;新增开关都默认关闭。单元测试 `tests/test_convection_inner_loop_*.py`、
  `tests/test_opacity_lagging.py`。
- 驱动:开发点扫描 `m_star_inner_loop_dev_sweep_20260923.py`(臂 `candidate`、`candidate_filter`、
  `candidate_strict_stop`、`s0`),验证 `m_star_inner_loop_validation_20260924.py`,续算、分阶段捕获、代理模型
  与诊断各有独立驱动;读数脚本在 `experiments/analyze_m_star_inner_loop_*`。
- 运行:重计算在 Garching(Node-04、05、08;共享 /nexus),单线程,每个求解进程约 20 GB 常驻内存,按节点
  空闲内存定并发数。结果目录 `results/m_star_inner_loop_*`(不入库),冻结热启动
  `results/m_star_inner_loop_validation_20260924/inputs/warm_starts.npz` 入库。
- 本线提交在 `codex/sync-20260826` 分支,自 `950d3ec` 至 `188093a`,均未推送。

## 记录索引

| 阶段 | 记录 |
|---|---|
| 平台与起点 | `m_star_platform_decomposition_20260923.md`、`m_star_platform_start_points_20260922.md`、`m_star_inner_loop_s3_20260922.md` |
| 五项修复 | `m_star_inner_loop_written_gradient_*`、`m_star_inner_loop_state_refresh_*`、`m_star_inner_loop_hold_correction_*`、`m_star_inner_loop_relaxation_*`、`m_star_inner_loop_fill_holes_*`;诊断 `m_star_s3wr_oscillation_diagnosis_20260923.md`、`m_star_d_round0_diagnosis_20260923.md` |
| 开发点扫描 | `m_star_inner_loop_dev_sweep_{preregistration,closeout}_20260923.md` |
| 验证 | `m_star_inner_loop_validation_{preregistration,closeout}_20260924.md` |
| 起点依赖续算 | `m_star_inner_loop_start_dependence_continuation_{preregistration,closeout}_20260924.md` |
| 不静止与交替模式 | `m_star_inner_loop_nonstationary_diagnosis_20260924.md`、`m_star_inner_loop_alternating_mode_source_20260924.md` |
| 滤波代理模型 | `m_star_inner_loop_filtered_written_gradient_surrogate_*`、`m_star_inner_loop_interior_filtered_written_gradient_surrogate_*`、`m_star_inner_loop_edge_projected_filter_*` |
| 滤波求解器诊断与重跑 | `m_star_inner_loop_solver_filter_diagnostic_{plan,closeout}_20260924.md`、`m_star_inner_loop_filter_rerun_{preregistration,closeout}_20260924.md` |
| 连续三轮停止 | `m_star_inner_loop_strict_stop_{preregistration,closeout}_20260925.md` |
