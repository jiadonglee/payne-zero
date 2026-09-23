# 内环写入梯度修正(S3w)预注册(2026-09-23)

对象是 `notes/m_star_platform_decomposition_20260923.md` 定位的 D 平台停滞。
本臂只改内环一通内的梯度更新方式;混合长度、EOS(pchip)、不透明度、全局
修正与重映射均不变,生产默认不变,S3 在开关关闭时逐位不变。

## 依据(阶段捕获 `results/m_star_s3_stage_capture_20260922/d/stages.json`)

- 内环沿 ln P_total 用后向差分写入 ∇(`apply_logarithmic_gradient_to_temperature`,
  写入值与设定值差 1e-15);MLT 用 column mass 上的抛物线中心导数读回 ∇
  (`compute_convection` → `differentiate_on_depth_grid`,P_total = g·m 严格
  成立)。深窗 ∇_ad 从 0.153(L67)升到 0.308(L79),读回减写入 ≈ ½Δ∇_ad:
  L67 0.0035/0.0035、L71 0.0058/0.0058、L73 0.0073/0.0074。δ ≈ 0.013 被多读
  45–60%,经 F ∝ δ^1.5 成为 L71–73 的 1.73–1.95 倍对流通量。
- 方向:输入态深窗 F/req = 1.3–2.8(对流通量过剩),第 1 通在 37 个掩码层
  中的 34 层把写入梯度调大,L67–79 全部升温(L71 +8.55 K)。
- 抵消发生在全局修正本身:L72 内环 +9.31 K、修正 −8.61 K;重映射
  Δlog m ≤ 1e-4、ΔT < 1 K。修正在响应被过冲推高的冻结场残差。
- 代理模型 `experiments/reduced_state_emulator/m_star_inner_loop_stencil_surrogate_20260923.py`
  (真实 m、P、T、∇_ad、冻结 H_rad 与读回模板,局部 F ∝ δ^1.5):S3 行复现
  捕获记录(第 8 通 L71–73 F/req 1.726/1.794/1.958 对记录 1.730/1.794/1.952;
  第 1 通 L71 ΔT +8.55 K 对 +8.55 K)。S3w 行 8 通后深窗 max|F/req−1| =
  0.041、均值 0.014,L71 ΔT ≈ −36 K、L79 ≈ −85 K。代理只含模板效应,不含
  EOS/不透明度随温度的响应。

## 改动

`convection_zone_inner_loop_correct_written_gradient`(默认 False)→
`ConvectiveInnerLoopConfig.correct_written_gradient`。每通仍由读回 δ 与
MLT 通量算出目标 ∇_target = ∇_ad + δ_trial;S3 把 ∇_target 直接写成后向
梯度,S3w 写入 `∇_written + (∇_target − ∇_read)`,使读回梯度而非写入梯度
趋向目标。掩码片段的下边缘层(下一层未掩码、且非网格底层)的中心读回跨
过被保持的下邻层,不随本层温度变化,该层保留 S3 的直接写法。每通记录
`max_abs_read_minus_written_gradient`。单元测试
`tests/test_convection_inner_loop_written_gradient.py`。

## 第一阶段:D-plat 单轮(诊断)

起点、求解配置、pchip 包、评价定义与
`notes/m_star_platform_start_points_20260922.md` 相同,只加本开关。本地单
线程,同机重跑一次 S3 作配对基线:

- `results/m_star_inner_loop_written_gradient_20260923/s3_local/d/`(S3)
- `results/m_star_inner_loop_written_gradient_20260923/s3w/d/`(S3w)

各自四态完整重评(iteration_input、inner_pass_01、inner_pass_08、
standard_grid_remap),先过 D 控制锚点 0.8081(±0.001)。

V0(平台核对):本地 S3 的各阶段深窗均值 |R_raw| 与 Garching 捕获一致到
1e-6;不一致时两臂只做同机对比,不与 Garching 数字混用。

预期(诊断,不作通过条件):

- P1:第 1 通 L67–79 全部降温。
- P2:内环退出时冻结场深窗均值 |R_raw| ≤ 0.10(S3 为 0.8589)。
- P3:全局修正在 L67–79 的 |ΔT| 小于同层内环 |ΔT| 的 0.3 倍(S3 为
  8.61/9.31 = 0.92)。

判据 G1(决定是否进入第二阶段):S3w 重映射态的完整物理深窗均值
|R_raw| 低于输入态(0.8128,本地锚点复核后以本地输入态重评值为准)。

## 决策表

| 结果 | 下一步 |
|---|---|
| G1 通过且 P1 成立 | 第二阶段:A 平台单轮同法捕获;再跑 D/A × S3/S3w 四条 10 轮轨迹(`m_star_platform_decomposition_20260923.md` 第 8 步),A it09 深窗均值不得超过 0.1698 |
| P2 成立而 G1 不通过 | 内环已达成冻结场目标,改善仍在修正/重建或完整物理中丢失;用四态重评区分后,"修正前刷新转移"另行预注册 |
| P2 不成立 | 查每通 read−written 缺口与限幅记录;本臂不调通数、不放宽阈值 |
| inner_pass_08 完整物理残差高于输入态 | 冻结场目标(κ_R/τ_R/ρ 冻结)在完整物理下误导;本臂止于第一阶段 |

生产默认、质量门与已注册阈值不因本臂结果改变。

## 第二阶段:D/A 10 轮轨迹(第一阶段后、运行前补充)

第一阶段结果:G1 与 P1–P3 通过,判定进入第二阶段(读数见
`results/m_star_inner_loop_written_gradient_20260923/stage1_summary.json`)。

指标:每轮**输入态**深窗均值 |R_raw| = |H_rad/H + H_conv,raw/H − 1|,
H_rad 取该轮转移、H_conv 取该轮内环之前对输入态的 MLT 评估。S3 轨迹记
录的 `R_raw_deep_mean_abs` 是冻结 H_rad 与内环后 H_conv 的混合量,S3w 下
按构造接近 0,不作判据。由既有记录重算的 S3 参照(与锚点一致:D it09 =
0.8128 = D-plat 输入态重评;A it00 = 0.1698 = 资格阈值):

| 轮 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| D S3 | 0.4349 | 4.1011 | 2.4526 | 0.8386 | 0.7559 | 0.7888 | 0.7800 | 0.7966 | 0.8037 | 0.8128 |
| A S3 | 0.1698 | 0.1478 | 0.3560 | 0.4565 | 0.3596 | 0.4727 | 0.4355 | 0.4725 | 0.4575 | 0.4803 |
| D S0 | 0.4349 | 5.1123 | 17.9788 | 8.1752 | — | — | — | — | — | — |
| A S0 | 0.1698 | 0.2012 | 0.2240 | 0.2443 | 0.2406 | 0.2332 | 0.2079 | 0.1734 | 0.1317 | 0.0960 |

S0 行取 `results/m_star_h2_paired_iteration_20260921/{d,a}_pchip/trajectory.json`(无内环,记录值即输入态指标;D S0 第 4 轮后终止)。

起点与 S3 轨迹相同(`controls_{D,A}/arrays/{d,a}_alpha_0p0.npz`)。S3w 两条
与 A-plat 单轮(捕获 + 重评)在 Garching Node-06 单线程运行,启动器
`experiments/reduced_state_emulator/run_m_star_inner_loop_written_gradient_garching_20260923.sh`;
S3 两条沿用既有 Garching 记录。

判据(不因结果调整):

- D:S3w it09 输入态指标低于 it00 的 0.4349,且各轮均低于 5.112。
- A 资格:S3w A it09 输入态指标 ≤ 0.1698。按同一指标 S3 两条均不满足。
- Hubeny 相消诊断:末五轮对流层内环 ΔT 与全局修正 ΔT 的符号相关,记录不判。
- 相对 S0:A 上 S3w 优于 S3 但 it09 高于 S0 的 0.0960 时,内环(求解顺序)在 A 上仍不优于 S0;D 上 S0 发散,S3w 有界即优于 S0。
