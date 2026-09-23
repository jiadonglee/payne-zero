# 内环试探温度状态重算(S3wr)预注册(2026-09-23)

对象是 `notes/m_star_inner_loop_written_gradient_closeout_20260923.md` 记录的
冻结回调与完整物理的差距。本臂在 S3w 上只加一项:内环物理回调在试探温度
下重算压力迭代态。混合长度、EOS(pchip)、不透明度、全局修正与重映射不变,
生产默认不变,开关关闭时 S3w 逐位不变。

## 依据

- 内环第 8 通:D-plat 冻结场 0.0137 对完整物理 0.5798,A-plat 0.0227 对
  0.6099(深窗均值 |R_raw|);差距全部在 H_conv,H_rad 两者一致。
- `compute_convection_finite_difference_samples` 采样后把 `runtime_state`
  (质量密度、电子密度、离子与分子布居)恢复为调用时的值;内环中这些值始终
  是本轮输入态的。MLT 的中心密度、∇_ad 与 c_P 的归一化因此用输入温度下的
  状态,而 ±扰动样本已在试探温度附近求解。
- A-plat 最深层(L72–79)降温后完整物理 H_conv 上升(L75 R 0.448 → 0.845),
  回调给出的方向相反。

## 改动

`convection_zone_inner_loop_refresh_state`(默认 False)。回调在有限差分采
样之前,于试探温度重解一次压力迭代态(`populate_species(code=0,
pressure_iteration_enabled=True)`;分子平衡的 kT 在
`molecular_convection_thermal_tracks_perturbation` 为真时跟随试探温度),
与采样器在每个扰动温度上的调用相同。气体压力保持,κ_R、τ_R、H_rad 仍为本
轮输入态。内环结束后的对流重算与全局修正使用内环末温度下的重算状态。
S3wr = S3w + 本开关。

## 第一阶段:D-plat 与 A-plat 单轮

起点与 S3w 相同(`d_pchip_s3_it09`、`a_pchip_s3_it09`),阶段捕获 + 完整
物理重评(D:iteration_input、inner_pass_01、inner_pass_08、
standard_grid_remap,先过 D 控制锚点 0.8081;A:iteration_input、
inner_pass_08、standard_grid_remap)。Garching Node-05 单线程(与 S3/S3w 的
Node-06 共用 /nexus 目录;跨机数值差为 1e-7 量级,见 S3w V0),产物在
`results/m_star_inner_loop_state_refresh_20260923/`。

C0(实现核对,不是臂结果):inner_pass_00(输入温度上的重算)回调 H_conv 与
不重算时一致,深窗逐层相对差 ≤ 1e-3。D 对照 Garching S3 捕获
`results/m_star_s3_stage_capture_20260922/d`,A 对照 S3w A-plat 捕获。

差距量:gap = inner_pass_08 完整物理 − 冻结场(深窗均值 |R_raw|);S3w 为
D 0.5661、A 0.5872。

判据(不因结果调整):

- G1:D-plat 重映射态完整物理 < S3w 的 0.6167。
- G2:A-plat 重映射态完整物理 < 输入态 0.4803(S3w 为 0.6227)。

## 第二阶段:D/A 10 轮轨迹(与第一阶段同时运行)

起点、输入态指标与 S3w 第二阶段相同;两阶段同时运行以节省墙钟,第二阶段
只在 G1 与 G2 都通过时作为本臂结果判定,否则作为诊断记录。

- A 资格:it09 ≤ 0.1698;与 S3w 的 0.0038 并列报告。
- D:it09 < 0.4349 且各轮 < 5.112。D 起点的外迭代不稳定在 S0 与 S3w 下都出
  现,预期本臂不消除它。

## 决策表

| 结果 | 下一步 |
|---|---|
| C0 不成立 | 实现错误;修正后重跑,不下臂结论 |
| G1 与 G2 通过 | 状态重算并入内环候选;下一步单独诊断 D 起点的全局修正稳定性(另行预注册) |
| G1 或 G2 不通过,但两态 gap 都 ≤ S3w 的一半 | 剩余差距归于 κ_R/τ_R/H_rad 冻结;下一臂在试探温度更新 κ_R(另行预注册) |
| 两态 gap 未减半 | 状态冻结不是主因;复查差距来源,不调参数 |

生产默认、质量门与已注册阈值不因本臂结果改变。
