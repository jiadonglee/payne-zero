# D 平台定位实验:固定起点与评价定义(2026-09-22)

本轮目标:拆开 D 的 S3 平台上的一轮更新,回答"局部改好了又被弄坏"还是
"局部根本没改好";并行做 71–73 层 EOS 可信度小对照。三个待答问题:内环
每次更新是否降低自身失配;全局修正/状态重建是否撤销改善;热力学导数的不
确定性是否足以改变上述判断。按约定,本阶段不启动 Garching 重算;本文件固
定起点、评价定义与文件归属,作为后续所有配对实验的基准。

## 固定起点

**主起点 D-plat(D 的 S3 平台态)**
- 状态文件:`results/m_star_h2_inner_loop_s3_20260922/d_pchip_s3/arrays/d_pchip_s3_it09.npz`
  的输入列(temperature、column_mass;该轮无全局修正被接受,输出即下轮
  输入,链条核查通过)。
- 缺失字段:该系列 npz 未存 gas_pressure/total_pressure,使用前经
  `_state_from_mt`(同一 (m,T) 重建,全部臂的既有约定)补齐并另存完整
  起点文件,不得改用其他重建。
- 热力学:pchip 包副本(`build_overlay('pchip', root)`),求解配置 =
  `_solver_config` + `dataclasses.replace(enable_opacity_lagging=False,
  opacity_recompute_interval=1, flux_residual_guided_damping=False,
  require_improving_flux_residual=False, convection_zone_inner_loop_passes=8,
  convection_zone_inner_loop_freeze_mask=True)`。
- 平台参考值:深窗(67–79)均值 |R_raw| = 0.8589,深窗最大 |R_smoothed| =
  1.2347,末轮 dTmax = 1.04 K;深窗对流全开。

**控制起点 A-ori(A 的原始起点)**
- 状态文件:`results/m_star_trial_comparison_20260920/controls_A/arrays/
  a_alpha_0p0.npz`(temperature、column_mass),同样经 `_state_from_mt`。
- 对照轨迹:`results/m_star_h2_paired_iteration_20260921/a_pchip/
  trajectory.json`(S0 式,pchip,10 轮收敛到 0.0960)与
  `results/m_star_h2_inner_loop_s3_20260922/a_pchip_s3/trajectory.json`
  (S3,平在 0.422)。
- 资格判据沿用:任何候选修复不得把 A it09 深窗均值推过 0.1698(其 S0
  起点)。

轨迹完整状态:A/D 的 S3 轨迹本地均已补齐为 10 轮(A 曾缺 3–9 轮,已从
远端重拉,未重算)。机器可读索引见
`results/m_star_trajectory_index_20260922/`(由
`experiments/build_m_star_trajectory_index_20260922.py` 生成)。

## 评价定义(全程统一)

- 主指标:深窗 67–79 均值 |R_raw|(等权);辅助:全层 max |R_raw|、深窗
  max |R_smoothed|。
- 残差语义:轨迹第 k 轮指标属于第 k 轮输入状态(控制锚点 0.2105/0.8081
  逐位校验);任何阶段记录必须标注辐射场来源(frozen=内环冻结场,
  current=完整转移重算),两者不得混画为一条曲线。
- "只评价状态"入口:单轮 iterations_per_trial=1、opacity lagging 关、
  返回输入状态残差、不接收修正;验证步骤=复现 D 控制态 pchip 基线
  0.8081。

## 文件归属(避免并行冲突)

- 状态传递(Luna):`experiments/reduced_state_emulator/m_star_s3_stage_capture_20260922.py`(新建,唯一允许包含阶段捕获与 evaluate_state_only 的文件;不改 payne_zero_atmosphere/)。
- 热力学(Luna):`experiments/analyze_h2_scan_threshold_20260922.py`、
  `experiments/reduced_state_emulator/m_star_h2_platform_scan_20260922.py`
  (新建;再分析现在做,目标扫描待批)。
- 结果分析(Luna):`experiments/build_m_star_trajectory_index_20260922.py`(新建)。
- 主代理:起点打包、远端提交、阶段判定;不并行改上述文件。

## 计算预算与执行顺序(待启动)

M-dwarf 首轮只用一个单线程重算进程,按依赖顺序:
1. 阶段捕获一轮(D-plat 起点,~3 min)→ 阶段图,定位失配在哪一步回升。
2. 四个完整重评(D:内环前/第 1 通后/第 8 通后/重建后;~8 min)→ 局部
   失配 vs 真实残差对照,A 再补三个评价。
3. 平台态 71–73 层目标扫描(三方案 × 两步长,~20 min)。
每阶段落盘;失败记录具体状态,不自动重试。按结果进入第四/五步(梯度实现
检查或最小配对对照),每步只开一个分支。

## 本阶段产物

- 本文件(起点说明与定义)。
- 轨迹索引与语义核查:`results/m_star_trajectory_index_20260922/`。
- 阶段捕获脚本 + 只评价入口(待远端执行)。
- 71–73 层再分析结论与目标扫描脚本(待远端执行)。
