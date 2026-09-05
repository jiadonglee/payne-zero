# M-star iteration tomography v1 — preregistration (2026-09-05)

## 问题

插值臂在矮星轨上留下了一个锋利的模式：富金属轨（logg 4.5、[M/H] 0）上
3500 K 全门通过、3400 K 形式收敛但 flux gate 失败、3300 K 再次通过；
贫金属轨（logg 4.5、[M/H] −0.5）上 3700 K 通过、3600 K primary flux gate
失败（restart 反而通过）。全部四个 case 的求解器都达到了形式收敛
（deep-layer ΔT/T < 5e-4），死活取决于 |flux error| 的 p95（冻结阈值 9.557%）：

| case | Teff | logg | [M/H] | 插值臂结局 | primary iters | flux p95 |
| --- | ---: | ---: | ----: | --- | ---: | ---: |
| A | 3500 | 4.5 | 0.0 | 全门通过 | 38 | 7.56% |
| B | 3400 | 4.5 | 0.0 | primary+restart flux gate 失败 | 36 | 13.98% |
| C | 3300 | 4.5 | 0.0 | 全门通过 | 28 | 8.61% |
| D | 3600 | 4.5 | −0.5 | primary flux gate 失败 | 25 | 11.56% |

要区分两个假设：

- **H1（solver 假象）**：3400 K 的 flux 残差来自迭代轨迹——修正量振荡、
  被 damping 压进假不动点（ΔT_applied → 0 而 ΔT_raw 不小）、或对流拓扑
  逐轮翻转拖住残差。
- **H2（物理地板）**：给定当前 1D LTE 物理（分子 EOS、TiO/H2O/atomic
  blanketing、MLT），3400 K 的收敛解就是带着 ~14% 的 p95 flux 残差，
  gate 是按暖星面板冻结的，cool dwarf 达不到。

本实验不裁决 H1 vs H2，只把每一轮迭代的完整状态记录下来，让判读有据。

## 方法

- 驱动：`experiments/reduced_state_emulator/m_star_iteration_tomography_v1.py`
- 四个 case 用与插值臂完全相同的种子构造（同 donor、同
  `interpolate_same_track_mt`、同 `reconstruct_full_atmosphere` 同步参数）
  和完全相同的求解设置（cap 60、strict all-layer 5e-4、production gates）。
  唯一区别是挂了 `after_iteration_hook`，只观测、不干预。
- donor（v1r2 过门产品，单侧最近邻）：
  - A/B/C ← `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3600`
  - D ← `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3900`
- 种子身份（`protocol` 阶段冻结，sha256 见 `results/<campaign>/protocol.json`；
  Garching 复算的种子 sha256 必须与此一致）：

| case | seed sha256 |
| --- | --- |
| A `…_t3500` | `09d34ef2d694249473753ee9826a9179d8837cde74d62ba57889d3c159fb0f24` |
| B `…_t3400` | `09d34ef2d694249473753ee9826a9179d8837cde74d62ba57889d3c159fb0f24` |
| C `…_t3300` | `09d34ef2d694249473753ee9826a9179d8837cde74d62ba57889d3c159fb0f24` |
| D `…_t3600` | `e4f83d1c65487406320b08a9881350f3991aed404b0d850600fcd68304728e0c` |

A/B/C 种子 (m,T) 完全相同（同一 donor 单侧拷贝），差别只在目标 labels
（Teff 进入 flux 目标与谱窗）；sha256 相同是构造的直接结果，不是错误。
protocol hash：`75972bbe479cdb3f9b144b3d0542706cd065c16db611c606c3b43e597d4a2369`。

- 过门的 case 追加 strict self-restart 腿（与插值臂同一定义），并做
  primary/restart path consistency、与插值臂历史产品的温度 parity
  （诊断改动不得扰动默认路径的对照）。

## 记录量（每轮迭代一个 NPZ，primary 与 restart 分目录）

- `raw_temperature_correction`（启发式与 damping 之前）、
  `applied_temperature_correction`、`temperature_pre/post`
- `flux_error_percent`（逐层）、`flux_ratio`（F_conv/F_tot）
- `superadiabatic_gradient`（∇−∇ad）
- `rosseland_opacity_post`、`electron_density_post`、`convective_flux_post`、
  `column_mass_post`、两套 log τ 轴
- `molecular_newton_iterations`、`molecular_newton_used_lstsq`（逐层）
- timing 标量（含 `maximum_abs_raw_relative_temperature_correction`、
  deep/all-layer ΔT、flux 统计）

## 判读协议

1. **signflip fraction 与 |ΔT_raw|/T 的深度-迭代热图**：若 B 在特定深度带
   持续振荡且 ΔT_raw 不衰减 → H1 的 damped-false-convergence 证据；
   若 ΔT_raw 也平滑衰减到零 → 迭代本身健康，残差是物理的。
2. **逐层 flux_error_percent 的迭代演化**：残差结构集中在分子敏感深度带
   （H2O/TiO rich）→ H2 倾向；残差随迭代在同号递减但停在 14% → H2 倾向。
3. **∇−∇ad 的层间翻转与 flux_ratio 跳变**：对流拓扑逐轮翻转与 ΔT 翻号
   同步出现 → 对流切换是失稳源（为 frozen-topology 实验提供依据）。
4. **molecular Newton 计数的深度带**：计数在特定深度带持续顶格（200）或
   频繁 lstsq → EOS 数值问题优先。
5. **parity 对照**：与插值臂历史产品温度最大差应为 0（同一台机器同一线程
   配置下位级一致）；非零时先查环境再谈物理。

## 边界

- 不改任何 physics、任何 gate、任何收敛判据；hook 只读。
- 本实验不做 S1–S3 solver arms、不做 ρ(J) 估计；依据其结果另行预注册。
- B/D 预期仍不合格——它们是数据点，不是失败。
