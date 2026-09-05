# M-star iteration tomography v1 — Garching closeout (2026-09-05)

Preregistration: `m_star_iteration_tomography_v1_preregistration_20260905.md`.
Driver: `experiments/reduced_state_emulator/m_star_iteration_tomography_v1.py`.
Results: Garching `payne-zero-mstar-emulator-v1-20260831/results/m_star_iteration_tomography_v1/`
(pulled to the local tree of the same name; 4 cases, 135 per-iteration NPZs,
wall 587.9 s at 4 workers).

## 结局（gate 口径与插值臂一致，gate 未动）

| case | Teff | [M/H] | primary iters | flux p95 | eligible | 失败原因 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A | 3500 | 0.0 | 38 | 7.584% | ✓ | — |
| B | 3400 | 0.0 | 30 | 9.146% | ✗ | restart_flux_gate（restart p95 17.54%） |
| C | 3300 | 0.0 | 28 | 8.609% | ✓ | — |
| D | 3600 | −0.5 | 25 | 11.560% | ✗ | primary_flux_gate |

与插值臂历史相比：A/C/D 轨迹复现（迭代数 38/28/25 相同，终态温度差
≤0.1 K，D 位级一致）。**B 不复现**：历史 36 轮收敛到 p95 13.98%（双 gate
失败）；本轮 30 轮收敛到 p95 9.15%（primary 过门），失败原因变成 restart
gate。四个 case 的第 1 轮诊断相对历史都有 ~1e-8 相对差（种子重建的跨次
微差；donor v1r2 产品已验证位级相同），只有 B 被这个扰动翻到了另一个
收敛终点（终态温度差 28.5 K）。

## 机制（来自逐轮断层）

1. **温度迭代是真收敛**。|ΔT_raw|/T 在所有深度衰减到判据以下；production
   damping=1.0，不存在被 damping 压出来的假不动点。
2. **分子 EOS 干净**。全部 135 轮 × 80 层：Newton 通过数 ≤2（上限 200），
   lstsq fallback 零次，无深度带。EOS Newton 不是瓶颈。
3. **flux 残差是深部对流区的性质**。末轮逐层 flux error 在 log τ < −2 的
   表面全部 <0.1%，±5–20% 的残差集中在 log τ ≈ +0.5–2.5 的超绝热区
   （∇−∇ad ≈ +0.125），与 gate 的 p95 尾部完全同源。
4. **停止判据与残差长尾松耦合**。对流在第 ~8 轮深部开启后 p95 残差先冲
   高（B 至 ~1300%，A 至 ~400%），再沿慢尾下降；deep-layer ΔT 判据在
   尾部中途触发。A 在第 33 轮时 p95 仍有 ~20%（当轮停就等于失败），
   拖到 38 轮才降到 7.5% 过门。gate 生死取决于 ΔT 越线那一刻残差尾部
   所在位置。
5. **3400 K 是刀尖点**。同一 B 案例内部：primary 端点 p95 9.15%，从它的
   (m,T) 严格自重启 3 轮后的端点 p95 17.54%，而两者 path consistency
   通过（温度差 <0.3%）。0.3% 的温度差让对流区残差近乎翻倍。外加
   1e-8 种子扰动翻转历史/本轮的主端点——3400 K 附近存在多个可达的
   形式收敛态，残差水平是路径函数。3300/3500 K 则对同样的扰动稳健。
6. 深部修正翻号（chatter）在所有 case 的对流区都存在（末 10 轮 0–25%
   翻号率），但它不阻止 ΔT 收敛——对流拓扑翻转是背景噪声，不是主死因。

## 判读

- 3400 K "hole" 不是分子 EOS 问题，不是初值网络问题，也不是 ΔT 被压制
  出来的假收敛；它是**深部对流区残差慢尾 + 停止判据赛跑 + 刀尖点多解**
  三者叠加的 solver-globalization 缺陷。3500/3300 的"成功"同样带着
  ~7.5–8.6% 的对流区残差，只是恰好落在门内。
- D（3600 K、[M/H] −0.5）的 primary 残差 11.56% 在位级复现意义下稳定
  出现（迭代数、终态与历史全同）——它的失败不是刀尖，是当前 1D MLT
  物理在该格点收敛解的稳定属性，归 physics/gate 口径问题。
- 因此下一轮的最小改进应打在 **residual-aware 的 solver policy** 上：
  把 flux 残差纳入停止/认证判据（或对 ΔT 判据加 residual 伴随条件）、
  对流区步长全球化（backtracking/trust region），以及以 continuation
  路径固定进入哪个收敛盆。S1–S3 arms 的设计以此为纲。

## 平台与身份

- 种子以 Garching 数组内容 sha256 为准（A/B/C
  `d0c22889…6dcca`，D `534233ce…c20c`；preregistration 已冻结）。
  种子重建跨 BLAS 后端不位级一致（macOS Accelerate vs Linux），Garching
  为 campaign 正本；donor 产品两边位级相同。
- 执行节点 astronode-garching（Node-06），单迭代 ~13.4 s（历史机器
  ~124 s/iter 的 ~9 倍）。

## 已知遗留

- continuation probe 首次真实运行即崩溃：`TRACK_A/TRACK_B` 载荷缺
  `class`/`role` 键，`_annotate_record` 读 `track_payload["class"]` 抛
  KeyError。已修复（载荷补齐两键，与插值臂结构一致）；修复后 probe
  复跑时又在后处理暴露第二处同类路径缺陷（`_sha256` 收到 str），同样
  已修复。两处均为该脚本首次真实执行暴露的既有缺陷，与 tomography 无关。

## Continuation probe 结果（同日，紧接 tomography）

探针按预注册执行（A：3500→3400；B：3700→3600，50 K 失败减半 25 K）：

| probe | target | 路径 | eligible | 对照（插值臂直跳） |
| --- | --- | --- | --- | --- |
| A | 3400 K, [M/H] 0 | 50 K 一步；primary 20 轮 p95 6.98%，restart 3.85% | ✓ | 36 轮 p95 13.98%，失败 |
| B | 3600 K, [M/H] −0.5 | 3700→3650 过；3650→3600 失败；25 K 重试最后一步发散（60 轮 p95 4.6e5%），minimum_step_failed | ✗ | 25 轮 p95 11.56%，形式收敛但超门 |

- **3400 K 被 continuation 救活且质量更好**：同一种子家族，直跳落在
  13.98%（历史）/9.15%（tomography 重跑）的刀尖盆，从 3500 K 走 50 K
  稳定落在 6.98% 的好盆，restart 3.85% 是该格点迄今最好值。路径决定
  归宿，与 tomography 的多解判读互为印证。
- **3600 K [M/H] −0.5 不是路径问题**：continuation 在最后 25 K 直接
  发散，直跳形式收敛但残差稳定在 11.56%（tomography case D 位级复现
  同值）。该格点收敛解的对流区残差本身就压不过暖星定标的 gate，
  归 physics/gate 口径。
- 预注册决策规则（1/2 eligible → 只走通过轨）触发：轨 A 打开，链种子
  为新过门的 `…_t3400`（primary p95 6.98%），walk 目标 3200/3100/3000 K；
  轨 B 关闭。

## Walk 结果（轨 A）

- `3400→3200`（50 K）：24 轮形式收敛但 p95 16.43%，gate 失败；按规则
  减半 25 K 重试，3400→3225 一步 60 轮发散（p95 4.98×10⁵%），随后
  再试同样发散（1.22×10⁴%）。`minimum_step_failed`，3200 K 格关闭，
  轨 A 停止（WALK_COMPLETE）。
- 至此富金属轨的地图：3500/3400/3300 过门（3400 经 continuation），
  3200 关闭。3200 的失败形态与 3600 mp 的最后一步相同——**发散**，
  与 3600 mp 直跳的"稳定收敛但残差超门"不同。更冷端（≤3200 K）的
  first障碍是迭代本身的失稳，不是残差地板；其对应的最小改进仍是
  步长全球化与残差伴随判据（residual-aware stopping），而非更多物理。
