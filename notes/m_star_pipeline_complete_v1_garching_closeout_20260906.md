# M-star pipeline complete v1 — Garching closeout (2026-09-06)

Preregistration: `m_star_pipeline_complete_v1_preregistration_20260905.md`。
目标：scaleout v1 中全部未认证节点跑通。策略仅限两个已实现工具：
`require_improving_flux_residual`（相位感知停止，cap 120–1500）与
continuation 种子路由。wall 合计约 3.5 h（三轮，含两次脚本传输事故重跑）。

## 结果（6/7 完成，1 个按预注册关闭）

| 节点 | v1 状态 | 本轮策略 | 结论 |
| --- | --- | --- | --- |
| dwarf 4000/[M/H]0 | 双腿守卫 False | 相位感知 cap120 | ✓ **22 轮，p95 0.43%，守卫 T/T** |
| dwarf 3800/[M/H]−1.0 | primary 守卫 False | 相位感知 cap120 | ✓ **17 轮，p95 6.92%，守卫 T/T** |
| dwarf 3500/[M/H]0 | MARCS 直跳发散（全 cap） | continuation 种子（已过门 3600 产品）+ 相位感知 cap120 | ✓ **38 轮，p95 7.59%，守卫 T/T** |
| giant 4000/2.5 | restart 120 轮不停 | restart cap 400→1500 | ✓ **primary 18 轮；restart 在数百轮慢弛豫后越线，守卫 T/T** |
| giant 3750/2.5 cap 扫描 | node_id 撞名跳过 | 唯一 id 重跑 | ✓ cap30 与 cap120 完全一致（14 轮，0.61%）|
| dwarf 3600/[M/H]−1.0 | MARCS 直跳发散 | continuation（3800/−1.0 产品，25→6.25 K） | ✗ **关闭**：25/12.5/6.25 K 全部发散 |

## 巨星 4000/2.5 的慢弛豫（新特征，值得记录）

restart 从 rematerialization 瞬态（deep ΔT ≈ 1.0×10⁻³）以极慢速率单调
下降：120 轮 7.4×10⁻⁴ → 400 轮 6.96×10⁻⁴ → 1500 轮内越线。flux 全程
良好（p95 0.5%、max 3.4%）。这是不动点邻域收敛速率接近临界的案例
（对应"ρ(J) 略小于 1 的慢模"），与矮星的振荡/发散是不同物种。

## 认证地图（pipeline v2 最终状态，轨/格点级）

- **巨星 [M/H]0：6/6 certified**（3500–4000 K，logg 1.5/2.5）。
- **矮星 [M/H]+0.5：3/3 certified**（3600–4000 K）。
- **矮星 [M/H]0**：4000 ✓、3500 ✓（continuation）、3400 ✓（continuation）、
  3300 ✓；3800 收敛且 flux 全过、restart 干净，仅 primary 相位守卫
  False——认证政策问题（是否允许 clean-restart 豁免），未擅自放宽；
  3200–3325 走廊关闭。
- **矮星 [M/H]−1.0**：4000 ✓、3800 ✓（相位感知）；3600 关闭（双侧
  6.25 K 均发散）。

## 运行事故记录

- complete_v1 首轮 worker 漏传 `marcs_grid_text`（15 节点全错）；第二轮
  `track_payload` 缺 `microturbulence_km_s`；第三轮 scp 未落地导致旧码
  重跑（3500 假错误）；第四次以 md5 双边核对后成功。三处均为新脚本
  首跑接线问题，与求解器无关；pkill 自杀一次，改用独立会话重启。
- 节点 id 已在本 campaign 内唯一（complete_/capscan_ 前缀）。

## 后续

- 唯一开放的政策问题：clean-restart 是否豁免 primary 相位（涉及
  3800/[M/H]0 与可能的更多节点）。
- 唯一开放的物理问题：断连走廊（[M/H]0 的 3275–3325、[M/H]−1.0 的
  <3800）与 [M/H]−1.0 3600 的发散出身——single-layer parity（原 P3）。
