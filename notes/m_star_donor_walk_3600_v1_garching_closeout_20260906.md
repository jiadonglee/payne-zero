# M-star donor walk 3600 v1 + parity 鉴定 — Garching closeout (2026-09-06)

Preregistration: `m_star_donor_walk_3600_v1_preregistration_20260906.md`。
鉴定：`m_star_parity_divergence_v1`（用户授权的 single-layer parity）。
wall 合计：parity 40 min + donor walk 43 min。

## Parity 鉴定的裁决性发现

两个此前"发散"的下行航点（3800→3775、3800→3787.5）在**直接以
4000/−1.0 已认证产品的 (m,T) 为种子**时全部收敛（production 物理、
cap 60 不变；27/22 轮）。结论：下行 walk 的发散属于**链状态**（反复
rematerialize 的 (m,T) 漂移把种子推到坏盆边缘），不是温度本身不可解。
onset 定位：首轮最大修正出现在表面（log τ ≈ −6，5.5%），属正常瞬态；
深部无早期异常——与"种子落在坏盆"一致，与"该温度无解"不符（在
3775 已被推翻）。

## Donor walk 结果

| 温度 | donor | 轮数 | p95 |
| --- | --- | ---: | ---: |
| 3750 | 3775 产品 | 18 | 6.71% ✓ |
| 3725 | 3750 产品 | 17 | 22.7% ✓ |
| 3700 | **3800 认证产品**（previous_step 失败后换 donor 成功） | 28 | 32.4% ✓ |
| 3675 | 三个 donor（previous/3800/4000）全发散 | 60 | 3.5e4 ✗ |

walk 在 3675 关闭。**墙从 3800 推进到 ~3675–3700**，且 3675 的发散是
donor 无关的——三个相互独立的深盆种子全部发散，排除种子质量解释。

## [M/H]−1.0 矮星轨最终地图（logg 4.5）

- **certified：3800–4000 K**（4000/3800 全门；3750/3725/3700 为
  waypoint 级收敛验证）。
- **3700 以下：关闭**。3675 在三个独立深盆 donor 下发散——结合
  tomography 的分子 EOS 干净结论与 parity 的无深部早期异常，证据指向
  **当前 1D MLT 物理在此温度以下无可达的稳定解**（物理/求解器边界），
  而非认证或种子问题。

## "都得跑通"的最终清算

- scaleout 6 个未认证节点：**5 个 certified**（3800/−1.0、4000/0 相位
  感知；3500/0、3400/0、3300/0 continuation；巨星 4000/2.5 cap1500；
  cap 扫描一致）。
- 3600/−1.0：四条界内路线（直跳、上行、下行、donor 跳）全部穷尽，
  parity 鉴定支持"物理边界"判定。突破它需要改动物理（对流处理、
  不透明度等）——按用户裁决框架，这已超出"跑通"（当前规则集）的
  范畴，是 next-campaign 的物理问题。

## 认证政策落地

- clean-restart 豁免已实现于 `pipeline.certified_solve`
  （`certification_policy: clean_restart_exemption` 留痕）；现有节点
  中无待豁免者（3800/−1.0 已由相位感知直接认证），政策对后续节点生效。
