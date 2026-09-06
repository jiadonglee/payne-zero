# M-star pipeline scaleout v1 — preregistration (2026-09-05)

3300–3500 K 稳定区间确立后，把已认证的 pipeline（MARCS/continuation
种子 → production 求解 → 严格自重启 → 冻结 flux gate → path
consistency → 认证相位守卫）收拢为单一入口
`m_star_pipeline.certified_solve`，并在三个维度铺开测试。求解器物理、
冻结 gate、认证判式全程不变；相位守卫双腿必须 True。

## 臂

- **G 巨星**：logg {1.5, 2.5} × Teff {3500, 3750, 4000}，[M/H] 0，
  vmic 2.0，MARCS 同节点种子，cap 60，共 6 点。v1r2 同区高通过率
  （53/80），本轮检验相位守卫时代认证是否保持。
- **M 金属丰度矮星**：logg 4.5 × Teff {3600, 3800, 4000} ×
  [M/H] {−1.0, +0.5}，vmic 1.0，MARCS 种子，cap 60，共 6 点。map
  稳定解随金属丰度的边界（−1.0 是 v1r2 3400 K 失败轨的延伸方向）。
- **I 迭代次数**：3 个锚点（巨星 3750/2.5/0、矮星 4000/4.5/0、
  矮星 3500/4.5/0）× cap {30, 60, 120}。回答：收敛所需迭代数是多少、
  认证结论是否随 cap 独立（预期：cap ≥ 收敛轮数时结论不变；cap 不足
  时按未收敛处理）。

## 判读

1. 各臂逐点报告 eligible、迭代数、三指标、双腿守卫。
2. G：与 v1r2 同区结果对比，回退即记录。
3. M：标出各金属丰度下最冷的 eligible Teff。
4. I：收敛迭代数的节点间差异；cap 30/60/120 下 eligible 的一致性。
5. 探索性目标，无采纳/拒绝动作；结果决定下一批节点的选取。
