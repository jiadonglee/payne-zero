# ESO M 星比较 v3：共享 RV 和旋转展宽

## 结论

对主目标 `IC2391-0096`，在 Ca I 6155–6170 Å 中让 Payne-Zero 与 MARCS 等权共同决定一套参数，得到：

- 相对目录 RV 的共同偏移：`-0.6918 km/s`；总 RV 为 `11.9082 km/s`。
- 共同 Korg 旋转展宽：`v sin i = 7.1125 km/s`。
- 仪器分辨率固定为 `R=80,000`。两个参数都没有撞到搜索边界。

展宽修正只部分成功，不能说 Ca I 线宽已匹配。用同一个可复现的 6162.173 Å 半深宽定义，观测 FWHM 是 `17.17 km/s`，拟合后 Payne-Zero 是 `30.23 km/s`，MARCS 是 `31.11 km/s`。这与上一轮“模型约 31.7 km/s、观测约 15.5 km/s”的直观结论一致：具体数值会随线宽定义略变，但模型仍明显太宽。共同 RV 则基本消除了线心偏差；观测与两模型的线心只差约 `+0.144 km/s`。

冻结这套共同 RV/展宽后，两个窗口的结果是：

| target | window | PZ RMS | MARCS RMS | ΔRMS (PZ−MARCS) |
|---|---|---:|---:|---:|
| IC2391-0096 | Ca I 6155–6170 Å | 0.082765 | 0.083294 | -0.000529 |
| IC2391-0096 | TiO 6650–6670 Å | 0.043975 | 0.046158 | -0.002183 |
| HD219215 | Ca I 6155–6170 Å | 0.096155 | 0.096299 | -0.000144 |
| HD219215 | TiO 6650–6670 Å | 0.069757 | 0.068718 | +0.001039 |

这里 `ΔRMS = RMS(obs−PZ) − RMS(obs−MARCS)`；负值表示 PZ 数值更低。主目标在 Ca I 中差异很小，基本打平；TiO 窗口 PZ 的 RMS 低 `0.00218`，约为 MARCS RMS 的 `4.7%`。这是一个 M dwarf、一个冻结 nuisance 的局部结果，不能外推成 Payne-Zero 在整个 M-star 网格胜出。

PI 复查把 TiO 窗口分成四个 5 Å 小段，`ΔRMS` 依次为 `-0.001531`、`-0.002738`、`-0.000985`、`-0.003092`。四段都局部偏向 PZ，因此这个差异不是由单个窄波段或一条异常谱线独占；它仍然只是同一颗星、同一段 TiO 光谱的证据。

`HD219215` 的共同参数为 `ΔRV=+0.1775 km/s`、`v sin i=9.3438 km/s`，但它相对采用节点仍有 `Δlogg=+0.674 dex`、`Δalpha=-0.529 dex`。它只保留为标签不匹配诊断，不参与 atmosphere 排名。

## 做法

重新从未施加旋转展宽、未施加 LSF 的 0.01 Å Korg 合成谱开始。Ca I 使用与 v1 相同的 VALD solar atomic-only 线表；TiO 6650–6670 Å 使用完整的
`/Users/jdli/Project/jorg/Korg.jl-1.0.1/data/linelists/GALAH_DR3/galah_dr3_linelist.h5`。

Korg 先用 `apply_rotation` 生成 `v sin i=0–40 km/s`、步长 `0.25 km/s` 的网格，再用 `apply_LSF` 固定到 `R=80,000`。Python 在 Ca I 窗口最小化两个 atmosphere 的平均残差平方；两套 atmosphere 各自只允许同阶的 `model × (a+b x)` 连续谱 nuisance。得到共同 RV/展宽后不再调整，直接评估 Ca I 与 TiO。

没有做 no-TiO 消融，因为 v2 已经直接验证该窗口的主要分子敏感性来自 TiO；重复消融不会改变本轮“共享运动学 nuisance 是否解释残差”的判断。

Payne-Zero 的内部 solver 成功率没有进入任何目标函数、评分或 atmosphere 判断，也不能作为 MARCS 更稳或更好的证据。

## 数据直接显示与局限

数据直接显示：降低共同旋转展宽并修正约 `-0.69 km/s` 的 RV 后，主目标两个模型的整体 RMS 都下降，线心基本对齐；但 Ca I 6162 Å 的模型线仍约比观测宽一倍。剩余问题不是简单的“目录 RV 或 v sin i 用错”就能解释，可能还混有强线压力展宽、原子数据、标签或 atmosphere structure 差异。

现在不能说明：哪套 atmosphere 对所有 M dwarf 更好，也不能把 TiO 单窗口的 PZ 小优势解释成整体光学谱、其它温度/重力节点或 M giant 上的优势。下一步最小检查应针对 Ca I 6162 Å 的内禀/压力展宽，而不是继续单独调 RV 或旋转核。

## 产物

- `results/m_star_eso_highres_comparison_v3_shared_nuisance/metrics.json`
- `results/m_star_eso_highres_comparison_v3_shared_nuisance/metrics.csv`
- `results/m_star_eso_highres_comparison_v3_shared_nuisance/figures/IC2391-0096_shared_nuisance_observed_models_residuals.png`
- `results/m_star_eso_highres_comparison_v3_shared_nuisance/processed/`
- `results/m_star_eso_highres_comparison_v3_shared_nuisance/models/spectra/`
- `experiments/reduced_state_emulator/compare_mstar_eso_highres_v3_shared_nuisance.py`

脚本语法、JSON/NPZ/HDF5 可读性和最终 M-dwarf 图均已检查。图中的观测、两模型、残差方向和共享参数标注正确。
