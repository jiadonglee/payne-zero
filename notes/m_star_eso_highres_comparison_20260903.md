# ESO 高分辨率 M 星：Payne-Zero 与 MARCS 最小实测比较

## 结论

这轮**不能判定 Payne-Zero 或 MARCS 谁总体更好**。

在标签最接近的 M dwarf `IC2391-0096` 上，两套模型在 Mg b 窗口几乎打平，在 Ca I 6162 窗口 Payne-Zero 的 RMS 小 `0.00043`；这个差别只有观测—模型 RMS（`0.091–0.166`）的约千分之几，而且两个窗口的数值胜者相反。M giant `HD219215` 虽然是非已知变星里离可用节点最近的对象，但采用节点仍有 `Δlogg=+0.674 dex`、`Δalpha=-0.529 dex`，只能算 label-mismatch diagnostic，不能用来判 atmosphere 胜负。

共同合成线表没有 TiO。图中大量观测细结构没有被两套模型复现，说明当前残差主要检验的是“atmosphere + 不完整线表 + 标签/丰度 + 恒星活动等共同前向模型”，不是纯 atmosphere。这里不做宽波段或分子带排名。

Payne-Zero v1r2 的 `31/108` 只描述该求解流程自身的成功数，**没有进入这次光谱评分，也不是 MARCS 更稳定的证据**。同理，Payne-Zero 自身的 flux residual 只能评价该 Payne-Zero 解，不能自动给 MARCS 加分。

## 目标与节点

参数和误差来自 Borisov et al. (2023) 的统一 UVES-POP 表。距离定义为
`sqrt[(ΔTeff/100 K)^2 + (Δlogg/0.5)^2 + (Δ[M/H]/0.5)^2 + (Δalpha/0.2)^2]`，只在 v1r2 `training_eligible=true` 的 Payne-Zero 节点中搜索。

| target | 类型 | 文献标签 `Teff/logg/[Fe/H]/alpha` | 共同采用节点 | 节点减文献标签 | 角色 |
|---|---|---|---|---|---|
| IC2391-0096 | M0–5? dwarf | `3728±37 / 4.9498±0.030 / +0.0391±0.030 / +0.0995±0.030` | `3750 / 5.0 / 0.0 / 0.0` | `+22 K / +0.050 / -0.039 / -0.100` | 主近邻诊断，距离 `0.559` |
| HD219215 | M1.5III | `3871±39 / 0.826±0.176 / -0.097±0.110 / +0.529±0.084` | `3900 / 1.5 / 0.0 / 0.0` | `+29 K / +0.674 / +0.097 / -0.529` | 标签错配诊断，距离 `2.989` |

HD219215 数学上更近的 `3900/logg=0.5/[M/H]=0/alpha=0` case 在 v1r2 中是 `ineligible`（primary/restart flux gate 与 path consistency 失败），所以没有用于合成。其它候选和排除原因见 [target_candidates.csv](../results/m_star_eso_highres_comparison_v1/target_candidates.csv)。其中 HD102212 是 SRB 变星；HD092305 在统一表中为 K4III；HD156274 为 G9V，不再按旧表的 M0V 分类使用。

## 数据与公平处理

两颗星都来自 ESO program `266.D-5655(A)` 的 UVES-POP 重处理 R80k 产品：

| target | RA, Dec (deg) | ESO OB ID | instrument/R | 范围与字段 | 公开产品 |
|---|---|---:|---|---|---|
| IC2391-0096 | `130.864945, -52.961260` | `200105380` | VLT/UVES, `R=80,000` | `3200–10249.98 Å`; FLUX, ERROR, PIXMASK, QUALITY | [R80k FITS](https://data.voxastro.org/uves-pop/model_spec/merged_221115/IC2391-0096_R80k.fits.gz) |
| HD219215 | `348.579870, -6.049340` | `200108136` | VLT/UVES, `R=80,000` | `3200–10249.98 Å`; FLUX, ERROR, PIXMASK, QUALITY | [R80k FITS](https://data.voxastro.org/uves-pop/model_spec/merged_221115/HD219215_R80k.fits.gz) |

FITS primary header 给出 `CTYPE1=AWAV`、`BUNIT=erg/cm^2/s/ang`、`FLUX_CAL=Calibrated`；表列本身的 unit metadata 是空的，因此单位采用 primary header 与产品文档，不补造列单位。使用 `PIXMASK=1`、`QUALITY=0`、有限正 flux/error，并排除 ESO 已知坏列和强 telluric 区。两个所选窗口不落在这些区间内。

观测波长用 Borisov 表的固定 RV（`12.60±0.38` 和 `1.90±1.02 km/s`）移到恒星静止系；没有为任一 atmosphere 单独调 RV。Korg 1.0.1 对两套 atmosphere 使用同一节点标签、丰度、atomic-only VALD solar line list（36,157 条）、hydrogen lines、平面几何、microturbulence、`v sin i`、`R=80,000` 和 air-wavelength 输出。模型都插值到同一批观测像素。

每个窗口把观测除以局部 95% 分位 flux；每个假设都只允许同样两个自由度的乘法直线连续谱 `model × (a+b x)`。评分采用完全相同的像素和等权重。ERROR 列存在，但产品相关噪声、连续谱与线表系统误差没有协方差模型，所以不报 reduced chi-square，只报 RMS、标准 MAD 和 `p95(|residual|)`。

## 观测—模型残差

| target | window | PZ RMS / MAD / p95 | MARCS RMS / MAD / p95 | `RMS_PZ-RMS_MARCS` | 调连续谱后两模型 RMS 距离 |
|---|---|---|---|---:|---:|
| IC2391-0096 | Mg b 5160–5190 Å | `0.166204 / 0.126591 / 0.297355` | `0.166180 / 0.126473 / 0.297646` | `+0.000024` | `0.001174` |
| IC2391-0096 | Ca I 6155–6170 Å | `0.091323 / 0.064440 / 0.166266` | `0.091753 / 0.064183 / 0.167847` | `-0.000430` | `0.001958` |
| HD219215 | Mg b 5160–5190 Å | `0.163073 / 0.116215 / 0.313938` | `0.163170 / 0.116023 / 0.311211` | `-0.000097` | `0.003269` |
| HD219215 | Ca I 6155–6170 Å | `0.096705 / 0.062308 / 0.184951` | `0.096884 / 0.062052 / 0.186785` | `-0.000179` | `0.003453` |

“数值更小”在不同窗口和不同指标间并不一致；差值也远小于共同的观测—模型残差。因此每颗星的科学判断都是 `cannot_determine`，而不是把微小数值差当作 atmosphere 排名。

- [IC2391-0096 观测、两模型与残差图](../results/m_star_eso_highres_comparison_v1/figures/IC2391-0096_observed_model_residuals.png)
- [HD219215 观测、两模型与残差图](../results/m_star_eso_highres_comparison_v1/figures/HD219215_observed_model_residuals.png)

两张图已目视检查：波长对齐和残差方向正确，PZ 与 MARCS 曲线几乎重合；明显较大的共同残差真实存在，不是画图尺度造成的。

## Payne-Zero 节点自身物理残差

| target/node | flux error median / p95 / max | temperature path p95 | column-mass path p95 |
|---|---|---:|---:|
| IC2391-0096 邻近节点 | `0.0494% / 1.6714% / 10.5142%` | `2.231e-4` relative | `0.001580 dex` |
| HD219215 采用节点 | `0.00783% / 0.80105% / 2.80239%` | `4.756e-4` relative | `0.000756 dex` |

这两套 PZ case 都是 v1r2 `training_eligible`。MARCS library 没有同定义、同流程的 solver/path residual，因此这张表只说明所用 PZ 节点自身状态，不能拿来比较两套 atmosphere 的稳定性。

## 来源与可复现产物

- 数据论文：[Borisov et al. 2023, arXiv:2211.09130](https://arxiv.org/abs/2211.09130)，DOI `10.3847/1538-4365/acc321`；早期重标定说明：[arXiv:1802.03570](https://arxiv.org/abs/1802.03570)。
- 实际论文查询数据库：arXiv export API；endpoint 为 `https://export.arxiv.org/api/query?search_query=all:%22UVES-POP%22&start=0&max_results=20&sortBy=relevance&sortOrder=descending`。原始 Atom 响应保存在 [arxiv_uves_pop_atom.xml](../results/m_star_eso_highres_comparison_v1/sources/arxiv_uves_pop_atom.xml)。
- 目标标签：[CDS J/ApJS/266/11](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/266/11)；ESO 原始库说明：[UVES-POP](https://www.eso.org/sci/observing/tools/uvespop.html)。
- 机器可读结果：[metrics.csv](../results/m_star_eso_highres_comparison_v1/metrics.csv)、[metrics.json](../results/m_star_eso_highres_comparison_v1/metrics.json)、[FITS inventory](../results/m_star_eso_highres_comparison_v1/fits_inventory.json)、[source manifest](../results/m_star_eso_highres_comparison_v1/source_manifest.json)。
- 处理脚本：[compare_mstar_eso_highres_v1.py](../experiments/reduced_state_emulator/compare_mstar_eso_highres_v1.py)；Korg 合成脚本：[korg_synthesize_atomic_windows.jl](../results/m_star_eso_highres_comparison_v1/models/korg_synthesize_atomic_windows.jl)。

最小下一步是先补齐并验证 TiO/其它关键分子线表，再在相同标签与相同 nuisance 下重做；巨星还需要 alpha 和 logg 都更接近可用 PZ 节点的非变星，或先得到与 HD219215 标签匹配的 PZ atmosphere。没有这两项，扩大波段或样本不会把当前结果变成公平的总体判决。
