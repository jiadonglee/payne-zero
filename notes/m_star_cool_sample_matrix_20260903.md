# Cool-star 观测覆盖与三套合成谱

## 结论

此前下载的数据不能填满目标观测矩阵。12 个格子里只有 3 个能放入已下载的 M 星作诊断：`IC2391-0096`、`HD219215`、`HD102212`；其中 `HD102212` 是 SRB 变星，`HD219215` 的 `logg/alpha` 明显错配，`IC2391-0096` 又不够 metal-rich。因此目标矩阵的严格定量观测覆盖是 **0/12**，不能把模型网格图说成观测验证。

本地 PZ 运行产品也不完整。若统一采用 `giant logg=1.5`、`dwarf logg=5.0`，可用结构是 **6/12**。为了先画最直观的模型图，展示矩阵只把 metal-poor giant 改用 `logg=2.5`，因此达到 **8/12**；每个 panel 都明确写出真实 `logg`。rich/poor giant 的跨行差异不能解释成纯 metallicity effect。

## 数值锚点

- metal-poor：`[M/H] = -1.0`。
- metal-rich：`[M/H] = +0.5`。solar 不算 metal-rich。
- `alpha = 0.0`、`[C/M] = 0.0`。
- 严格矩阵：giant `logg=1.5, vmic=2 km/s`；dwarf `logg=5.0, vmic=1 km/s`。
- 展示矩阵：poor giant 用本地三温都存在的 `logg=2.5`；rich giant `1.5`；poor dwarf `4.5`；rich dwarf `5.0`。

这些节点来自现有 v1r2 MARCS-seeded PZ grid，而不是事后把观测参数四舍五入。运行产品位于 `results/m_star_emulator_v1r2_marcs100/cases/.../products/primary/*.npz`，已从 Garching 取回本地但被 `.gitignore` 忽略；普通 `rg --files` 会漏掉，需用 `rg -uu --files` 或 `find` 盘点。

## 12 格观测 coverage matrix

表中差值采用“观测减目标格点”。`alpha` 是观测表的 `[alpha/M]`。MISSING 表示此前下载的本地数据中没有合适对象，不用别的温度或重力硬凑。

| Teff | 金属/类型 | 已下载对象 | 观测 `Teff/logg/[Fe/H]/alpha` | `dTeff/dlogg/d[Fe/H]/dalpha` | 本地数据 | 定量？ |
|---:|---|---|---|---|---|---|
| 3200 | poor giant | MISSING | — | — | — | 否 |
| 3200 | rich giant | MISSING | — | — | — | 否 |
| 3200 | poor dwarf | MISSING | — | — | — | 否 |
| 3200 | rich dwarf | MISSING | — | — | — | 否 |
| 3500 | poor giant | MISSING | — | — | — | 否 |
| 3500 | rich giant | MISSING | — | — | — | 否 |
| 3500 | poor dwarf | MISSING | — | — | — | 否 |
| 3500 | rich dwarf | MISSING | — | — | — | 否 |
| 4000 | poor giant | HD102212, M1III/SRB | `3982/1.135/-0.423/+0.897` | `-18/-0.365/+0.577/+0.897` | `HD102212_R80k.fits` 与 `hd102212.tfits` | 否：SRB，alpha 大错配 |
| 4000 | rich giant | HD219215, M1.5III | `3871/0.826/-0.097/+0.529` | `-129/-0.674/-0.597/+0.529` | `HD219215_R80k.fits` | 否：只作 label-mismatch diagnostic |
| 4000 | poor dwarf | MISSING | — | — | — | 否 |
| 4000 | rich dwarf | IC2391-0096, M dwarf | `3728/4.9498/+0.0391/+0.0995` | `-272/-0.0502/-0.4609/+0.0995` | `IC2391-0096_R80k.fits` | 否：离 `+0.5` 太远；它适合作 3750 K solar-near 诊断 |

另有 `hd156274.tfits`，但统一表把它定为 `G9V`、`5334 K`，不属于本矩阵。Astropy 用只读 context manager 实查到 3 个 R80k FITS（`HD102212`、`HD219215`、`IC2391-0096`）和 2 个旧 tfits（`HD102212`、`HD156274`）。R80k 文件是 3-HDU 图像产品；tfits 表列为 `Wave/Flux/Sigm`。

## 本地 PZ atmosphere 覆盖

| 类别 | 3200 K | 3500 K | 4000 K | 覆盖 |
|---|---|---|---|---:|
| poor giant, `logg=2.5` | ✓ | ✓ | ✓ | 3/3 |
| rich giant, `logg=1.5` | MISSING | ✓ | ✓ | 2/3 |
| poor dwarf, `logg=4.5` | MISSING | MISSING | ✓ | 1/3 |
| rich dwarf, `logg=5.0` | MISSING | ✓ | ✓ | 2/3 |

严格固定 `logg=1.5/5.0` 时，poor giant 只有 3200/3500，rich giant 3500/4000，poor dwarf 0，rich dwarf 3500/4000，共 6/12。

## 三套谱的公平条件

主图是 TiO-rich `6650–6670 A` air 窗口，另做一个 4000 K metal-poor dwarf 的 Ca I `6155–6170 A` atomic 窗口。三条线都使用同一格的 PZ structured atmosphere、同一 elemental abundance、`alpha=0`、同一 `vmic`、`RV=0`、无旋转、同一 `R=80,000` Gaussian convolution、同一真空合成到 air 绘图转换，并以卷积后的 total/continuum 定义 normalized flux。没有拟合观测，也没有 continuum nuisance。

TiO 图的 line-list audit：

| pipeline | 实际传入窗口的 atomic / molecular / TiO | 说明 |
|---|---:|---|
| native Payne-Zero | `2,489 / 404,290 / 353,897` | PZ compiled native catalog；TiO 源自 Schwenke-family compiled assets |
| Korg + GALAH DR3 | `606 / 14,050 / 12,981` | Korg default GALAH DR3 loader；全库 `307,000` 条 |
| Korg + PZ/Kurucz | `2,434 / 401,639 / 351,246` | 读取 v5 导出的 PZ compiled Kurucz-family matched-transition TSV；不是本轮直接解析 raw Schwenke binary |

Ca 图中 Korg + VALD 实际 99 条，Korg + PZ/Kurucz 实际 2,072 条 atomic transitions。两套 Korg 输入真实不同，不是只换图例。

三者仍不只差 line list：native PZ 使用其存储 populations、continuum 与辐射转移；Korg 会把同一 PZ column-mass structure 映射为 planar atmosphere，并使用 Korg EOS、partition functions、molecular equilibrium、continuum、line profiles 与 transfer。因此 PZ 对 Korg 的差异不能归因成纯 line-list effect；只有两条 Korg 曲线之间主要是 line-list effect。

## 直接结果

TiO 窗口成功生成 **8/12** 格三谱。`RMS(Korg+GALAH - Korg+PZ/Kurucz)` 在可用格中为 `0.0071–0.0819`；PZ 与 Korg+GALAH 的 RMS 距离为 `0.0082–0.1580`。最冷 poor giant 从 3200 到 4000 K 的 PZ–Korg+GALAH RMS 由 `0.1029` 降到 `0.0082`，说明最冷端的 pipeline 差异明显放大。rich giant 3500 K 的差异最大（PZ–GALAH `0.1580`），但 poor/rich giant 的 `logg` 不同，不能把它解释成 metallicity 单变量效应。

Ca 最小对照在 `4000/4.5/-1.0`：PZ–Korg+VALD RMS `0.02824`，PZ–Korg+PZ/Kurucz `0.02693`，两套 Korg `0.03832`。

数据直接显示的是上述 normalized-spectrum RMS 与曲线形状差异。关于 TiO completeness、EOS 或 continuum 是主因，只能算待拆分解释；这里没有观测支持任何“谁更准”的排名。PZ campaign 的 `31/108` 没有用于判断光谱或 MARCS/Korg 稳定性。

## 论文图与复现

- TiO 8/12 矩阵：`results/m_star_cool_grid_v6_three_spectra/figures/cool_grid_tio_three_spectra.pdf` 与 `.png`。
- Ca 单节点：`results/m_star_cool_grid_v6_three_spectra/figures/cool_grid_ca_three_spectra_t4000_poor_dwarf.pdf` 与 `.png`。
- 每格 spectra：`results/m_star_cool_grid_v6_three_spectra/spectra/`。
- metrics 与逐格 atmosphere 绝对路径：`results/m_star_cool_grid_v6_three_spectra/metrics.json`、`ca_metrics.json`。
- 脚本：`experiments/reduced_state_emulator/compare_mstar_grid_three_spectra_v6.py`、`compare_mstar_ca_three_spectra_v6.py`、`korg_three_linelist_v6.jl`。

绘图直接调用 `payne_zero_figures.style.configure("PAPER")`、`style.DOUBLE`/`style.SINGLE` 和 `style.inward(ax)`；输出 vector PDF 与 300 dpi PNG，三条线同时用颜色和 solid/dashed/dotted 编码。两张 PNG 已目视检查。

运行命令：

```bash
.venv/bin/python experiments/reduced_state_emulator/compare_mstar_grid_three_spectra_v6.py
.venv/bin/python experiments/reduced_state_emulator/compare_mstar_grid_three_spectra_v6.py --plot-only
.venv/bin/python experiments/reduced_state_emulator/compare_mstar_ca_three_spectra_v6.py
```
