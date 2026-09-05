# ESO M 星比较 v2：加入本地 TiO/分子线表

## 结论

本地确实有一套现在就能用的 TiO 线表，不需要下载约 626 MB 的 Kurucz Git-LFS 文件：

`/Users/jdli/Project/jorg/Korg.jl-1.0.1/data/linelists/GALAH_DR3/galah_dr3_linelist.h5`

它是实际 HDF5（2,652,174 bytes），`Korg.get_GALAH_DR3_linelist()` 成功读出 307,000 条线：19,206 条原子线、287,794 条分子线，其中 TiO 236,552 条。GALAH 线表不是全光学连续覆盖；经典 7054 Å 附近没有线，因此 v2 只选择它实际覆盖、同时避开 UVES bad/telluric mask 的 `6650–6670 Å`。含 5 Å 合成边缘的 `6645–6675 Å` 中有 9,469 条线，其中 9,088 条分子线、8,373 条 TiO。

这个窗口的分子效应不是只靠数行数命名。用同一 GALAH 线表构造 `no-TiO` 和 atomic-only 两个消融后，原始连续谱归一化合成谱的变化为：

| target | PZ `RMS(full−noTiO)` | MARCS `RMS(full−noTiO)` | PZ `RMS(noTiO−atomic)` | MARCS `RMS(noTiO−atomic)` |
|---|---:|---:|---:|---:|
| IC2391-0096 | `0.2068` | `0.2366` | `0.0188` | `0.0205` |
| HD219215 | `0.0723` | `0.0810` | `0.0354` | `0.0344` |

这说明该窗口的分子效应以 TiO 为主，尤其是 M dwarf；其它分子并非完全为零，但明显较小。

full 分子线表也确实降低了同一窗口的观测残差：

| target | atmosphere | RMS atomic-only | RMS no-TiO | RMS full |
|---|---|---:|---:|---:|
| IC2391-0096 | Payne-Zero | `0.059611` | `0.059610` | `0.051989` |
| IC2391-0096 | MARCS | `0.059755` | `0.059886` | `0.051918` |
| HD219215 | Payne-Zero | `0.083355` | `0.080240` | `0.070689` |
| HD219215 | MARCS | `0.083436` | `0.080308` | `0.069746` |

对 M dwarf，去掉 TiO 后的残差几乎退回 atomic-only 水平；对 M giant，其它分子先带来约 `0.0031` 的 RMS 改善，TiO 再带来约 `0.0096–0.0106`。所以 v1 的“缺 TiO/关键分子线表”现在从限制变成了直接可测的效应，且这个窗口的改善确实主要来自 TiO。但它仍没有把 atmosphere 胜负分开：

- `IC2391-0096`：full-list RMS 为 PZ `0.051989`、MARCS `0.051918`，MARCS 数值低 `0.0000715`。这只有总残差的约 `0.14%`，不足以作稳健排名，判断仍是**不能判定**。
- `HD219215`：full-list RMS 为 PZ `0.070689`、MARCS `0.069746`，MARCS 数值低 `0.000943`。但该星相对采用节点仍有 `Δlogg=+0.674 dex`、`Δalpha=-0.529 dex`，只能叫 label-mismatch diagnostic，不能用它判 atmosphere 胜负。

## 公平性保持不变

v2 沿用 v1 的同一批 UVES-POP R=80,000 像素、PIXMASK/QUALITY/bad/telluric mask、固定文献 RV、节点标签、丰度、microturbulence、`v sin i`、平面几何、Korg 1.0.1、仪器分辨率、air wavelength 和两自由度乘法连续谱。PZ 与 MARCS 的 primary 比较都用完整 GALAH list；`no-TiO` 与 atomic-only 仅用于 sensitivity control。没有分别优化物理标签，也不报没有协方差依据的 reduced chi-square。

`IC2391-0096` 仍是近邻诊断：`ΔTeff=+22 K, Δlogg=+0.050, Δ[M/H]=-0.039, Δalpha=-0.100`。`HD219215` 使用 logg=1.5 是因为数学上更近的 v1r2 logg=0.5 case 不 eligible，不是任意换节点。

Payne-Zero `31/108` 没有进入任何 v2 评分，也不是 MARCS 更稳定的证据。PZ 的内部 flux/path residual 仍只评价 PZ 解本身。

## jorg 线表审计

可直接使用的路径：

- Korg GALAH DR3 HDF5：真实文件；Korg reader、TiO equilibrium constant、TiO partition function和实际合成都已成功。
- `jorg/jorg/data/linelists/exomol/CaH/`：真实 ExoMol states/transitions，可补 CaH，但不是 TiO。
- `jorg/kurucz/molecules/` 下的 CaH、FeH、CH、CO、OH、MgH 等多份 `.dat` 是真实 MB 文件，可由 Kurucz/SYNTHE 的分子预处理器使用。

现在不能直接使用的路径：

- `kurucz/molecules/tio/schwenke.bin` 工作树只有 134 bytes，是 LFS pointer，声明真实大小 603,911,984 bytes。
- `kurucz/molecules/tio/eschwenke.bin` 只有 133 bytes，是 LFS pointer，声明真实大小 22,621,208 bytes。
- Korg 1.0.1 的 Kurucz molecular-text parser 会直接抛出“不支持分子 Kurucz line list”；jorg Python reader 的 `read_molecular_linelist` 仍是 TODO 并返回空列表。
- jorg Python 的 `molecular_cross_sections` 参数目前只出现在 `synthesize` 接口和文档中，没有接入主合成计算；其中 cross-section builder 也明确是 simplified approximation。它不能替代本轮的 Korg 实际合成。

因此当前最干净、公平、无需新下载的路线就是本轮采用的：同一 Korg backend + 同一 GALAH HDF5 + 两套 atmosphere。Kurucz/SYNTHE 路线只有在下载 TiO LFS 数据、编译 line list、并把 PZ 与 MARCS 都转换为同一 SYNTHE atmosphere 输入后才可能成为第二套独立 backend 检查；现在不能把它当作已完成证据。

## 产物

- [metrics.csv](../results/m_star_eso_highres_comparison_v2/metrics.csv)
- [metrics.json](../results/m_star_eso_highres_comparison_v2/metrics.json)
- [line-list audit JSON](../results/m_star_eso_highres_comparison_v2/linelist_audit.json)
- [IC2391-0096 full/no-TiO comparison](../results/m_star_eso_highres_comparison_v2/figures/IC2391-0096_tio_6650_observed_model_residuals.png)
- [HD219215 full/no-TiO comparison](../results/m_star_eso_highres_comparison_v2/figures/HD219215_tio_6650_observed_model_residuals.png)
- [Python processing script](../experiments/reduced_state_emulator/compare_mstar_eso_highres_v2.py)
- [Korg synthesis script](../results/m_star_eso_highres_comparison_v2/models/korg_synthesize_tio_window.jl)

两张图已目视检查，波长对齐、full/no-TiO 控制、残差方向和图例均正常。底部消融曲线比较的是合成谱自身的连续谱归一化 flux；上两栏的观测评分则对每个模型应用相同形式的局部连续谱 nuisance。

## 还缺什么

本轮验证了格式可读、species 数量、实际 TiO 敏感性以及观测残差改善，但没有审计 GALAH TiO 的完备度与 oscillator-strength 准确度。总体 atmosphere 判决仍需要更多有线表覆盖的 TiO 窗口，以及标签更匹配的非变 M giant，或在 HD219215 实测标签上得到合格 PZ atmosphere。
