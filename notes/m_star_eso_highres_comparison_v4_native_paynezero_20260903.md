# IC2391-0096 原生 Payne-Zero 光谱重跑

原生 `payne_zero_synthesis` 在这两个窗口都比 v3 的 Korg + 同一个 Payne-Zero atmosphere 更接近 UVES-POP。Ca I 的 RMS 从 `0.082765` 降到 `0.061837`，TiO-rich 窗口从 `0.043975` 降到 `0.039074`。

| 窗口 | native PZ RMS / MAD / p95 | Korg v3 RMS | 观测噪声 RMS | native-Korg prediction RMS | residual correlation |
| --- | --- | ---: | ---: | ---: | ---: |
| Ca I 6155–6170 Å | `0.061837 / 0.040968 / 0.119468` | `0.082765` | `0.025818` | `0.054886` | `0.741` |
| TiO-rich 6650–6670 Å | `0.039074 / 0.024965 / 0.072245` | `0.043975` | `0.029164` | `0.035282` | `0.644` |

这次直接读取已收敛、`training_eligible` 的 3750 K dwarf primary product，没有重新调用 label initializer。光谱以原生 `R_grid=600,000` 采样；这只是本征 log-lambda 网格密度。随后固定施加 `v sin i=7.1124722692 km/s` 的 Gray rotation，再施加 `ΔRV=-0.6918215629 km/s` 和唯一一次 `R=80,000` 仪器 LSF，最后采样到 UVES-POP 像素。RV 和旋转没有重拟合。

UVES-POP 使用空气波长，Payne-Zero 原生网格使用真空波长。投影时先把观测像素从 air 转成 vacuum，图上仍显示 catalogue-rest air wavelength。观测仍按局部 95 百分位归一化；native 和 Korg 分别拟合相同形式的 `model × (a+b×x)` continuum nuisance。

原生原子线来自 `source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz`。分子线来自 `molecular_band_lines.npz`，TiO 是 manifest 记录的 Schwenke TiO `tio/schwenke.bin` 转换产物 `titanium_oxide_lines.npy`。

PI 复查确认，原生 Ca 合成上下文实际包含约 `2,100` 条原子线和 `445,609` 条分子线，其中约 `406,017` 条是 TiO；TiO-rich 上下文包含约 `2,489` 条原子线和 `404,290` 条分子线，其中约 `353,897` 条是 TiO。作为量级对照，Korg 的 GALAH HDF5 在 6645–6675 Å 中只有 `9,469` 条总线、其中 `8,373` 条 TiO。原生 PZ 的改善因此很可能主要受更完整的 Schwenke/Kurucz 线表推动，不能直接归功于辐射转移算法。

Ca I 6162 Å 的半深 FWHM 从 Korg 的 `30.23 km/s` 改善到原生 PZ 的 `24.59 km/s`，但观测只有 `17.17 km/s`，强线仍然过宽。TiO 窗口分成四个 5 Å 小段后，原生 PZ 在三段更好，在 6660–6665 Å 一段更差；所以整体 RMS 的改善真实存在，但不是每个局部谱段都占优。

这说明在这颗 M dwarf、这两个窄窗口和当前固定 nuisance 下，原生 Payne-Zero 完整 forward pipeline 更好；其中 TiO 已接近噪声水平，但仍有约 2.6% 的去噪后残差尺度。它不是严格的 radiation-transfer-only benchmark：native Payne-Zero 使用自己的原子和分子线表；Korg 的 TiO 窗口使用 GALAH full molecular list，Ca I v3 使用 atomic-only VALD list。差异同时包含线表与合成实现。

Payne-Zero campaign 的 solver 成功率没有进入光谱评分，也不用于判断光谱稳定性。标签不匹配的 M giant 没有参与比较。

产物：

- `results/m_star_eso_highres_comparison_v4_native_paynezero/metrics.json`
- `results/m_star_eso_highres_comparison_v4_native_paynezero/processed/`
- `results/m_star_eso_highres_comparison_v4_native_paynezero/spectra/`
- `results/m_star_eso_highres_comparison_v4_native_paynezero/figures/IC2391-0096_native_paynezero_vs_korg.png`
- `experiments/reduced_state_emulator/compare_mstar_eso_native_paynezero_v4.py`
