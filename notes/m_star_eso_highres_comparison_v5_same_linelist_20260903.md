# IC2391-0096 同 transitions 对照

把 line-list completeness 差异拿掉后，native Payne-Zero 和 Korg 明显靠近了。Korg 在两个窗口的 RMS 都略低，但优势很小：Ca I 低 `0.00366`，TiO-rich 低 `0.00080`。这不足以说明 Korg 整体辐射转移更好。

| 窗口 | 公共 transitions | native PZ RMS / MAD / p95 | Korg RMS / MAD / p95 | noise RMS | 两模型 prediction RMS | residual correlation |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Ca I 6155–6170 Å | 2,072 atomic | `0.085066 / 0.062337 / 0.154945` | `0.081404 / 0.057896 / 0.150578` | `0.025818` | `0.016965` | `0.980` |
| TiO-rich 6650–6670 Å | 2,434 atomic + 404,290 molecular | `0.039074 / 0.024965 / 0.072245` | `0.038270 / 0.025200 / 0.072811` | `0.029164` | `0.017136` | `0.902` |

## 最直接的变化

相对 v4 各自完整 pipeline 的对照，Ca 的 native–Korg prediction RMS 从 `0.054886` 降到 `0.016965`，下降 69%；TiO 从 `0.035282` 降到 `0.017136`，下降 51%。残差相关性同时从 Ca `0.741`、TiO `0.644` 升到 `0.980`、`0.902`。先前两套完整 pipeline 的大部分差异确实来自 line list，而不是 atmosphere。

Ca 强线仍然同样过宽：观测 FWHM 是 `17.17 km/s`，native PZ 是 `29.09 km/s`，Korg 是 `29.29 km/s`。两套 synthesizer 在相同 transitions 下给出几乎一样的错误线宽，压力展宽参数或共同 atmosphere 条件比辐射转移代码更可疑。

## 公共线表核对

Ca 窗口的 PZ source catalog 原有 2,097 条 metal records。Korg 只提供 I–III 级 partition functions，因此将 25 条 IV–VI 级 transitions 从两边同时剔除，公共集合为 2,072 条。PZ 和 Korg 实际使用数都是 2,072。atomic number、ion stage、log gf 和 E_low 完全相同；wavelength 最大回读差 `1.14e-13 nm`，gamma_vdW 最大相对差 `2.25e-16`，都只是文本回读舍入。gamma_rad 和 gamma_stark 完全回读一致。

TiO-rich 窗口双方实际都使用 406,724 条 records：2,434 条 atomic 和 404,290 条 molecular。其中 TiO 353,897 条、AlO 30,427 条、VO 18,994 条，其余为 H2、CH、OH、C2、CN、MgH、CaH 和 NaH。wavelength、aggregate species code、isotope-adjusted log gf、E_low 和 damping 字段全部通过逐项容差核对。

TiO 部分只能叫 **matched-transition list**，不能叫 identical line list。PZ compiled molecular records 已把同位素权重并入 gf，只保留聚合 molecular species code，无法恢复每条线原始 isotopologue label。

Korg 的 `use_internal_reference_linelist=true` 只用于 anchored `tau5000` 的参考 opacity，不进入 6155–6170 Å 或 6650–6670 Å 的目标窗 line opacity。目标窗 Korg 输入只有导出的公共 transitions；`line_buffer=1000 Å` 保证所有导出 records 都被保留，实际 used count 与输入 count 相同。

## 仍然不同的物理

相同 transitions 去除了 line-list completeness 差异，但这仍不是纯 radiation-transfer A/B。两边的 EOS、partition functions、ionization 和 molecular equilibrium、continuum opacity、阻尼随温度和密度的缩放、line profile、辐射转移实现仍不同。Korg 还把同一个 PZ column-mass structure 重新映射为 planar atmosphere。两边最终使用相同的固定 RV、rotation、`R=80,000` LSF 和观测像素投影；continuum nuisance 形式相同，但系数分别拟合。

Ca continuum `(a,b)`：native `(0.97248, -0.03694)`，Korg `(0.93074, -0.05701)`。TiO continuum `(a,b)`：native `(1.21582, 0.00855)`，Korg `(1.10848, 0.01980)`。

Payne-Zero campaign 的 `31/108` 没有进入评分，也不用于判断稳定性。标签不匹配的巨星没有参与。

产物：

- `results/m_star_eso_highres_comparison_v5_same_linelist/metrics.json`
- `results/m_star_eso_highres_comparison_v5_same_linelist/inputs/`
- `results/m_star_eso_highres_comparison_v5_same_linelist/spectra/`
- `results/m_star_eso_highres_comparison_v5_same_linelist/processed/`
- `results/m_star_eso_highres_comparison_v5_same_linelist/figures/IC2391-0096_ca_i_6162_same_atomic_transitions.png`
- `results/m_star_eso_highres_comparison_v5_same_linelist/figures/IC2391-0096_tio_6650_matched_transitions.png`
- `experiments/reduced_state_emulator/compare_mstar_eso_same_linelist_v5.py`
- `experiments/reduced_state_emulator/korg_synthesize_pz_matched_lines_v5.jl`
- `experiments/reduced_state_emulator/korg_synthesize_pz_matched_mixed_lines_v5.jl`
