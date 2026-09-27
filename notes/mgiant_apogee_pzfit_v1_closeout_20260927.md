# M-giant 拟合管线在 DR20 APOGEE M 巨星上的测试 v1 — closeout

日期：2026-09-27

## 结论

ESO UVES 上使用的 Payne Zero M-giant 拟合管线（v4 (m,T) 快速模型 →
`fit_normalized_spectrum` → `refine_with_physical_atmosphere`）可以直接在
APOGEE H 波段运行：`mgiant_aspcap_fit_v1` 的 8 颗 DR20 M 巨星全部完成快速拟合与
收敛大气精修，8 个收敛大气都通过独立数值与通量检查，精修 8/8 成功（5 颗通过
快慢谱门槛，3 颗驻定）。

与 ASPCAP 的标签对比不成立为准确度结果：6/8 颗停在 v4 盒 logg=0.5 下界
（另 1 颗 0.51），Teff 平均高 165 K（rms 202 K），[M/H] 平均低 0.24 dex
（rms 0.25），偏差随 ASPCAP Teff 降低而增大（3200–3500 K：+140 至 +310 K；
3700–3900 K：+6 至 +76 K）。χ²/像素 227–1303，残差 RMS 2.7–4.9%，而噪声为
0.08–0.18%，模型系统误差仍主导拟合。这与 ESO 光学窗口中 PZ 与 Korg + MARCS
共同偏冷偏贫的方向不同，H 波段这里是偏热偏贫。

`mgiant_aspcap_fit_v1` 的角点结果（8 颗都在 4000 K / logg 0.5 / 低金属）主要来自
视向速度处理：mwmStar coadd 已在静止系，而 v1 把模型按 ASPCAP `v_rad` 再次
平移。同一模型谱上，按 `v_rad` 平移使 χ²/像素从 1196 升到 6074（86810089）、
从 585 升到 3971（96257659），与 v1 报告的约化 χ² 6262、3785 相当。v1 closeout
中"线强系统不符、拟合被推到网格角"的判断需以静止系重拟合为准（见下）。

## 方法

- 管线：`experiments/mgiant_apogee_pzfit_v1.py`，复用
  `mgiant_eso_bestfit_v1.V4FastModel` 与 `ConvergedModel`，只替换合成窗口
  （1510–1700 nm 真空，R_grid=120,000，原生 PZ 线表）与观测投影
  （`ObservedSpectrumOperator`，高斯 LSF R=22,500 + 拟合的额外高斯展宽）。
  生产 `fitter.apogee` 未使用：其标签初始化器训练在 4000 K 以上。
- 数据：mwmStar-0.6.0 coadd，flux 除以文件自带 continuum；数据与每个投影模型
  都用 v1 的 339 像素滑动中值伪连续谱归一化，再由拟合器逐芯片轮廓化线性乘性
  连续谱。坏像素 ivar=0 掩掉，只用三芯片内像素（7151–7231 个）。
- 参数：Teff、logg、[M/H]、RV、σ_extra；α=C=0、vmic=2 km/s 固定。起点为
  ASPCAP 标签（截到 v4 盒内），RV 起点由起点谱 ±30 km/s 扫描得到
  （−0.10 至 +0.36 km/s，确认 coadd 已在静止系），拟合范围 ±5 km/s。
  快速拟合最多 8 次迭代；精修门槛与 ESO 相同（轮廓化 RMS 差 < 2×10⁻³、
  χ²/像素增加 < 0.1、最多 4 个收敛大气）。
- 运行：Garching Node-06，CPU float64，每星一进程（约 18 GB）。
  86810089 与 96257659 另在本机 MPS float32 上跑快速拟合作交叉核对，
  参数路径与 Garching 在求解器 deck 精度上一致。

## 结果

标签写作 Teff/logg/[M/H]；χ²/像素从 ASPCAP 起点到快速拟合终点：

| sdss_id | ASPCAP | PZ 拟合 | 触到的盒面 | χ²/像素 | RV / σ_extra (km/s) | 精修 |
|---|---|---|---|---:|---|---|
| 86810089 | 3200/0.45/−0.10 | 3483/0.50/−0.29 | logg 下界 | 1194 → 893 | +0.06 / 3.37 | 驻定 |
| 65420634 | 3300/0.59/−0.03 | 3608/0.50/−0.23 | logg 下界 | 1027 → 579 | +0.12 / 3.83 | 驻定 |
| 66450191 | 3400/0.80/+0.05 | 3686/0.50/−0.21 | logg 下界 | 1611 → 950 | +0.08 / 3.88 | 驻定 |
| 114992838 | 3500/0.45/−0.44 | 3638/0.50/−0.53 | logg 下界 | 821 → 619 | +0.03 / 3.93 | 门槛一致 |
| 54948768 | 3600/1.10/+0.11 | 3815/0.51/−0.25 | — | 974 → 639 | +0.07 / 2.99 | 门槛一致 |
| 96257659 | 3700/0.83/−0.22 | 3776/0.50/−0.54 | logg 下界 | 584 → 457 | +0.08 / 2.68 | 门槛一致 |
| 74858601 | 3800/0.65/−0.55 | 3812/0.50/−0.71 | logg 下界 | 253 → 227 | +0.36 / 3.88 | 门槛一致 |
| 77720555 | 3900/1.08/−0.36 | 3906/0.85/−0.65 | — | 1582 → 1303 | +0.16 / 3.27 | 门槛一致 |

- 所有快速拟合都以 line search 失败停止（与 ESO 相同），用 12–38 次合成、
  8–24 min；每星精修只用 1 个收敛大气（3–5 次求解迭代，1–6 min），
  快慢谱差 2×10⁻⁴–1.1×10⁻³ RMS。驻定的 3 颗在收敛大气上 χ²/像素增加
  2.9–4.8，超过 0.1 门槛，但相对量级不到 1%。
- 相对 ASPCAP 起点，拟合使 χ² 降 10–44%。

### v1 网格拟合的静止系对照

用 v1 的 78 节点缓存库与 v1 的 `fit_star`，只把 RV 平移设为 0：

| sdss_id | 按 `v_rad` 平移（v1） | 静止系 |
|---|---|---|
| 86810089 | 4000/0.5/−0.5，6262 | 3600/0.5/+0.0，2336 |
| 65420634 | 4000/0.5/−1.0，6829 | 3600/0.5/+0.0，1773 |
| 66450191 | 4000/0.5/−1.0，10873 | 3700/1.5/+0.5，3145 |
| 114992838 | 4000/0.5/−1.0，6114 | 3600/0.5/+0.0，2681 |
| 54948768 | 4000/0.5/−1.0，5867 | 4000/0.5/+0.0，1714 |
| 96257659 | 4000/0.5/−1.0，3784 | 4000/0.5/−0.5，1755 |
| 74858601 | 4000/0.5/−1.0，3568 | 4000/0.5/−0.5，960 |
| 77720555 | 4000/0.5/−1.0，6614 | 4000/0.5/−1.0，4803 |

静止系下约化 χ² 降 1.4–4.7 倍，[M/H] 不再全在 −1。节点拟合仍大多落在
logg=0.5 与盒面上，χ² 比连续拟合高 2–5 倍（节点稀疏、展宽固定 3.5 km/s）。

## 局限

- logg=0.5 是 v4 训练盒下界，6/8 颗停在这里；真实最优可能在盒外，Teff 与
  [M/H] 随之相关偏移。
- 只用原生 PZ 线表；`linelist_calibration/data/sun_arcturus_fts_hband_shared.npz`
  原子线校准与 H 波段分子线（OH、CO、CN、H₂O）的差分检验未做。
- α=C=0、[C/M]=0、vmic=2 km/s 固定；M 巨星第一次挖掘后的 C/N 与 ¹²C/¹³C
  直接影响 CO/CN 带。LSF 为常数 R=22,500 高斯，不是逐 visit/fiber LSF；
  未做 telluric 与 airglow 掩码。
- ASPCAP 本身不是真值；样本只有 8 颗，SNR 531–1258。
- 精修门槛与 ESO 相同，未为 H 波段另行预注册；`accuracy_validated` 意义上
  没有准确度阈值。

## 文件

- 脚本：`experiments/mgiant_apogee_pzfit_v1.py`（`fast --star ID [--physical]`、
  `collect --stars-dir D`）
- 启动：`experiments/reduced_state_emulator/run_mgiant_apogee_pzfit_garching_20260927.sh`
- 结果：`results/mgiant_apogee_pzfit_v1/`
  - `summary.json`、`labels_vs_aspcap.png`
  - `garching/stars/<sdss_id>/`（`summary.json`、`fast/`、`physical/`、模型通量）
  - `garching/logs/`、`logs/`（本机交叉核对）、`launch_status.json`
  - `v1_grid_rest_frame_control.json`
- Garching 运行目录：`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mgiant-eso-pzfit-20260924`
  （收敛大气在 `results/mgiant_apogee_pzfit_v1/stars/<sdss_id>/solves/`）
