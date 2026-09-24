# ESO UVES M giant：Payne-Zero 与 Korg.jl + MARCS 最佳拟合对比 v1

日期：2026-09-24

## 结论

7 颗 ESO 归档 UVES M/晚 K 巨星在 6480–6735 Å 上的最佳拟合中，Korg.jl + MARCS
的 χ² 在每颗星上都低于 Payne-Zero：同一组 78 个标签节点上 χ² 比为 0.81–0.97，
连续拟合为 0.70–0.89；残差 RMS 从 PZ 的 0.051–0.079 降到 0.048–0.067。
两者都远未到噪声水平（χ²/像素 107–925）。

差距来自线表，不来自大气结构，也不来自合成代码：PZ 用完整 GALAH 线表并在分子平衡中加入
ZrO 后，同大气下 χ²_Korg/χ²_PZ = 0.98–1.09，7 颗中 5 颗 PZ 更低（见"PZ 使用 Korg 线表"与
"PZ 加入 ZrO 分子平衡"两节）。把 Korg 直接跑在同一批 78 个
Payne-Zero 收敛大气上，χ² 与 Korg + MARCS 相差不到 1%，7 颗星中 6 颗选中
同一节点。按子窗口拆开，在完全相同的 PZ 大气上，Korg + GALAH 线表在
6480–6640 Å 对 7 颗星都更接近观测，在 PZ 最佳节点和 Korg 最佳节点上都是如此；
总 χ² 的差距由这半个窗口贡献。红半窗 6640–6735 Å 的胜负随所用节点改变
（PZ 最佳节点上 PZ 占 6/7，Korg 最佳节点上约各半），不构成稳定结论。

参数层面，三条流水线共享同一个系统偏差：两颗有 Gaia benchmark 参考值的星，
所有臂都偏冷偏贫。连续 Korg + MARCS 拟合给出 α Cet 3529/0.50/−0.87
（GBS 3796/0.68/−0.45）、α Tau 3685/1.09/−0.64（GBS 3927/1.11/−0.37），
节点拟合都落在 3600 K。在 PZ 最佳节点上，6480–6700 Å 三个子窗口里两套
模型的中位线深对 7 颗星都浅于观测。这个偏差不随连续谱模型改变，也不属于
某一条流水线，所以本轮只能比较两条流水线的相对拟合质量，不能把任一方的
参数当作准确度结果。

## 样本与数据

全部来自 ESO Phase 3 归档（`ivoa.ObsCore`，`instrument_name='UVES'`），
程序 `266.D-5655(A)`（UVES-POP），同一 580 nm 设置，R=74,450，
472.7–683.5 nm，`FLUXCAL=ABSOLUTE`，拓扑心波长（air）。每颗星取覆盖拟合窗口的
最高 SNR 产品：

| 星 | 光谱型 | ESO 产品 | SNR | 拟合像素 |
|---|---|---|---:|---:|
| α Cet | M1.5IIIa | ADP.2020-08-04T13:42:09.676 | 366 | 13928 |
| α Tau | K5+III | ADP.2021-08-29T17:14:32.447 | 483 | 13899 |
| ψ Phe | M4III | ADP.2021-08-31T16:17:29.615 | 244 | 13919 |
| φ Aqr | M1.5III | ADP.2020-08-10T12:46:04.636 | 454 | 13934 |
| m Vir | M1+III | ADP.2020-08-14T11:42:02.428 | 412 | 13898 |
| 87 Vir | M2III | ADP.2020-08-14T11:42:02.474 | 416 | 13897 |
| ER Vir | M4III | ADP.2021-08-27T05:08:20.475 | 355 | 13909 |

参考标签只用 Gaia FGK benchmark（本地 SVO GBS 目录）：α Cet、α Tau。
Borisov et al. (2023) 对这些 M 巨星给出的标签不适合作参考（如 α Tau
[Fe/H]=−1.16、α Cet 4180 K/+0.52、[α/Fe] +0.5 至 +1.2），只存入
`targets.json` 作背景。α Tau 是 K5 巨星，作为暖端锚点保留。

## 方法

四个模型臂共享同一观测处理、像素掩码、air→vacuum 换算、仪器核、
展宽/RV 扰动与连续谱，只有本征归一化谱不同：

| 臂 | 大气 | 合成与线表 | 标签集 |
|---|---|---|---|
| `pz_nodes` | Payne-Zero 收敛产品（`m_star_cool_corpus_mgiant_v2`） | 原生 PZ 合成（Kurucz 原子 + Schwenke TiO 等分子），R_grid=600k | 78 节点 |
| `marcs_nodes` | Korg `interpolate_marcs`（巨星为球面） | Korg 1.0.1 + GALAH DR3（80,741 条，77,055 条分子线） | 同 78 标签 |
| `korg_pzatm` | 同 78 个 PZ 大气（平面，τ₅₀₀₀ 由 Korg 连续不透明度积分） | 同上 | 同 78 标签 |
| `marcs_dense` | Korg `interpolate_marcs` | 同上 | 3200–4200 K/50 K × logg 0–2.5/0.25 × [M/H] −1.5–+0.5/0.25，共 2079；三线性谱插值连续拟合 |

另报 `marcs_dense_pzbox`：同一连续拟合限制在 PZ 语料的标签盒内。所有臂
vmic=2 km/s、[α/M]=0、标度太阳丰度。

- 窗口：6480–6735 Å（air），GALAH DR3 第 3 波段，是 Korg 自带线表在光学里
  唯一带 TiO 的区间。掩去 Hα 6555–6572 Å（色球）和 Li I 6707.0–6708.8 Å
  （巨星锂亏损，各代码默认 A(Li) 不同，与大气无关）。
- 前向模型：本征谱重采样到公共 0.25 km/s 对数网格；高斯核
  σ² = σ²_LSF(R=74,450) + σ²_extra；RV 平移后线性插值到观测像素；
  连续谱是乘性三次 B 样条，25 Å 节点（13 个系数），逐模型线性最小二乘求解。
  按 ERR 列逆方差加权。
- RV：每星用两臂同一标签（3800/1.5/0）模板扫描，两模板结果相差 ≤0.1 km/s，
  取均值作起点；有 `ESO QC VRAD BARYCOR` 的三颗星，拟合 RV 与
  Borisov RV − BARYCOR 之差（拟合减预期）为 −0.04（α Tau）、−0.18（φ Aqr）、−0.60 km/s
  （α Cet，LB 型变星）。
- 节点臂：78 节点 × 17 个 σ_extra（0–8 km/s）栅格，取前 5 名再连续优化
  (RV, σ_extra)。连续臂：以最佳稠密节点为起点，对
  (Teff, logg, [M/H], RV, σ_extra) 做 Nelder–Mead；14 次拟合全部收敛。
  无约束拟合只有 87 Vir 触到 [M/H]=−1.5 网格下界；PZ 盒内拟合中 α Cet
  停在 logg=0.5、87 Vir 停在 [M/H]=−1.0 的盒边界。

## 结果

标签写作 Teff/logg/[M/H]；χ² 比以 `pz_nodes` 为分母：

| 星 | GBS 参考 | PZ 节点 | Korg+MARCS 同节点 | Korg on PZ 大气 | Korg+MARCS 连续 | χ² 比（MARCS 节点 / PZ 大气 / 连续） | RMS（PZ / 连续） |
|---|---|---|---|---|---|---|---|
| α Cet | 3796/0.68/−0.45 | 3600/1.50/−1.00 | 3600/0.50/−0.50 | 3600/0.50/−0.50 | 3529/0.50/−0.87 | 0.843 / 0.847 / 0.740 | 0.0640 / 0.0546 |
| α Tau | 3927/1.11/−0.37 | 3600/0.50/−1.00 | 3600/0.50/−1.00 | 3600/0.50/−1.00 | 3685/1.09/−0.64 | 0.928 / 0.935 / 0.883 | 0.0508 / 0.0478 |
| ψ Phe | — | 3400/1.50/−1.00 | 3600/2.50/+0.00 | 3600/2.50/+0.00 | 3429/1.50/−0.62 | 0.811 / 0.808 / 0.771 | 0.0791 / 0.0673 |
| φ Aqr | — | 3750/2.50/−0.50 | 3600/0.50/−0.50 | 3600/0.50/−0.50 | 3622/1.25/−0.56 | 0.936 / 0.945 / 0.852 | 0.0718 / 0.0660 |
| m Vir | — | 3500/1.50/−1.00 | 3500/0.50/−0.50 | 3700/1.50/+0.00 | 3536/0.87/−0.65 | 0.815 / 0.813 / 0.698 | 0.0638 / 0.0524 |
| 87 Vir | — | 3500/1.50/−1.00 | 3500/1.50/−1.00 | 3500/1.50/−1.00 | 3372/0.72/−1.50 | 0.971 / 0.978 / 0.894 | 0.0587 / 0.0555 |
| ER Vir | — | 3400/1.50/−1.00 | 3600/2.50/+0.00 | 3600/2.50/+0.00 | 3505/1.94/−0.34 | 0.856 / 0.862 / 0.826 | 0.0756 / 0.0655 |

PZ 盒内连续拟合只在 87 Vir 上与无约束结果不同（3484/1.49/−1.00，χ² 比 0.963）。
PZ 节点拟合 7 颗中有 6 颗落到 [M/H]=−1.0、6 颗落到 logg 1.5 或 2.5，
这反映该不规则语料在 logg 0.5–1.5、[M/H] −0.5 附近的节点空缺
（logg 1.5/[M/H]=−0.5 只有 3100、3200 K；logg 2.0 为空），不宜读作物理偏好。

### 同一大气上的合成差异

固定一个 PZ 大气节点及其 RV/展宽，比较原生 PZ 合成与 Korg 在同一 PZ 大气上
的合成（各自拟合连续谱）的观测残差 RMS。节点分别取 `pz_nodes` 最佳和
`korg_pzatm` 最佳，因为前者对 PZ 有选择优势，后者对 Korg 有选择优势。
表中为残差更小的星数（PZ : Korg）：

| 子窗口 | PZ 最佳节点 | Korg 最佳节点 |
|---|---:|---:|
| 6480–6555 Å | 0 : 7 | 0 : 7 |
| 6572–6640 Å | 0 : 7 | 0 : 7 |
| 6640–6700 Å（TiO 带头） | 6 : 1 | 3 : 4（α Cet 两者相等至 1e-4） |
| 6700–6735 Å | 6 : 1 | 4 : 3 |

在 PZ 最佳节点上，两种合成之间的 RMS 为 0.019–0.063，与各自观测残差
（0.044–0.099）同量级。6480–6700 Å 三个子窗口中两模型的中位线深对 7 颗星
都浅于观测（例如 α Cet 6640–6700 Å：观测 0.157，PZ 0.102，Korg 0.111）；
6700–6735 Å 里 PZ 与观测相当或略浅，Korg 在 α Tau、φ Aqr、87 Vir 上深于观测。
在 Korg 最佳节点上，ψ Phe、ER Vir 取 3600/2.50/+0.00，红半窗里 PZ 比观测深，
Korg 略浅（ψ Phe 6640–6700 Å：观测 0.212，PZ 0.221，Korg 0.209）。

### 连续谱敏感性

把 B 样条节点间距换成 50、100、300 Å（8、6、4 个系数）重跑三个节点臂：
28 个（间距 × 星）组合中 Korg 两臂的 χ² 都低于 PZ，比值 0.77–0.99；
α Tau 在所有间距下三臂都停在 3600/0.50/−1.00。冷偏差不是 25 Å 样条吸收
TiO 带深造成的。

## Payne Zero 自有拟合

PZ 另用自己的 `fitter` 做了连续拟合（`pz-fit` 阶段，Garching Node-04）。快速模型为
v4 M-giant (m,T) 三种子中位数 → 物理重建 → 原生 PZ 合成 → `ObservedSpectrumOperator`；
`fit_normalized_spectrum` 在 v4 训练盒内拟合 Teff、logg、[M/H]、RV、σ_extra，起点为
PZ 最佳节点，连续谱用同一 25 Å B 样条。随后 `refine_with_physical_atmosphere` 在候选
标签处用未改动的求解器从 v4 起点解到收敛（`solve_experimental_point`，需通过数值检查与
M-giant 通量门槛），门槛为轮廓化 RMS 差 < 2×10⁻³、χ²/像素增加 < 0.1，最多 4 个收敛大气。
两路回调都按求解器 deck 精度取标签（Teff 取整 K，logg、[M/H] 取 1e-4），收敛产品因此
记录的正是其求解标签。

| 星 | PZ 拟合（收敛大气） | 精修结果 | χ²_Korg连续/χ²_PZ拟合 |
|---|---|---|---:|
| α Cet | 3503/0.64/−0.95 | 达到 4 个大气上限 | 0.85 |
| α Tau | 3617/0.55/−0.96 | 驻定 | 0.88 |
| ψ Phe | 3372/1.13/−1.00 | 达到上限 | 0.77 |
| φ Aqr | 3733/2.45/−0.52 | 快慢谱一致 | 0.85 |
| m Vir | 3478/0.55/−0.96 | 驻定 | 0.77 |
| 87 Vir | 3500/1.50/−1.00 | 驻定 | 0.89 |
| ER Vir | 3400/1.50/−1.00 | 快慢谱一致 | 0.83 |

13 个收敛大气上快慢谱差 7.6×10⁻⁵–4.1×10⁻³ RMS。连续拟合使 χ² 比最佳节点低至多 13%
（α Cet、m Vir），其余 5 颗变化不超过 0.3%。Korg + MARCS 连续拟合的 χ² 仍对每颗星更低
（0.77–0.89）；两颗 benchmark 星的 PZ 连续标签仍偏冷偏贫，3 颗停在 v4 盒的 [M/H]=−1 下界。
所有快速拟合都以 line search 失败停止。机器可读结果：`pz_fit_results.json`，各星
`pz_fit/<star>/`（快速轨迹、物理检查、收敛产品），运行记录 `pz_fit/launch_status.json`，
发射脚本 `experiments/reduced_state_emulator/run_mgiant_eso_pzfit_garching_20260924.sh`。

## PZ 使用 Korg 线表

把 Korg 的 GALAH DR3 窗口线表喂给原生 PZ 合成（`pz-galah` 阶段），在同样 78 个保存的 PZ
大气上合成，再用共用拟合器在保存标签上拟合（`fit-extra`）。原子线按 PZ 解析目录格式
写入并经 PZ 自己的目录构建器（实际使用 3,469 条；230 条 ABO 线按 3500 K 等效 γ_vdW 换算），
分子线直接写成 compiled 分子数组（72,454 条）；H、He 线、EOS、配分函数与连续谱仍为 PZ 原生。
PZ 打包的分子平衡中没有 ZrO，这一步不带入 4,601 条 ZrO 线（加入 ZrO 的比较见下一节）。分子注入做过往返对照：把 PZ 原生
的 229 万条同类分子线转成 GALAH 格式再注入，与直接注入的合成谱最大差 2.8×10⁻⁸。

ZrO 对 Korg 影响很大，因此另做了去掉 ZrO 的 Korg 臂（`korg_pzatm_nozro`）作为同线表对照。
保存标签上的 χ²：

- 换成 GALAH 线后，PZ 的 χ² 对每颗星下降 3–13%。
- 同线表（GALAH 去 ZrO）、同大气下，χ²_Korg/χ²_PZ = 0.98–1.08，4/7 颗 PZ 更低；
  6480–6555 Å 两套合成的差从各用自有线表时的 0.026–0.082 RMS 降到 0.005–0.012。
- 6700–6735 Å 两套合成仍差 0.011–0.044 RMS，是 TiO 主导的宽带状差异，不是 Ca I 6717；
  这里 PZ 在 14 个星×大气组合中 13 个更接近观测，原因未确定。
- ZrO 使 Korg 的 χ² 降低 2–15%，拟合后在 6480–6555 Å 改变谱 0.020–0.047 RMS，
  6700–6735 Å 不超过 8×10⁻⁴。原先"蓝半窗 Korg 7/7 更好"主要来自线表，尤其是 ZrO。

所以原生 PZ 与 Korg 的差距在线表，不在大气或合成代码；观测谱在有 ZrO 线时拟合更好，但
恒星是否真有太阳标度 Zr 下这么强的 ZrO 吸收，本轮没有检验。结果：`fit_pz_galah.json`、
`fit_korg_pzatm_nozro.json`、`pz_galah_inventory.json`、`pz_galah_nodes.npz`、`galah_lines.tsv`。

## PZ 加入 ZrO 分子平衡

PZ 打包的合成分子表（190 个分子、23 个方程）没有 Zr。本实验在影子 source-catalog 根下
（`zro_equilibrium/source_catalogs/`，其余文件为符号链接）把 Zr 作为元素方程插在电子方程前，
加入 Zr I–III 与 ZrO（Kurucz 码 840，线物种 id 570），PZ 打包数据和默认行为不变
（`zro-table`、`pz-galah-zro` 阶段；本进程内注册物种 id 与质量）。ZrO 常数取自 Korg 使用的
Barklem & Collet (2016) 数据：D₀ = 7.90 eV 使 PZ 的 Saha 形式线布居在 2500–4500 K 与 Korg
的 n/U 相差 < 6×10⁻⁴ dex；网络形成常数的多项式拟合 1500–8000 K 残差 0.004 dex。同一构造用于
TiO 时给出 D₀ = 6.869 eV（PZ 表中 6.87），线布居与 Korg 相差 ≤ 0.003 dex。对照：不加 ZrO 时
从求解列重建 EOS 状态，合成谱与已存产品最大差 6×10⁻⁸。在同一大气上，PZ 的 ZrO 在 6482–6557 Å
造成的平均通量下降约为 Korg 的 90%。

保存标签上的结果（完整 GALAH 线表，双方都有 ZrO，同一 PZ 大气）：

- ZrO 使 PZ 的 χ² 再降 2–13%；相对 PZ 自有线表降 6–24%。
- χ²_Korg/χ²_PZ = 0.98–1.09，7 颗中 5 颗 PZ 更低（α Cet 0.99、m Vir 0.97 为 Korg 略低）。
- 6480–6555 Å 两套合成差 0.006–0.022 RMS；6700–6735 Å PZ 在 14 个星×大气组合中全部更好。

结果：`fit_pz_galah_zro.json`、`pz_galah_zro_nodes.npz`、`pz_galah_zro_inventory.json`、
`zro_equilibrium/zro_table.json`、`zro_equilibrium/korg_molecular_constants.tsv`（由 `experiments/mgiant_eso_bestfit_v1_korg_constants.jl` 生成）。

## PZ 自有拟合 + GALAH 与 ZrO

用 PZ 自己的 fitter 在 GALAH 线表（含 ZrO，分子平衡加入 Zr/ZrO）下重做连续拟合
（`pz-fit --linelist galah_zro`，Garching Node-04；起点为 `fit_pz_galah_zro.json` 的最佳保存节点）。
快速模型与收敛大气在合成前都用扩展分子表重算 EOS 状态；在起点节点上快速模型 χ² 与节点拟合相差 0.02%。

| 星 | PZ 拟合（GALAH+ZrO） | 精修 | χ²_Korg连续/χ²_PZ | Korg + MARCS 连续 |
|---|---|---|---:|---|
| α Cet | 3521/0.60/−0.88 | 驻定 | 1.00 | 3529/0.50/−0.87 |
| α Tau | 3619/0.55/−0.96 | 达到上限 | 0.95 | 3685/1.09/−0.64 |
| ψ Phe | 3457/1.50/−0.76 | 驻定 | 1.03 | 3429/1.50/−0.62 |
| φ Aqr | 3657/1.43/−0.58 | 快慢谱一致 | 1.00 | 3622/1.25/−0.56 |
| m Vir | 3517/0.77/−0.80 | 驻定 | 0.99 | 3536/0.87/−0.65 |
| 87 Vir | 3500/1.50/−1.00 | 驻定 | 1.00 | 3372/0.72/−1.50 |
| ER Vir | 3441/1.51/−0.89 | 达到上限 | 1.02 | 3505/1.94/−0.34 |

- 相对 PZ 自有线表的拟合，χ² 降低 7–25%；连续拟合的 χ² 比从 0.77–0.89 变为 0.95–1.03。
- Korg 限制在 v4 训练盒内时，两者 Teff 相差不超过 67 K；PZ 的 [M/H] 比 Korg 低 0–0.55 dex。
- 精修 5/7 成功（1 个门槛一致、4 个驻定），α Tau、ER Vir 达到 4 个收敛大气上限；快慢谱差
  2.1×10⁻⁴–6.7×10⁻³ RMS，大于自有线表时。
- 两颗 benchmark 星仍偏冷偏贫（α Cet 3521/0.60/−0.88，α Tau 3619/0.55/−0.96）。

结果：`pz_fit_galah_zro_results.json`、`pz_fit_galah_zro/<star>/`、`pz_fit_galah_zro/launch_status.json`。

## 局限

- `pz_nodes` 臂被量化到 78 个不规则节点，节点间不做谱插值（该网格上插值误差为
  3% 量级，见 `mgiant_experimental_interface_v1`）；连续的 PZ 结果来自 PZ 自有拟合，
  受 v4 训练盒限制，其中 2 颗星的精修达到 4 个收敛大气上限。
  `mgiant_training_expansion_20260922` 的 48 个补点尚未产出合格样本。
- 同线表对照不是完全相同的物理：两边的 EOS、配分函数、分子平衡、连续谱不透明度和
  阻尼缩放仍不同；ZrO 常数取自 Korg 的数据而非 PZ 自有来源，230 条 ABO 线用单温度等效 γ
  近似，H/He 线用 PZ 原生。
  6700–6735 Å 的剩余代码差异来源未定。
- 只有一个 255 Å 窗口。它受 Korg 自带线表的 TiO 覆盖限制；其他光学或
  H 波段窗口的结论可能不同（H 波段 PZ 线表的问题见
  `mgiant_aspcap_fit_v1_closeout_20260914.md`）。
- 仅两颗星有可信参考标签；所有臂共同的冷/贫偏差来源未定（线深系统性偏浅、
  1D LTE、固定 vmic=2 km/s、[α/M]=0 与 [C/M]=0、M 巨星变星性）。第一次
  挖掘后 C 亏损会使自由 O 增多、TiO 变强，固定太阳 C/O 的模型需要更低 Teff
  才能达到同样的 TiO 强度；本轮没有检验这一项。
- 额外展宽用单一高斯 σ_extra 代表宏湍流与旋转，拟合值 2.2–4.4 km/s，
  低于 Borisov 的 vsini 10–14 km/s；所有臂处理相同。
- 未做 telluric 掩码（窗口内只有弱 H₂O 线）；未用 per-order LSF。
- MARCS 臂在 logg<3.5 使用球面大气，PZ 大气为平面；`korg_pzatm` 与
  `marcs_nodes` 的对比因此也包含几何差异。
- χ² 由模型系统误差主导，不给参数统计误差。

## 复现

项目根目录，按顺序：

```sh
.venv/bin/python experiments/mgiant_eso_bestfit_v1.py fetch
.venv/bin/python experiments/mgiant_eso_bestfit_v1.py export
.venv/bin/python experiments/mgiant_eso_bestfit_v1.py pz --device mps --workers 2
K=/Users/jdli/Project/jorg/Korg.jl-1.0.1; R=results/mgiant_eso_bestfit_v1
julia -t 3 --project=$K experiments/mgiant_eso_bestfit_v1_korg.jl --root $R --arm marcs_nodes
julia -t 3 --project=$K experiments/mgiant_eso_bestfit_v1_korg.jl --root $R --arm korg_pzatm
for k in 1 2 3; do julia -t 2 --project=$K experiments/mgiant_eso_bestfit_v1_korg.jl --root $R --arm marcs_dense --part $k --parts 3; done
.venv/bin/python experiments/mgiant_eso_bestfit_v1.py fit
.venv/bin/python experiments/mgiant_eso_bestfit_v1.py report
.venv/bin/python experiments/mgiant_eso_bestfit_v1.py continuum-sensitivity
```

本机耗时：PZ 78 节点约 2.2 h（与 Korg 进程并行、负载饱和），Korg 节点臂各约
25 min，稠密网格约 2 h，拟合约 10 min。

## 文件

`results/mgiant_eso_bestfit_v1/`：

- `targets.json`：ESO ObsCore 记录、FITS 头、GBS 与 Borisov 标签
- `observations/`：7 个 ESO Phase 3 FITS
- `node_labels.tsv`、`dense_labels.tsv`、`nodes.json`、`pz_atmospheres/`
- `pz_nodes.npz`、`korg/{marcs_nodes,korg_pzatm,marcs_dense}/`：本征谱
- `fit_results.json`：每星每臂最佳拟合、前 5 节点、全节点 χ²、RV 检查、两个节点上的同大气诊断
- `metrics.csv`、`continuum_sensitivity.json`
- `fits/*.npz`：观测、掩码与各臂最佳预测
- `figures/labels_by_arm.png`、`figures/<star>_bestfit.png`
