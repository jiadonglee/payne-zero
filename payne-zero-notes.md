# Payne Zero 学习笔记

> 基于 `payne-zero` 仓库代码整理(2026-08-04)。所有 `file.py:行号` 引用对应 commit `9c44001`。

- **第一章 从 0 合成一条光谱** —— 怎么用,以及内部到底发生了什么
- **第二章 不用 Emulator 怎么收敛** —— 经典迭代的完整物理

两章的关系:第一章的 ① 造大气 这一步,在快速路线上由神经网络一步猜出;第二章讲的是这一步在**没有神经网络**时如何用物理迭代求解。

---
---

# 第一章 从 0 合成一条光谱

先给你一句话的总览:

```
5个数字(Teff, logg, [M/H], [α/M], ξ)
   ↓ ① 造大气:算出这颗星表面往下 80 层每层多热、多密
   ↓ ② 算成分:每层里 H/Fe/Mg… 各有多少、电离到第几级、结了多少分子
   ↓ ③ 算不透明度:每个波长上,这层气体有多"挡光"
   ↓ ④ 辐射转移:让光从深处往外爬,看最后逃出来多少
一条光谱(波长 + 流量)
```

下面把每一步拆开讲。

## ① 先造一个"大气"

**什么是大气模型?** 你可以把恒星外层想成一叠**千层饼**——代码里正好是 **80 层**。你看到的光不是从某一个"表面"发出来的,而是从这 80 层里各处冒出来、层层穿透后叠加的结果。所以要算光谱,必须先知道:第 1 层多热?第 2 层气压多大?一直到第 80 层。

每层需要 6 个物理量(`source_data_files/atmosphere_emulator/README.md:15`):柱质量、温度、气压、电子密度、Rosseland 不透明度、辐射加速度。

**传统做法**很慢:先猜一个温度分布,算光怎么传,发现能量不守恒,再修正温度,再算……反复迭代到收敛,一颗星要几分钟到几小时。

**Payne Zero 的关键提速**:训练了一个神经网络(52,199 条收敛好的大气当训练集),直接**一步猜出**这 80 层的结构。这就是所谓 initializer(`payne_zero_atmosphere/warm_start.py`)。

于是你有两条路:

| 路线 | 做法 | 用在哪 |
|---|---|---|
| **快速路线** | 神经网络猜出来的大气,直接拿去合成 | 探索参数、拟合时反复算几万条谱 |
| **收敛路线** | 用神经网络的猜测当**起点**,再跑真正的物理迭代到收敛 | 论文里要报告的最终结果 |

收敛路线默认要求:至少 3 次迭代、深层温度相对变化 < 5e-4(`payne_zero_atmosphere/README.md:52`)。一次迭代在 16 核 CPU 上大约 2–5 秒。

> 这一步的物理细节见第二章。

## ② 算每层的"化学清单"

有了每层的温度和压强,就能算这层里到底有些什么东西。这一步叫**状态方程 (EOS)** 和**分子平衡**(`equation_of_state.py`、`molecular_equilibrium.py`)。

具体要回答三个问题:

- **电离到第几级?** 铁在 4000 K 主要是中性 Fe I,到 10000 K 就大量变成 Fe II 了。同一个元素,电离态不同,吸收的波长完全不同。
- **结了多少分子?** 温度低于 4000 K,CO、TiO、OH 这些分子开始大量形成(`molecular_equilibrium.py:569`),它们会在光谱上糊出成片的分子带。
- **自由电子有多少?** 这个特别重要,因为太阳型恒星最主要的连续吸收来源是 **H⁻ 负氢离子**,而 H⁻ 的多少直接取决于自由电子有多少。

这一步输出的是每层、每个元素、每个电离级的粒子数密度。

## ③ 算"不透明度":这层气体有多挡光

这是整个计算里最重的部分。**不透明度**就是"光穿过这层气体有多难"。它分两部分:

**连续不透明度**(`continuum.py`,5651 行):跟波长关系平滑,不挑波长。主要是 H⁻ 的束缚-自由和自由-自由吸收、氢的光电离、电子散射(Thomson)等。它决定了光谱的整体形状——也就是那条平滑的"底座"。

**谱线不透明度**(`atomic_lines.py`、`hydrogen_lines.py`、`molecular_lines.py`、`line_opacity.py`):这是光谱上那些**吸收线**的来源。原理是:某个原子从能级 A 跳到能级 B 需要一份精确的能量,对应一个精确的波长,那个波长上的光就被吃掉了。

每条谱线不是一根无限细的针,而是有宽度的,因为:

- **多普勒展宽**:原子在热运动,朝你跑的和背你跑的看到的波长不同(温度越高越宽);
- **微湍流 ξ**:除了热运动,气体本身还有小尺度湍动 —— 这就是那第 5 个标签的物理含义;
- **压力/碰撞展宽**:周围粒子撞它,能级被扰动,线就被抹宽了(矮星比巨星宽,因为 logg 高、密度大)。

代码里线表数据量巨大,所以做了**窗口不变量缓存**(`WindowInvariants`):跟恒星参数无关的那部分(哪些线落在你要的波长区间、线的位置和强度表)算一次存起来,后面换参数重算时直接复用。这就是为什么第一次跑要 10–20 分钟编译缓存,之后就快了。

## ④ 辐射转移:让光爬出来

现在每层有多热、有多挡光都知道了,最后一步是解**辐射转移方程**(`radiative_transfer.py`)。

白话讲:每一层都在按自己的温度发光(普朗克函数 `planck_bnu`),同时也在吸收下面传上来的光。从最深处开始往外走,一层一层累加"发射"、扣掉"吸收",走到最外层剩下的,就是逃出来被你望远镜接到的光。

直觉上有个很有用的图像:**你在某个波长上看到的,是"光学深度 ≈ 1"那一层的温度**。

- 在连续谱波长上,气体透明,你能看很深 → 看到的是深处的**热**气体 → 亮。
- 在谱线中心,气体极不透明,你只能看到很浅的一层 → 那里**冷** → 暗。

**这就是吸收线的本质**:不是"光被吃掉了",而是"在那个波长上你只能看到更冷更浅的地方"。

代码里同时解两遍:一遍带谱线(得到 `flux_total`),一遍只有连续不透明度(得到 `flux_continuum`)。两者相除就是 `normalized_flux`——归一化光谱,也就是大多数人真正要用的东西。

## ⑤ 最后:变成"望远镜看到的样子"

到这里出来的是**本征光谱**(intrinsic spectrum),即恒星真实的、无限高分辨率的光谱。真实观测还要再加三层:

1. **视向速度**:整条谱被多普勒平移;
2. **仪器展宽 (LSF)**:望远镜光谱仪本身把细节抹平;
3. **重采样**:落到观测的像素格点上。

这一步由 `fitter.ObservedSpectrumOperator` 负责,和合成是分开的。

⚠️ **一个高频误解**:`--r-grid` **不是**仪器分辨率!它是本征光谱的**采样密度**(相邻采样点的 λ/Δλ),控制你把这条真实光谱画得多细。仪器分辨率是后面单独加的。`--r-grid` 给小了会漏掉窄线的细节,给大了纯粹浪费算力。

## 实操:真的跑起来

**第一次(装环境,10–20 分钟)**

```bash
git clone https://github.com/tingyuansen/payne-zero.git
cd payne-zero
./install.sh
```

`install.sh` 会下载并校验数据文件、装 Python 包、预热 400–900 nm / R_grid=20000 的标准缓存。慢的部分全在这里,一次性的。

**合成一条太阳光谱(最快路线,几秒)**

```bash
payne-zero-synthesis \
  --effective-temperature 5777 \
  --log-surface-gravity 4.44 \
  --metallicity 0.0 \
  --alpha-enhancement 0.0 \
  --microturbulence-km-s 1.0 \
  --wl-start-nm 500 --wl-end-nm 510 --r-grid 20000 \
  --out sun.npz
```

**看结果**

```python
import numpy as np
d = np.load("sun.npz")
print(d.files)          # wavelength_nm, flux_total, flux_continuum, normalized_flux, seconds
# 直接画归一化谱:
# plt.plot(d["wavelength_nm"], d["normalized_flux"])
```

`normalized_flux` 就是你要的那条带吸收线的、在 1.0 附近起伏的光谱。

**如果要收敛的物理大气(慢一些,但严谨)**

```bash
# 第一步:迭代求解大气(CPU,几秒到几十秒)
payne-zero-atmosphere \
  --effective-temperature 5777 --log-surface-gravity 4.44 \
  --out runs/sun

# 第二步:拿这个大气去合成(GPU)
payne-zero-synthesis runs/sun/payne_zero_structured_atmosphere.npz \
  --wl-start-nm 500 --wl-end-nm 510 --r-grid 20000 \
  --out runs/sun/spectrum.npz
```

两条路线用的是**完全相同的合成内核**,唯一区别是大气是"神经网络猜的"还是"物理迭代收敛的"。

## 几个容易踩的坑

- **参数范围**:Teff 必须在 4000–10500 K,logg 0.7–5.3,[M/H] −2.5–+0.5,ξ 0.5–4.0 km/s。超出范围快速路线直接报错。
- **换波长窗口会变慢**:换到没缓存过的区间,第一次要重建线表。可以先用 `python -m payne_zero_synthesis.prewarm --wavelength-start-nm ...` 预热,别把这个一次性开销算进你的计时里。
- **设备选择**:自动按 CUDA → Apple Metal → CPU 挑。大气迭代是 CPU 的活(顺序迭代,GPU 帮不上),合成是 GPU 的活(波长方向天然并行)。
- **别把 initializer 的大气当成收敛结果报告**:代码专门给它打了 `atmosphere_product_role: "learned_initializer_prediction"` 的标记来防止混淆。

---
---

# 第二章 不用 Emulator 怎么收敛:经典迭代的完整物理

Emulator 只是替你**猜了一个起点**。真正的物理收敛过程它一点都没省——`payne-zero-atmosphere` 跑的就是完整的经典迭代。把 emulator 拿掉,你要做的就是自己提供起点,然后跑同一套循环。

## 核心问题:什么叫"收敛"?

一个模型大气要同时满足**两个守恒条件**,在**每一层**都成立:

```
① 力学平衡:  dP_total/dm = g              (气体不塌不飞)
② 能量守恒:  ∫F_ν dν + F_conv = σT_eff⁴   (每层流进=流出,不囤积能量)
```

难点全在 ②。你随便给一个 T(τ) 分布,算出来的流量一定不是常数——深处多了浅处少了。**收敛就是反复修正 T(τ),直到流量在所有深度都等于 σT_eff⁴。**

条件 ① 反而很简单,因为代码用**柱质量 m** 当深度坐标而不是几何高度。这样静力平衡方程可以直接积分成代数式:

```python
# hydrostatic.py:25
pressure = g * column_mass - radiation_pressure - turbulent_pressure
```

`P_gas = g·m − P_rad − P_turb`,一行就解决了。这是选 m 当自变量的最大好处。

## 迭代循环的骨架

对应 `runner.py:1645` 的主循环,每一轮做这五件事:

```
给定 T(m) 和 P(m)
  ↓ (A) 状态方程 → ρ, n_e, 各粒子布居
  ↓ (B) 不透明度 → κ_ν(每层每频率), 以及 Rosseland 平均 κ_R 和 τ_R
  ↓ (C) 辐射转移 → J_ν, H_ν(每层每频率)
  ↓ (D) 对流 → F_conv(混合长理论)
  ↓ (E) 温度修正 → ΔT(m),更新 T
  ↓ (F) 静力平衡 → 更新 P
回到开头,直到 ΔT 足够小
```

下面逐个说关键方程。

## (A) 状态方程:Saha + Boltzmann + 分子平衡

`equation_of_state.py`(1954 行)、`molecular_equilibrium.py`

给定 T 和 P,要算出这层里有多少自由电子、每个元素电离到第几级、结了多少分子。

**Saha 方程**(相邻电离级之间的分配):

```
N_{i+1}·n_e / N_i = 2 · (U_{i+1}/U_i) · (2πm_e kT/h²)^{3/2} · exp(−χ_i/kT)
```

**Boltzmann 分布**(同一电离级内部各能级的分配):

```
n_j/N = (g_j/U(T)) · exp(−E_j/kT)
```

**分子解离平衡**:

```
n_A · n_B / n_AB = K_AB(T)
```

**电荷守恒**:`n_e = Σ_i Z_i · n_i`

⚠️ 这里有个**内层不动点**:Saha 方程需要 n_e 才能算,但 n_e 又是所有电离结果加起来的。所以每一层内部要先迭代到自洽(代码里 `eos_tolerance=1e-5`)。这个内循环是整个 EOS 的主要开销。

对低温星还要额外注意:H⁻ 的浓度对 n_e 极其敏感,而 H⁻ 又是主要的连续不透明度来源,所以 EOS 不收敛的话后面全错。

## (B) 不透明度与 Rosseland 平均

**单色不透明度** κ_ν = 连续(H⁻、氢光电离、电子散射…)+ 谱线(原子、氢、分子)。这部分是最贵的:`continuum_opacity.py` 6197 行 + `line_opacity.py` 3395 行。

**Rosseland 平均不透明度**——这是温度修正里的"主力坐标":

```
1/κ_R = ∫ (1/κ_ν)·(∂B_ν/∂T) dν  /  ∫ (∂B_ν/∂T) dν
```

代码就是照着这个写的(`rosseland_mean.py`):mode 2 累加分子 `(∂B/∂T)/κ_ν · w_ν`,mode 3 用 `4σT³/π` 去除:

```python
# rosseland_mean.py:63
accumulator[:] = 4.0 * (σ/π) * temperature**3 / accumulator
optical_depth = integrate_on_depth_grid(column_mass, accumulator, ...)
```

注意 Rosseland 平均是**调和平均**(权重在 1/κ 上),物理上是因为在光学厚区能量走"最容易漏出去"的频率,所以透明的窗口主导。这和 Planck 平均(算数平均)刚好相反,别搞混。

然后 `τ_R = ∫ κ_R dm`。

**频率取样**:温度修正用的不是几十万个波长点,而是约 **343 个连续参考频率**(`continuum_opacity.py:1212`)。这是 ATLAS 风格的 opacity sampling——修正温度不需要分辨每条线,只需要正确的频率积分流量。

## (C) 辐射转移

`radiative_transfer.py` + `transfer_kernels.py`

```
μ · dI_ν/dτ_ν = I_ν − S_ν
```

源函数带散射,这是麻烦的根源:

```
S_ν = (1 − σ_ν/χ_ν)·B_ν(T)  +  (σ_ν/χ_ν)·J_ν
                                        ↑
                        J 是 I 的角度积分,而 I 又依赖 S ⇒ 非局部积分方程
```

解法是 Feautrier 型的二阶差分,把它变成三对角线性系统一次解出来。输出每层每频率的:

- **J_ν**(平均强度)——判断是否辐射平衡
- **H_ν**(Eddington 流量)——判断是否能量守恒

代码内部一律用 Eddington flux H,公开接口才转成 `F = 4πH`。

## (D) 对流:混合长理论

`convection.py`

深层辐射搬不动能量时,气体会翻滚。判据是 **Schwarzschild 判据**:

```
∇_rad > ∇_ad   ⇒  对流不稳定
其中 ∇ ≡ dlnT/dlnP,  ∇_ad = (∂lnT/∂lnP)_S
```

对流流量(混合长近似,`l = α·H_p`,默认 α=1):

```
F_conv ~ ρ · c_p · T · v_conv · (∇ − ∇_ad)
v_conv ~ (l/2)·√( −(P/ρ)·(∂lnρ/∂lnT)_P · (∇−∇_ad) · g/H_p )
```

代码里 `c_p`、`∇_ad`、`(∂lnρ/∂lnT)_P` 这些热力学导数**不是用理想气体公式,而是对 EOS 做数值微分**得到的(`compute_convection_finite_difference_samples`,`runner.py:374`)——因为电离区里 c_p 会剧烈变化,理想气体假设会错得离谱。没有有限差分数据时才退回理想气体路径。

对流一旦开启,能量守恒条件就变成 `F_rad + F_conv = σT_eff⁴`,温度修正必须把 F_conv 对 T 的响应也算进去,否则会震荡。

## (E) 温度修正 —— 收敛的核心

这是 `temperature_correction.py` 全部 931 行在干的事,也是整个求解器最精妙的部分。

**朴素做法为什么不行:** 最自然的想法是 Λ 迭代——用当前 T 算 J,用 J 更新 S,再算 J……但这在光学厚区**会卡死**。原因是 Λ 算符在 τ≫1 时的信息传播距离只有约一个平均自由程,每次迭代温度只改一丁点,几百次都收敛不了(著名的 Λ-iteration stagnation)。

**代码的做法是把三种修正加起来**,各管一段:

### ① 流量误差修正(Avrett–Krook 型)——管光学厚的深层

直接由"流量差多少"驱动:

```python
# temperature_correction.py:674
integrated_flux_error = 积分因子 · (H_rad + H_conv − H_target) / (∂H/∂T 型分母)
```

其中 `H_target = σT_eff⁴/(4π)`(代码里写成 `5.6697e-5/12.5664 * Teff⁴`,`runner.py:1746`)。

然后沿 τ_R 积分,再转成温度步长:

```
ΔT_flux = −Δτ · (dT/dm) / κ_R
```

物理直觉:如果这一层流量偏小,说明它太不透明或太冷,就把温度标度沿光学深度平移一点。这一步在深层收敛极快,因为深层本来就是扩散近似,流量对 T 的响应几乎是局部的。

### ② Λ 型修正(Unsöld–Lucy 风格)——管光学薄的表层

由辐射平衡的**残差**驱动:

```
残差 = ∫ κ_ν (J_ν − S_ν) dν      (辐射平衡要求它 = 0)
ΔT_Λ = −残差 / (对角 Λ 项的 ∂/∂T)
```

关键在那个分母 `diagonal_lambda_accumulator`——它是 Λ 算符的对角元乘以 `∂B/∂T`,相当于做了**算符分裂 / 加速 Λ 迭代 (ALI)**。除掉对角元之后,原本卡死的 Λ 迭代就恢复了正常的收敛速度。

代码里有一个很明确的开关(`temperature_correction.py:734`):

```python
if not (convective_ratio < 1.0e-5 and rosseland_optical_depth < 1.0):
    lambda_temperature_derivative[layer_index] = 0.0
```

**只在 τ_R < 1 且没有对流的层用 Λ 项**——正好是流量修正失效、Λ 修正有效的区域。两种方法的分工是按光学深度切开的。

### ③ 表面修正

最外层单独处理,因为那里 τ→0,任何积分型修正都没意义:

```python
# temperature_correction.py:750
surface_step = (H_target − H[0]) / H_target * 0.25 * T[0]
```

### 阻尼(非常重要)

每次修正被硬性限制:

```python
# temperature_correction.py:709
maximum_temperature_step = effective_temperature / 25.0
```

太阳就是每轮最多改 231 K。**不加这个限制,迭代几乎一定会震荡发散**——因为不透明度对温度是强非线性的(H⁻ 的 κ 大致 ∝ T^10 量级),一步走大了下一轮会过冲得更厉害。

另外还对对流流量做了 [0.25, 0.5, 0.25] 的三点平滑(`temperature_correction.py:528`),抑制对流边界处的层间锯齿。

## 收敛判据:代码里的实际数字

`convergence.py`:

```python
# 判据主体:深层温度相对变化
start = 39; stop = layers - 5          # 80 层里取第 39~75 层
max|ΔT/T| < 5e-4                       # 生产阈值
```

为什么**只看深层**?表层(0~38)对温度极不敏感、噪声大,而且强线的线心会让上层缓慢弛豫;深层温度稳了才说明流量真的守恒了。最底下 5 层排除是因为边界条件影响。

另外还有全层判据 `max_normalized_column_delta` 作为可选补充,专门抓"深层稳了但表层还在慢慢漂"的情况。

生产设置还要求:**至少 3 次迭代**、**至少 1 次连续满足**、每个初值最多 15 次迭代、最多 2 个初值试探(`--max-trials 2`)。

## 不用 Emulator,起点从哪来?

经典答案是**灰大气解析解**。假设 κ 与频率无关,严格解是:

```
T⁴(τ) = (3/4)·T_eff⁴ · [τ + q(τ)]
```

`q(τ)` 是 Hopf 函数,从 q(0)=1/√3≈0.577 单调升到 q(∞)≈0.710。Eddington 近似直接取 q=2/3:

```
T(τ) = T_eff · [ (3/4)(τ + 2/3) ]^{1/4}
```

有了 T(τ) 之后,压强由

```
dP/dτ = g/κ_R(T,P)
```

从表面往里积分(每步要调一次 EOS 来更新 κ_R)。这就是完整的起手式——**用 20 行代码就能生成一个能用的初值**,只是收敛需要的迭代次数比 emulator 起步多一些(典型 15~30 轮 vs. 3~5 轮)。

在这个仓库里可以直接接进去,`AtmosphereInput.initial_atmosphere`(`config.py:15`)接受任意 `ModelAtmosphere`:

```python
from payne_zero_atmosphere.atmosphere_io import parse_atmosphere_deck
from payne_zero_atmosphere.config import AtmosphereConfig, AtmosphereInput

my_start = parse_atmosphere_deck(my_gray_deck_text, source="<gray>")
config = AtmosphereConfig(inputs=AtmosphereInput(initial_atmosphere=my_start, ...))
```

CLI 走的是 emulator 分支(`cli.py:357`),但 Python 层完全没有强制要求 emulator。同理,你也可以直接喂一个 Kurucz/MARCS 的 deck 当起点。

## 一句话总结

| 方程 | 决定什么 | 代码 |
|---|---|---|
| `dP/dm = g` | 压强结构 | `hydrostatic.py` |
| Saha + Boltzmann + 分子平衡 | 每层有什么粒子 | `equation_of_state.py` |
| `κ_ν`,Rosseland 平均 | 光有多难穿过 | `continuum_opacity.py`, `rosseland_mean.py` |
| `μ dI/dτ = I − S` | 辐射场 J, H | `radiative_transfer.py` |
| `∇_rad > ∇_ad`,混合长 | 对流流量 | `convection.py` |
| **`∫F_ν dν + F_conv = σT_eff⁴`** | **收敛的目标本身** | `temperature_correction.py` |

Emulator 替代的**只有初值猜测**这一件事,上面六个方程一个都跑不掉。真正的技术难点不是写下这些方程,而是**温度修正的构造**——如何在光学厚区用流量误差、在光学薄区用加速 Λ 迭代、并用步长上限和平滑压住非线性震荡。这是 ATLAS 系列几十年积累出来的工程经验,也是 `temperature_correction.py` 里那些看着很"魔法"的常数的来历。
