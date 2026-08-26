# v3 无 emulator 解析初始化器 — 进展交接 (hand-off)

> 生成于 2026-08-17（接 `analytic_initializer_v2_handoff.md` 之后）。
> 权威数字一律取自 `results/analytic_initializer/*.json`，**不要引用本文或任何
> notes 的叙述文字**（`paper/README.md` 自己警告过 notes 混了被否决的 checkpoint）。
> 详细过程见 `analytic_initializer_execution.md` 末尾六节，以及
> `entropy_closure_convergence_retest.md`（polytrope 路线的预注册重测）。
>
> **状态：四步全部完成，funnel 已跑，polytrope 重测已判决。没有进行中的任务。**

---

## 0. 先读这一条：目标被用户明确重新界定过

**这是本轮最重要的上下文。** v1/v2 熵闭合是被一个 deep p95 ≤ 0.015/0.020 dex 的
**精确性门槛**否决的。2026-08-17 用户被直接问到时确认：**那不是目标。**

真正的目标是三条：

1. 去掉 emulator
2. `(m,T)` 求解器照样稳
3. 解析解**符合物理**

(1)(2) 在 H2 上**已经达成**（首试收敛 55/60 vs production 54/60，
`h2_paired_funnel60_result.json`），当时深部误差约 0.09 dex。所以「符合物理」只能是
**结构性**的，不是「准到某个阈值」。

「符合物理」被拆成三种读法，用户选了 **1 + 3**：

| | 读法 | 状态 |
|---|---|---|
| 1 | **结构物理** — 永不吐出非物理大气 | ✅ 完成（第一步）|
| 2 | 逐项可解释 — 每个常数有物理名字 | ❌ 明确不追求（就是被否两次的熵闭合路线）|
| 3 | **无网格 / 非表格** — 深度轴是解析函数 | ✅ 完成（第二步）|

**≤600 常数是用户偏好，不是科学要求。** 实测代价见 §2。

相关记忆：`~/.claude/projects/-Users-jdli-Project-payne-zero/memory/payne-zero-analytic-initializer-goal.md`

---

## 1. 第一步（已完成）：四条物理不变量

`monotone_temperature.py` + `monotone_initializer.py` + `run_monotone_invariants.py`

H2 有两个漏洞：**858/52199 行（1.64%）温度反转**（真值全是单调的，最坏 −122 K/层），
以及**标签盒外静默外推**（Teff=12000 K 返回峰值 62543 K 且不报错）。

修法：温度改用「顶层锚点 + 每区间 ln T 增量」表示，增量取指数所以恒正 → 单调是**表示的
性质**。这和 H2 本来就用在质量上的做法是同一招（预测 log κ，积分 dm/dτ=1/κ）。

**关键的设计教训（测出来的，不是想出来的）**：拟合和表示必须分开。直接拟合增量差三倍
（p95 0.0197 → 0.0560），因为最小二乘独立平衡每个增量而剖面是它们的累积和，79 个误差
随机游走。最终做法是**在累积量上拟合，在增量空间表示**。

| | H2 | 现在 |
|---|---|---|
| κ > 0 | 成立 | 成立 |
| m 严格单调 | 52199/52199 | 52199/52199 |
| T > 0 | 52199/52199 | 52199/52199 |
| **T 严格单调** | **51341/52199** | **52199/52199** |
| **盒外标签** | **静默外推** | **拒绝** |

代价：held-out T p95 0.019717 → 0.020122（+0.00041）。

产出：`results/analytic_initializer/monotone_invariants.json`、
`monotone_profile_parameters_v1.npz`、`tests/test_analytic_initializer_monotone.py`（24 测试）

---

## 2. 第二步（已完成）：解析深度轴，公式不再是表格

`analytic_depth.py` + `compact_initializer.py` + `run_compact_frontier.py`

逐层 80 维向量 → `ln τ` 的切比雪夫级数（生产网格在 ln τ 上严格等距，步长 0.2878）。

**网格自由不但免费，而且倒赚**（held-out，对照 H2 表格版 4591 floats / 0.0201 / 0.0869）：

| 配置 | 常数 | T p95 | deep | mass p95 |
|---|---|---|---|---|
| `COMPACT_CONFIGURATION` | 589 | 0.0389 | 0.0392 | 0.1698 |
| `PARITY_CONFIGURATION` | 2407 | 0.0201 | 0.0199 | 0.0870 |
| `PHYSICAL_CONFIGURATION` | 3851 | 0.0146 | 0.0152 | 0.0597 |

**≤600 预算的代价是 1.95 倍**（两个场均衡）。

**这一步暴露了一个固定网格藏住的缺陷**：保温度单调的 floor 原来是「每区间」的 —— 那是
关于网格的陈述。同一批常数在 80 层和 791 层（同一区间）之间漂移 **1.9%**。已改成
`d lnT / d ln τ` 的**梯度 floor**（`GRADIENT_FLOOR = 1e-4`），按间距缩放。第一步的资产
已按新语义重新生成，数字不变。**资产格式已 bump 到 `..._v2`。**

**网格自由的精确程度**：底层级数在共享深度上一致到 6.0e-10；经单调投影后，原始级数本来
就单调的行一致到 2.8e-6，剩余差异（p50 4.4e-4，max 7.5e-3）全落在需要修复的行上（99%），
因为夹多少取决于凹陷解析得多细 —— 这是任何单调投影的固有性质。层间相对线性插值的偏离
p95 2.0e-4 dex（温度）/ 2.7e-3 dex（opacity），**无 Runge 振荡**。

**两个场要不同配置**：opacity 只有 1 个模时崩到 0.53 dex，2 个模回到 0.17；温度反过来
宁要标签阶数不要模。配成一样是浪费预算。

**一个被自己否掉的警报，别重提**：底层某热巨星算出 66035 K 看着像爆炸，但真值那行是
54915 K，held-out 真值最高 56165 K，表格版 H2 在同一行给 66023 K。热端低重力角落的既有
精度极限，不是基的问题。

产出：`compact_frontier.json`、`compact_profile_parameters_{parity,600}.npz`、
`tests/test_analytic_initializer_compact.py`（21 测试）

---

## 3. 表面-Hopf 混合基：假设正确，但优化了不绑定的项（不采用，勿重做）

`run_hopf_basis_probe.py` + `hopf_basis_probe.json` + `tests/test_analytic_initializer_hopf_basis.py`

用户挑战「切比雪夫有物理依据吗」→ 没有，它是通用基。但把目标反解成隐含 Hopf 函数
（`T⁴ = ¾T_eff⁴(τ+q)`）发现：**最表面一段就是教科书 Hopf 函数**（τ=0.013 处 q 中位 0.567 vs 经典 0.580），
往下 q 高于 grey（τ=0.237 处 0.719 vs 0.624）—— 那是线吸收致暖，非灰效应，深部 q 变负发散（τ=75 时 −2.4，τ=1000 时 −471），
因为对流接管。一条级数在跨接两个物理区域 —— 这才是系数谱要到第 26–28 项才衰减的原因。

**假设对了一半**：加入精确 grey 列 `¼log₁₀((τ+q)/(τ+2/3))` 使同维度表示地板改善最多 1.9×。
**端到端只值 0.5%**，因为：

| 阶段 | 累计 p95 dex | 新增 | 占比 |
|---|---|---|---|
| 深度基单独 | 0.00136 | 0.00136 | **16%** |
| + 秩-5 截断 | 0.00449 | 0.00313 | 37% |
| + 标签多项式 | 0.00856 | 0.00407 | **48%** |

深度基只占 16%，完美的深度轴也只把 p95 从 0.00856 降到约 0.00845。**判决：不采用**，生产
路径保持纯切比雪夫。按 v1/v2 惯例留作有复现器、有测试的负面参考。

---

## 4. 第三步（已完成）：物理标签坐标 —— 本轮最大收获

`physical_labels.py` + `run_label_map_probe.py` + `label_map_probe.json`

打 48% 那一项。**先做两个对照决定能不能修**：

- *可学吗* — 提高阶数一直兑现（3 次 0.00856 / 4 次 0.00698 / 5 次 0.00670，oracle 地板
  0.00449）。分辨率不够，不是饱和。
- *光滑吗* — 4 万训练行的 k-近邻**输给**三次多项式（k=10 是 0.00862，k=1 是 0.01037，
  k=40 是 0.01096；柱质量上 0.29–0.39 dex 对 0.087）。振幅函数**全局光滑** → 要换坐标，
  不是换更灵活的拟合器。

**缺失的坐标是萨哈电离度**。它是 T_eff 的 sigmoid —— 正是总阶数多项式要花很多项逼近的
形状。物理上：氢电离决定对流区起点并供给 H⁻ 所需电子。

| 标签映射 | 特征 | 项数 | T p95 | mass p95 | 关掉缺口 |
|---|---|---|---|---|---|
| 标准 3 次 | 5 | 56 | 0.00856 | 0.0870 | 0% |
| 标准 4 次 | 5 | 126 | 0.00698 | 0.0636 | 39% |
| 标准 5 次 | 5 | 252 | 0.00670 | 0.0570 | 46% |
| **物理 3 次受限** | 7 | **104** | **0.00614** | 0.0597 | **59%** |
| 物理 4 次受限 | 7 | 254 | 0.00563 | 0.0524 | 72% |

**104 项打赢 126 项的 4 次多项式**。三个失败对照各排除一种更廉价的解释：
**替换**而非添加是 0.0268（差三倍）；线性丰度 `10^[M/H]` 零收益（起作用的就是 sigmoid）；
把电离特征封在 1 次省 16 项且精度不变（证明贡献的是形状不是自由度）。

**秩的墙动了 —— 这是对 Gate B 的一个限定性修正。** Gate B 说「5→12 模只动 0.0002 dex，
低秩基饱和」，那个结论**条件于标准标签**：

| K | 标准 achieved | 物理 achieved | 7–8kK 深部（物理）|
|---|---|---|---|
| 5 | 0.00856 | 0.00614 | 0.02323 |
| 8 | 0.00821 | **0.00538** | 0.02171 |
| 16 | 0.00819 | 0.00522 | 0.02171 |

标准标签 5→16 只买 4%，物理标签买 15%。原因在 achieved−oracle 的 gap：标准标签从 0.00407
涨到 0.00638（高阶模振幅预测不出来），物理标签只从 0.00165 涨到 0.00341。
7000–8000 K 深部峰（当年否决 v1/v2 的那个 bin）从 0.0300 降到 0.0217，**改善 28%**。

**已进生产路径**：`PHYSICAL_CONFIGURATION`，3851 floats，比它替换的表格版**更小**、网格
自由、两个场都好约 30%，四条不变量在生产网格和没见过的网格上对每一行都成立。

产出：`compact_profile_parameters_physical.npz`、
`tests/test_analytic_initializer_physical_labels.py`（15 测试）

---

## 5. 第四步（已完成）：五臂 funnel —— 求解器在不在乎离线精度？

60 星配对（seed 20260817），公式臂各一次 15 轮试验，production 保留两次试验但按首试比较。
`results/analytic_initializer/multi_arm_comparison.json`。

### 5.1 收敛率：平的

| 臂 | 常数 | 离线 T p95 | 首试收敛 | Wilson 95% |
|---|---|---|---|---|
| analytic（H2 表格版） | 4580 | 0.0197 | **55/60** | [0.82, 0.96] |
| parity | 2407 | 0.0201 | 54/60 | [0.80, 0.95] |
| physical | 3851 | **0.0146** | 54/60 | [0.80, 0.95] |
| compact600 | 589 | 0.0389 | 53/60 | [0.78, 0.94] |
| production（首试） | — | — | 54/60 | [0.80, 0.95] |

离线跨 **2.7 倍**，收敛跨 **2 颗星**。所有配对 McNemar p=1.0（只有一对的不一致数够到
显著性门槛，而它的拆分是 3/4）。这是预期结果，不是发现。

### 5.2 迭代数：有检验力，而且不跟随离线精度

二元终点只用得上几个不一致对；迭代数用得上每颗同向收敛的星。Wilcoxon signed-rank，
对照表格版 H2：

| 臂 | 离线相对 H2 | 迭代均差 | p | 去混杂后 |
|---|---|---|---|---|
| compact600 | 差 1.97× | **+2.79** | 1.5e-08 | +2.86, p=4.8e-08 |
| parity | 四位数字持平 | +0.54 | 3.6e-03 | **+0.51, p=7.6e-03** |
| physical | **好 1.35×** | +0.87 | 2.4e-02 | +0.55, **p=0.10** |

**混杂**：`analytic` 臂是**无守卫的原始 H2**，其余公式臂都过单调投影。60 星里 3 颗
（13980, 19980, 28780）原始剖面需修复。去掉后 compact600/parity 效应完好，physical
掉到不显著。脚本 `run_multi_arm_comparison.py` 里有 `confound_sensitivity` 块自动做这个。

### 5.3 三条结论

**(a) ≤600 预算不是免费的。** compact600 比 H2 多 **2.9 轮**（中位 10 对 7，≈+40%），
p=5e-8，而收敛率几乎没动 —— 这正说明二元终点是评判它的错误尺子。

**(b) 萨哈坐标在运行上什么也没买到。** physical 离线好 26%、比 parity 多 1444 常数，
迭代上和 parity 区分不开（+0.33, p=0.56），相对 H2 也不更好。**离线增益是真的，但不传导。**

**(c) 离线四位数字一致的两个起点，迭代上有显著差别。** parity 复现 H2 的 p95 到
0.0201 vs 0.0197，仍慢半轮（p=0.008）。这是论文 §6.1 从新方向再现，而且是**双向的**：
指标一致也不保证盆里一致。

### 5.4 两个小观察（n 都很小）

- `compact600` 收敛了 **13265** —— 那颗其余四臂**含 production** 全部失败的星。最不准的
  臂是唯一够到它的。一颗星，是奇观不是结论。
- 那 3 颗需修复的星让 `physical` 花 **12–13 轮**（H2 6–7，parity 7–8，compact600 8）。
  萨哈臂在需要单调修复的剖面上有特异的坏表现。三颗星，是线索。

### 5.5 建议交付 `parity`（2407 常数）

网格自由、物理良构、比表格版小 1.9 倍，代价半轮迭代 —— 那是网格无关性的价钱。
physical 多花 60% 常数换不到可测量的东西；compact600 达标但赔 40% 迭代。

**必须写清的一句**：production 仍比所有公式臂快（中位 5 对 7–10，p ≤ 6e-6）。
「去掉 emulator」成立在**收敛率**上，从来不在迭代数上。

### 5.6 ⚠️ 一个静默的 wiring bug（已修，但记着它的形状）

`_solve_payload` 原来按 `payload["arm"] == "analytic"` 分派，三个新臂全落进 emulator 路径。
**它不报错** —— emulator 会收敛，输出完全正常，唯一破绽是 `trials_used` 从只允许一次
试验的臂返回 2。44 分钟结果作废。

已修：分派改按 `payload["mass"] is None`（语义而非名字）；加了运行时守卫，公式臂只要有
一颗星 `trials_used != 1` 就在**第一颗星上** SystemExit；`tests/test_analytic_initializer_multi_arm.py`
覆盖之，并**验证过对旧代码是失败的**。

---

## 6. polytrope 路线按收敛率重测（已完成，第三次否决）

完整预注册与记录：`notes/entropy_closure_convergence_retest.md`（245 行）。
起因是 §5 证明了当年否决 v1/v2 的 dex 门槛没有预测力。

**Phase 1**：把四个能工作的臂放到 v2 自己的尺子上（深窗每星 max|Δln T| 的 p95，
7000–8000 K）：H2 **0.2320**、parity 0.2294、physical 0.1758、compact600 0.2770，
而被否的 v2 是 **0.0990**、v1 是 **0.0530**。**被否的族比每个能跑的臂准 2–4 倍**，
那个 0.015 阈值比项目里任何在用的东西严 15 倍。→ 否决理由不成立，路线重开。
（限定：v1/v2 是 oracle 上界，每星 5 个自由参数 + 真值压强；四臂是零 per-star 自由度的
真实映射。）

**Phase 2 判决：否。** 60 星跑了 44 颗，收敛 16、**超时 26**，最大可达 32 < 通过线 52
—— **算术上已确定**，剩余 16 颗全收敛也翻不了盘（这是「结果已定」的提前停止，不是
「趋势难看」的提前停止）。失败极度局部化：

| Teff | 超时 |
|---|---|
| 8000–10500 | **0/14** |
| 5500–7000 | **13/13** |
| 4000–5500 | 11/14 |

**闭合只在它不起作用的地方能用**：8000 K 以上范围内几乎无对流，闭合退化成 parity 于是
全过。另外 5500–7000 K 全灭，而 v1/v2 当年全部工作锁定 7000–8000 K，从没看过这个区间。

**真正的收获是那个天花板**（对照阶梯，`entropy_convergence_retest.json`）：

| | 误差 |
|---|---|
| 真值梯度 + 真值压强 | 0.0118 |
| 真值梯度 + **预测压强** | **0.2323** |
| parity，**完全不积分** | **0.0678** |

> **在预测坐标里积分梯度，天花板比直接预测剖面的实现值还差 3.4 倍。**
> 因为积分权重是 `Δ ln P`，是预测量的**差分**，差分放大相对误差。

这对**任何** ∇ 形式的物理闭合都成立 —— polytrope、混合长 —— 只要压强是预测的。它一次性
解释了 v1/v2 的 oracle（用真值压强）为何好看而可部署版做不到。

---

## 7. 统一结论：物理该以什么形式进入

把本轮所有物理尝试排开，规律很干净：

| 物理以什么形式进入 | 结果 |
|---|---|
| **归一化**（grey/Hopf 的 `T_grey`） | ✅ 承重，一直在用 |
| **构造约束**（`dm/dτ = 1/κ` 保 m 单调） | ✅ 承重 |
| **坐标**（萨哈电离度） | ✅ 离线好 30%，关掉 59% 缺口 |
| **额外基函数**（显式 Hopf 列） | ⚠️ 假设正确但不绑定（端到端 0.5%）|
| **积分预测梯度**（polytrope / 任意 ∇） | ❌ 有硬天花板 |

**物理当坐标、当归一化、当约束时都赢；以「积分一个预测梯度」的形式进入时输。**

注意 Hopf 不是失败：`T_grey` 就是 Hopf 结构，本来就在交付的公式里承重。反解出的 q(τ)
在 τ=0.013 处是 0.567 对经典 0.580，最表面一段就是教科书 Hopf；再深 q 高于 grey
（0.719 对 0.624）是线吸收致暖，深部变负发散是对流接管。失败的只是「再加更多显式
Hopf 列」，因为深度基只占误差 16%（秩 37%、标签映射 48%）。

---

## 8. 下一步建议

1. **交付 `parity`**，并按 §5.5 把「收敛率 parity、迭代数不如 emulator」写清楚。
2. **标签映射还剩 41%**（oracle 0.00449，物理映射 0.00614）。未试方向：更多物理坐标
   （H⁻ 不透明度标度、∇_rad 判据），或把电离势 χ 也拟合而不是固定 13.598/7.6。
   但**先看 §5.3(b)** —— 离线增益不传导，所以这只在你要论文级精度时值得做。
3. **秩在物理标签下松动了**（5→16 买 15%，标准标签只买 4%），但 K=5→8 要多约 2214 floats。
   同样受 §5.3(b) 约束。
4. **补完 entropy 臂剩下 16 颗**（约 45 分钟，判决不变，只为记录完整）—— 可选。
5. **不要做**：重开 v1/v2 oracle、调那些门槛、再试 Hopf 混合基、再换一个闭合族。
   前三条有记录在案的否决；第四条要先解决 §6 那个天花板，而不是换族。
6. 唯一可能救活 polytrope 的条件已定位：**让压强不再是预测量** —— 从求解器当前状态取 P
   （solver-in-the-loop），或改成对累积量 `ln T` 直接做物理参数化而不是对梯度。
   两条都要**新预注册**。

---

## 9. 环境与约定（踩过的坑）

- 工作目录 `/Users/jdli/Project/payne-zero`，求解器用 **`.venv`**（不是 conda base）。
- **`NUMBA_THREADING_LAYER=workqueue` 必须设**，否则 macOS 上 segfault。
- 测试：`PYTHONPATH=. uv run --no-sync pytest tests/ --ignore=<7 个 solver 模块>`。
  `uv run pytest` 会解析到 anaconda 的 pytest + numba 0.58.1，所以 7 个 import
  `payne_zero_atmosphere` 的模块必然收集失败 —— **既有问题，不是你弄坏的**：
  test_bench / test_integration / test_diffatm / test_reduced_state{,_restart} /
  test_cool_star_step_test / test_grey_start_benchmark。
- 还有 **1 个既有的顺序依赖失败**：`test_entropy_closure_v2::test_no_torch_import_in_prediction_path`
  断言 `torch not in sys.modules`，但同批里 `test_reduced_state_emulator` 先导入了 torch。
  单独跑通过。**不是你弄坏的。**
- 当前离线套件：**126 passed / 2 skipped / 1 既有失败**。
- 写探针模块时让求解器 import 保持惰性（`pytest.importorskip`），这样在轻量环境里还能跑。
- **数字只从 `results/*.json` 引用**，不从 notes。
- `experiments/`、`notes/`、`tests/`、`results/analytic_initializer/` 目前**全部 untracked**。

## 10. 本轮新增文件清单

```
experiments/analytic_initializer/
  monotone_temperature.py          单调表示 + 支撑盒
  monotone_initializer.py          第一步组装（表格版 + 两个守卫）
  analytic_depth.py                切比雪夫深度基（含 label_features/degree_caps 钩子）
  compact_initializer.py           第二/三步组装（网格自由），三个配置常量
  physical_labels.py               萨哈电离度坐标 + 受限指数表
  run_monotone_invariants.py       第一步探针
  run_compact_frontier.py          第二步探针（Pareto 前沿 + 网格无关性）
  run_hopf_basis_probe.py          负面结果探针
  run_label_map_probe.py           第三步探针（含秩饱和复测）
  entropy_hybrid.py                polytrope 重测用的混合臂（三个全局常数）
  run_multi_arm_comparison.py      五臂配对比较（收敛 + 迭代两个终点 + 混杂敏感性）
  run_h2_solver_funnel.py          【已改】六个臂，全部跑过

tests/
  test_analytic_initializer_monotone.py         24
  test_analytic_initializer_compact.py          21
  test_analytic_initializer_hopf_basis.py       10
  test_analytic_initializer_physical_labels.py  15
  test_analytic_initializer_multi_arm.py        14  (含 dispatch bug 回归)

results/analytic_initializer/
  monotone_invariants.json / monotone_profile_parameters_v1.npz
  compact_frontier.json / compact_profile_parameters_{parity,600,physical}.npz
  hopf_basis_probe.json
  label_map_probe.json
  multi_arm_comparison.json
  funnel60_{physical,compact600,parity}.json{,l}   各 60 行
  funnel60_entropy.json{,l}                        44 行（判决已定，见 §6）
  entropy_convergence_retest.json
```

另：`profile_closure.py` 拆出了 `evaluate_profile_closure`（让 target 可以住在 τ 网格以外
的轴上），`predict_profile_closure` 保留为带长度校验的包装，旧调用点不变。

Corpus sha `092cf3c4...244284`，split seed `20260816`，funnel seed `20260817`。
