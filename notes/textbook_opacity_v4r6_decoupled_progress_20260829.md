# v4r6 decoupled 工作进展快照

日期：2026-08-29（约 12:30 CEST）

这是一份工作存档，不是预注册，也不是改写已冻结判决。15 轮 development-60 的
`FAIL_STOP_DEVELOPMENT` 仍是该候选在冻结门上的正式结果。v4r6 离线
`FAIL_STOP` 未重开。生产求解器、默认初始化器、光谱门、密封 holdout、H⁻
实现级修复都没有动。

候选名：`v4r6_decoupled_mgrey_tconv_v1`

## 当前状态（一句话）

解耦种子（灰体柱质量 + 对流 \(T\)、对流后不重积分 \(m\)）在 15 轮门上
**未通过**；把迭代上限放到 60 后，同一批 60 星从 37/60 升到 **54/60**，
说明 15 轮上限挡住了晚收敛。剩下 6 颗全是墙钟超时，正在 garching 上用
100 轮 / 3600 秒重解；已写完 2/6，两颗仍是超时。

## 候选构造

```text
T_grey     = Eddington
m_grey     = RK4 积分 dm/dtau = 1 / kappa_v4r6(T_grey, g m)，8 子步
P_grey     = g m_grey
T_conv     = 现有 Saha-aware 绝热替换，作用在 (T_grey, P_grey)
m_seed     = m_grey          ← 对流后不重积分
kappa_seed = kappa_v4r6(T_conv, P_grey)
```

种子审计（WP3）已通过 `PASS_STRUCTURAL`：

- \(m\) 与灰体逐位相同，\(T\) 与对流逐位相同
- \(\kappa\) 在 \((T_\mathrm{conv}, P_\mathrm{grey})\) 上重算，最大相对残差 0
- 拟合参数 0；冷星深层 \(\tau>10\) 的 \(|\Delta\log_{10} m|\) 中位 0.094 dex
  （对流臂 0.719 dex）

## 已冻结的求解结果

同一 paper development-60（60 星）。灰体 / 对流对照是 15 轮、900 秒，没有用
60 轮重跑。

| 切分 | 对流 15 轮 | 灰体 15 轮 | 解耦 15 轮 | 解耦 60 轮 |
|---|---:|---:|---:|---:|
| 全部 | 20/60 | 37/60 | 37/60 | **54/60** |
| 冷星 \(T_\mathrm{eff}<7500\,\mathrm{K}\) | 0/27 | 6/27 | 12/27 | **24/27** |
| 热星 | 20/33 | 31/33 | 25/33 | **30/33** |
| 冷矮星 | 0/11 | 2/11 | 8/11 | 10/11 |
| 冷巨星 | 0/16 | 4/16 | 4/16 | **14/16** |
| 超时 | 3 | 3 | 5 | 6 |

### 15 轮解耦：`FAIL_STOP_DEVELOPMENT`

机器门（全部强制，见后）：冷星 12/27 过线；热星 25/33、合计 37/60、相对灰体
丢掉 10 颗、净增益 0、超时 5，均失败。相对对流净 +17、对流成功星一颗没丢。

因果（15 轮）：保留灰体质量后，现有对流 \(T\) 对冷星有用，对热星有害。
`H1` 作为合取被拒绝。

### 60 轮解耦：`ITER60_DIAGNOSTIC_COMPLETE`

`authorizes_fresh_open = false`。相对冻结的 15 轮解耦臂：捞回 17、丢掉 0。
18 颗用满 15 轮未收敛的星里，17 颗在第 16–29 轮进盆（中位约 19，均值 20.8）。
1 颗（`33053`）变成 900 秒超时。原来 5 颗墙钟超时全部还在。

结论：那 17 颗的死结是 **15 轮上限**，不是解耦种子本身。灰体 / 对流未用
60 轮重跑，因此不能拿 54/60 去改写 15 轮继续门，也不能开 fresh-open 120。

## 正在跑：6 颗残差，100 轮 / 3600 秒

宿主：`astronode-garching`，PID `1535114`，约 02:40 已过。
Runner：`experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_dev60_iter100_residual.py`

只解这 6 颗（= 60 轮超时集），从同一解耦种子重开，不是中途续算：

| 序号 | 标签 | 15 轮 | 60 轮 | 100 轮（截至本快照） |
|---|---|---|---|---|
| 6152 | 热巨星，8625 K，log g 0.87 | timeout | timeout | **timeout，3600.1 s，iters=None** |
| 33051 | 冷巨星，7306 K，log g 3.37 | timeout | timeout | **timeout，3600.1 s，iters=None** |
| 33053 | 热巨星，7770 K，log g 2.77 | not_converged | timeout | 进行中 |
| 44167 | 冷巨星，4050 K，log g 2.10 | timeout | timeout | 排队 |
| 46124 | 冷矮星，7473 K，log g 3.71 | timeout | timeout | 排队 |
| 48708 | 热巨星，8943 K，log g 1.26 | timeout | timeout | 排队 |

前两颗在 3600 秒内仍走不完一轮可记录的迭代。最坏还要约 4 小时。
产物尚未写 JSON，只有 JSONL 两行。

## 权威产物与哈希

| 产物 | SHA-256 |
|---|---|
| 解耦 source manifest | `ebc932f2402d15a936cd1a96465c15262f4f5197ad3636794fe24245db7f152c` |
| 样本 `convergence_metrics_learned_monotone.json` | `5e0238098f5811de7738d6e8fcf5b9eb5d94fe85a8fa9505ad3513769179e27e` |
| 种子审计 | `4d7179fce0af46889cf753dc462b28bcd66436d0efcec274bf653e5ed533e2be` |
| 对流 15 轮 JSON | `c0c08c9727e522916085941bc5dcb40a96d67fea05852f8d88ddb4cae4cdd3e5` |
| 灰体 15 轮 JSON | `caeee639e37600952be8439d259bdb99f68d992bf9d4c2a50749530f68bf015a` |
| 解耦 15 轮 JSON | `81211e9e61cf9e2ab39a517d3bc4c455a9a1fd702a9a92db35d70abd7609afb0` |
| 解耦 15 轮 JSONL | `bc3ec0583cd45cf5566ffd4ae8cd37d63bb96ff6bf86630cbd78a60948966bf3` |
| 解耦 60 轮 JSON | `9926fd75ba1ba948407a37c3886803e177f5cfe033d8754e6e2f64f42c7d595f` |
| 解耦 60 轮 JSONL | `7c0041c649d35ce1acd585ddc07f0fdb5bfc924ffb128d0c85c2d2d5960215c5` |

离线 v4r6 `FAIL_STOP` 仍在
`results/analytic_initializer/textbook_opacity_v4r6_offline_validation_20260828.json`。

## 笔记与代码入口

- 工作计划：`notes/textbook_opacity_v4r6_decoupled_mgrey_tconv_workplan_20260828.md`
- 15 轮预注册 / closeout：
  `notes/textbook_opacity_v4r6_decoupled_dev60_preregistration_20260828.md`，
  `notes/textbook_opacity_v4r6_decoupled_dev60_closeout_20260828.md`
- 60 轮预注册（含 post-run）：
  `notes/textbook_opacity_v4r6_decoupled_dev60_iter60_preregistration_20260829.md`
- 100 轮残差预注册：
  `notes/textbook_opacity_v4r6_decoupled_dev60_iter100_residual_preregistration_20260829.md`
- 种子：`experiments/analytic_initializer/textbook_opacity.py`
  （`build_textbook_reduced_state_v4r6_decoupled`）
- 漏斗：`experiments/analytic_initializer/run_h2_solver_funnel.py`
  （`--iterations` 默认仍 15）
- 15 / 60 / 100 轮 driver：
  `run_textbook_opacity_v4r6_decoupled_dev60.py`，
  `run_textbook_opacity_v4r6_decoupled_dev60_iter60.py`，
  `run_textbook_opacity_v4r6_decoupled_dev60_iter100_residual.py`
- 冻结门：`experiments/analytic_initializer/textbook_opacity_v4r6_decoupled_gates.py`

远程：`astronode-garching`，
`/home/jdli/xiasangju/jdli/payne-zero`（与
`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero` 同一树），
`.venv-linux/bin/python`。

## 明确未授权、不要做的事

- 不要用 60 轮分数把 15 轮门改成 `PASS_TO_FRESH_OPEN`
- 不要开 fresh-open 120、耦合 ODE、光谱门、密封 holdout
- 不要把解耦种子或纯灰体 \(T\) 注册成生产默认
- 不要改写已落盘的 15 轮 / 60 轮 JSON
- H⁻ 实现级修复仍是另一条线，本候选没有授权它

## 残差跑完之后该记什么

100 轮 JSON 落盘后，只把 6 颗残差的 recovered / still-timeout 记进
`iter100` 预注册的 post-run。不要回写 15 轮 closeout。
前两颗已在 3600 秒仍 `iters=None`：若后四颗同类，残差失败就是卡住的单步 /
墙钟，不是迭代上限。
