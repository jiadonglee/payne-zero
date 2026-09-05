# M-star 收敛：四个缺格定向续算

## 核心判断

下一步不要继续盲目增加迭代次数，也不要放宽收敛标准。四个缺格不是同一种问题：

- 富金属巨星 3200 K：表层压力失效，直接初值崩溃。
- 贫金属矮星 3500/3200 K：温度修正和通量残差发散。
- 富金属矮星 3200 K：最终 atmosphere 定点格式化失败。

最值得尝试的是：从同一 `logg/[M/H]/vmic` 轨道上最近的、已经完整通过 eligibility 的模型逐步降温。

## 计算方案

新增独立实验 `m_star_four_gap_continuation_v1`，不覆盖 v1r2，也不修改尚未运行的 v1r3。

固定起点：

| 目标 | 起点 |
|---|---|
| 3200 K, logg=1.5, [M/H]=+0.5, vmic=2 | 3500 K 合格模型的 restart atmosphere |
| 3500 K, logg=4.5, [M/H]=-1.0, vmic=1 | 3800 K 合格模型的 restart atmosphere |
| 3200 K, logg=4.5, [M/H]=-1.0, vmic=1 | 新得到且完整合格的 3500 K restart atmosphere |
| 3200 K, logg=5.0, [M/H]=+0.5, vmic=1 | 3400 K 合格模型的 restart atmosphere |

每条轨道采用确定性的 adaptive continuation：

1. 先直接跨越剩余温差。
2. 失败后从最后一个成功节点重试，步长依次减半。
3. 最小步长固定为 25 K；25 K 仍失败就停止该路径。
4. 首先使用完整 atmosphere carry。
5. 如果目标模型没有通过完整 eligibility，则从原始合格起点重新运行一次 `(m,T)` rematerialized 路径。
6. 不扫描 damping，不修改物理残差阈值，不用非合格中间模型作为下一条轨道的正式起点。

三条独立轨道先在 Garching 同时运行：富金属巨星 3200、富金属矮星 3200、贫金属矮星 3500。只有贫金属矮星 3500 完整合格后，才继续算其 3200 K。

## 收敛与验收

所有步骤保持 v1r2 的设置：最多 60 次迭代、全层温度残差 `≤5×10⁻⁴`。

中间节点只要求：

- solver 收敛；
- 80 层 atmosphere 全部有限；
- 压力、密度和深度结构满足现有基本物理检查。

最终目标必须通过原有全部 eligibility，才算真正补格：

- primary 和严格 self-restart 均收敛；
- 两次 atmosphere 均有限且物理有效；
- 两次均通过原始 flux gate；
- primary/restart 通过现有 path-consistency 检查。

不能把“solver 成功”单独记为成功，也不由成功率推断 MARCS 或 Payne-Zero 谁更稳定。

## 格式化故障处理

富金属矮星 3200 K 先走 continuation。若仍出现 `No layer rows parsed from READ DECK6`：

- 保存失败前第一行定点格式文本；
- 检查每一列在格式化前是否有限、是否超出固定列宽；
- 若输入本身非有限或溢出，判为 atmosphere/solver 路径失败；
- 只有在全部输入正常、单纯 formatter-parser roundtrip 失败时，才做最小解析修复并加一个复现测试。

不裁剪异常值，也不改变 ATLAS 固定列格式。

## 输出与完成标准

结果写入 `results/m_star_four_gap_continuation_v1/`，每个目标保存路线、步长、primary/restart atmosphere、残差和最终 eligibility。

- 4/4 完整合格：重新生成三套光谱的 12 格论文图，并更新 Markdown 汇总。
- 部分合格：保留缺失面板，逐格报告失败发生在哪个物理残差或 eligibility gate。
- 本轮不扩大到完整 M-star grid，也不因为失败改阈值。

不新增公共 API；只增加一个针对四个目标的实验驱动脚本，直接复用 v1r2 的验收函数和现有 continuation 实现。
