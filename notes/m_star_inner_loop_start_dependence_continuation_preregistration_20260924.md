# 起点依赖的续算检验预注册(2026-09-24)

对象:`notes/m_star_inner_loop_validation_closeout_20260924.md` 中候选起点无关性不成立的
三点(g4.50 m−1.0 3800 K、g4.75 3850 K、g4.75 4000 K)。那里两种解读都与数据相容:两个
起点收敛到不同的不动点;或收敛到同一不动点,停止判据在离它不同距离处触发。本检验
区分这两者。它是诊断,不重判验证的 E3。

## 续算

- 6 条求解:每点候选的主求解产品与参照起点求解产品,各从其 (m, T) 经
  `_reconstruct_from_mt` 重建(与自重启相同),候选配置(验证驱动的 `ARMS['candidate']`),
  pchip 包。
- 停止判据关闭,固定 10 轮(与 `m_star_iteration_tomography_v1` 的
  `continuation_from_terminal_mt_stop_disabled` 相同);每轮用 tomography 钩子记录。10 轮
  后用 runner 的产品写出函数(`save_product_structured_atmosphere`)写出终态产品。

## 测量

- 差距 g(k):第 k 轮后两条续算的 T 相对差 p95 与 Δlog m p95(标准网格,钩子记录),
  k = 1…10;g(0) 为原两个产品的差。
- 10 轮后两个续算产品的路径一致性(T p95 ≤ 3e-3、Δlog m p95 ≤ 7.7e-3 dex)与 TiO
  (665–667 nm,R = 20000,三项 ≤ 5e-3),与验证相同。
- 每条续算每轮的全层相对温度变化与通量误差 p95;每条续算终态相对其起始产品的偏移。

## 判定(每点,不因结果调整)

- 收拢:10 轮后路径一致且 TiO 过线。
- 未收拢:否则。另报 g(10) 与 g(0) 的比值:< 1 记为"仍在缩小",≥ 1 记为"未缩小"。

| 结果 | 解读与下一步 |
|---|---|
| 三点都收拢 | 失败与停止判据的精度相容;下一步预注册候选的更严停止判据(连续多轮或通量残差条件),在验证集上重估起点无关性;本次验证的 E3 仍记为不成立 |
| 某点未收拢且未缩小 | 该点两个起点落在不同的稳定状态;逐层诊断该点的顶部 column mass 与 TiO 形成区 |
| 某点未收拢但仍在缩小 | 10 轮不足以判定;报告缩小速率 |

## 运行

驱动 `experiments/reduced_state_emulator/m_star_inner_loop_start_dependence_continuation_20260924.py`,
Garching Node-08 单线程,6 个进程(约 20 GB/进程),全部结束后运行一次配对步骤;产物
在 `results/m_star_inner_loop_validation_20260924/continuation/`。

## 范围

三点,候选一臂;续算经 (m, T) 重建,不是原求解状态的逐位延续。生产默认、质量门与已注
册阈值不因本结果改变。
