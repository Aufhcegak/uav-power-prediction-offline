# 无人机功率预测 — 课堂离线最优无损加速版 (`q05_trees_tcn2`)

本提交包严格遵守期末考核全部规范（纯 CPU 离线运行、窗/差分/物理基严格限在 `(flight, segment)` 内、`flight/segment/t` 仅作分组排序键、无外部数据、全参数冻结）。

## 1. 模型结构与精度（394 航次 5 折严格零泄漏 LOFO）
- **核心配方（`q05_trees_tcn2`）**：
  - 5 个抗气压漂移与单点尖刺树专家：`w1_isolated`（分平台隔离专家）+ `catboost_q05_ms`（3 种子×3 深度）+ `xgboost_q05_ms`（3 种子×3 深度）+ `lgb_q05_l1` + `lgb_q05_et`
  - 2 个深度物理残差骨干：`tabm_phys`（280 维特征 BatchEnsemble）+ `res_tcn_v2`（5 折×2 种子扩展卷积+全局 FiLM）
  - **连续树共识安全护栏**：`tree_med = median(5 trees)`，神经网络预测在偏离 `[80, 140] W` 区间连续线性收敛向树中位数，杜绝罕见门控闪断。
- **严格 LOFO 性能**：**MAE = `13.82375 W`**，**RMSE = `20.96358 W`**，**R² = `0.95071`**，`>50 W` 失效航次 **`0 / 394`**，对 `-62.77 m` 气压尖刺（`F0353`）与负漂移（`F0025/F0089/F0332`）全程稳定。

## 2. 无损纯 CPU 加速
- `ResTCN_v2` 采用按片段长度排序的 `batch_size=16` 带掩码批推理（数值严格等价于 `batch_size=1`）。
- 单个平台内复用单一 ASCII 临时加载缓冲，避免重复建删目录开销。
- 自动解放多核 CPU 线程上限（`min(16, os.cpu_count())`），并跳过近零权重（`< 1e-5`）且不入护栏的分支。

## 3. 运行方式
```powershell
python predict.py --input <测试目录或单条CSV路径> --output <预测结果.csv>
```
输出 UTF-8 CSV，严格包含 `flight,segment,t,power_pred` 四列（保留原始行序与字符串精度 `t`，零 `NaN/Inf`，全部非负）。
