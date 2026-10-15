# 本科生研究任务书：两量子比特纠缠性质学习

**研究目标：** 在有限次数的可信局域 Pauli 测量下，明确区分“机器学习猜测的纠缠值”和“测量数据可以数学保证的纠缠下界”。这不是让神经网络替代全部量子信息理论，而是学习怎样提出可证伪的科研问题。

## 第一次上手（约半天）

1. 阅读 `README.md`，安装 `requirements.txt`，运行 `python -m pytest -q`。请在自己的电脑记录 Python、NumPy、PyTorch 与 SciPy 版本。
2. 运行 `python run.py --quick --sdp-engine slsqp`，查看 `results/metrics.csv`、`results/certificate_summary.csv` 与 `figures/`。
3. 用笔算证明 `docs/METHODS.md` 中的两态反例，尤其是为什么 `XX`、`ZZ` **完整四结果概率**相同，以及为什么加测 `YY` 就能区分。
4. 分别解释实验中的 `nn_negativity`（点预测）、`ls_negativity`（对重建状态计算）、`sdp_numerical_lb`（数值近似的最小兼容纠缠）、`certified_combined_lb`（有限抽样下有数学保证的保守下界）。这四个数**不能都叫作真实纠缠估计**。

## 实验设置要点

- `full`：九种设置 XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ；`partial`：只有 XX 和 ZZ。
- `shots_per_setting=1024` 时，两组分别消耗 **9216** 和 **2048** 份量子态。严禁据此说“完整测量比不完整测量效率更高”；必须先统一总预算。
- IID 训练/测试：Wishart 混合态+可分态；OOD：Bell 各向同性混合态+随机旋转的近纯态。训练与测试状态不可混用，模型超参数不能根据 OOD 测试成绩挑选。
- 输出中 `alpha=0.05` 意味着**对每个固定待测量的状态**，在假定各设置内 iid 抽样和 Pauli 轴已正确校准时，下界不超过真实 Negativity 的概率至少 95%。这**不是设备无关（DI）**安全性。

## 建议的第一篇可写结果（仅作为探索课题）

首先让学生尝试“在固定总测量预算中，如何选 Pauli 测量设置，使可证明的 Negativity 下界最强”。必须比较数学上有效的见证下界、SDP 基准、NN 辅助选择和朴素随机选择，并报告真实纠缠误判率及对先验分布变化的鲁棒性。能够做出新的理论保证或实质上优于已发表方法的系统结果，才考虑发表；仅报告 NN 的 MAE 优于一个弱层析基线，不具备充分创新性。

**工具约束：** 默认环境未安装 CVXPY，因此 `--sdp-engine slsqp` 只用于数值探索；学生第 3 周需在可联网机器安装 CVXPY 和合适锥规划求解器，重新核对结果。代码保留 `--model gru`，让愿意学习旧项目结构的同学比较循环编码器，但优先检查物理和统计层面的正确性。
