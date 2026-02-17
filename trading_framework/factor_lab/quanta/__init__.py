"""QuantaAlpha 多智能体因子挖掘框架 (Experiment 013)

严格按论文方法复现核心挖掘流程:
- Phase A: 多样化初始规划 (Idea Agent + Factor Agent + Eval Agent)
- Phase B: 轨迹级自我演化 (Mutation + Crossover)
- Phase C: 最终评估 (Alpha158 + 挖掘因子 → LightGBM)
"""
