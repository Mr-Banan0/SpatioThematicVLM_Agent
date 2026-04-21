"""
SpatioThematicVLM Evaluation Framework
=======================================
无需 Ground Truth 的自动化评测体系，覆盖五个研究问题:

  RQ1: 结构化输出有效性 (Schema Compliance)
  RQ2: 跨运行稳定性 (Cross-run Consistency)
  RQ3: 生成参数敏感性 (Parameter Sensitivity)
  RQ4: 输入鲁棒性 (Input Robustness)
  RQ5: 多智能体流水线一致性 (Inter-agent Coherence)

Usage:
    python -m evaluation.run_all --image test.png --mode quick
"""
