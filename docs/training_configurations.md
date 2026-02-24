# Configurations for Experiments

The following configurations are mainly used to reproduce results from the paper.

Available configurations can be found in `tf/configs`:

| Name | Purpose | Description |
|------|------------|-------------|
| `12m_multi_gpu_rmsprop-wAPE.yaml` | Positional Encoding | 12M-parameter configuration for RPE ablation experiments. Uses absolute positional encoding (APE). |
| `6m_multi_gpu_rmsprop-wRPE.yaml` | Positional Encoding | 6M-parameter configuration for RPE ablation experiments. Uses relative position encoding (RPE). |
| `6m_multi_gpu_rmsprop-wAPE.yaml` | Positional Encoding | 6M-parameter configuration for RPE ablation experiments. Uses absolute positional encoding (APE). |
| `6m_multi_gpu_rmsprop-wRPE-newData.yaml` | Training Data Effect | 6M parameter model using newer data from 2026. Used to check if training data impacts model performance. |
| `6m_multi_gpu_rmsprop-wRPE-smallSet.yaml` | Training Data Effect | 6M parameter model using older data from 2021. Used to check if training data impacts model performance. |
| `6m_multi_gpu_adamw-wRPE.yaml` | Optimizer Effect | Initial experiments used RMSprop because Nadam led to NaNs in multi-GPU setups. This config uses AdamW to check for improvement or regression compared to RMSprop. |
| `240m_multi_gpu_adamw.yaml` | Not applicable | 240M parameter model using AdamW optimizer. Hardware was not sufficient for training. |
| `debug_cpu.yaml` | Setup | Minimal debug configuration for CPU. Only used to verify setup. |
| `debug.yaml` | Setup | Minimal debug configuration for GPU. Only used to verify setup. |
| `example.yaml` | Not applicable | Example configuration from https://github.com/daniel-monroe/lczero-training. |

# Training Configurations for Hyperparameter Search

The hyperparameter search configurations below are designed to systematically explore different ratios between policy and value loss weights. Following the paper/reference implementation, we keep the total loss-weight budget fixed at 14.0 and redistribute it between the policy heads (policy, policy_soft) and the value heads (value_winner, value_q, value_st, value_q_err, value_st_err). Keeping the total constant avoids unintended effective step-size changes and enables direct comparison of how the policy/value balance affects training outcomes. The baseline split (9/14 policy, 5/14 value) matches the paper, while other variants test more value-focused or policy-focused allocations.

Available hyperparameter search configurations can be found in `tf/param_search`:

| Name | Purpose | Description |
|------|---------|-------------|
| `6m_search_base.yaml` | Baseline | 6M parameter baseline configuration allocating 64% of the loss budget to the policy and 36% to value heads. |
| `6m_search_pol-0.25.yaml` | Policy vs. Value | 6M parameter configuration allocating 25% of the loss budget to the policy and 75% to value heads. |
| `6m_search_pol-0.35.yaml` | Policy vs. Value | 6M parameter configuration allocating 35% of the loss budget to the policy and 65% to value heads. |
| `6m_search_pol-0.5.yaml` | Policy vs. Value | 6M parameter configuration allocating 50% of the loss budget to the policy and 50% to value heads. |
| `6m_search_pol-0.75.yaml` | Policy vs. Value | 6M parameter configuration allocating 75% of the loss budget to the policy and 25% to value heads. |
