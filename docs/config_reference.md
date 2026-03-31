# Training Configuration Reference

Training is configured through YAML files located in `tf/configs/`. Below is a comprehensive reference of all available parameters.

## General Settings

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `name` | string | Unique identifier for the training run. Used for checkpoint directories and TensorBoard logs. | `"my-model"` | :white_check_mark: |
| `gpu` | string (int for Single-GPU) | GPU configuration. `"none"` for CPU, `0` for single GPU, `"0,1,2,3"` for specific GPUs, `"all"` for all available. | `"0,1,2,3"` | :white_check_mark: |

## Dataset Settings (`dataset`)

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `num_chunks` | int | Maximum number of chunk files to load. Set high and use `allow_less_chunks: true` to load all available. | `500_000_000` | Kept at default |
| `allow_less_chunks` | bool | If `true`, training proceeds even if fewer chunks than `num_chunks` are found. | `true` | Kept at default |
| `sort_type` | string | How to sort chunk files before selecting `num_chunks`. `"mtime"` = by modification time (newest first), `"name"` = alphabetically, `"number"` = by game number in filename. "Latest" files first. | `"name"` | Kept at default |
| `input` | list | List of paths (relative or absolute) to directories containing `.gz` chunk files. | See example above | :white_check_mark: |
| `train_ratio` | float | Fraction of data used for training (remainder used for testing). | `0.95` | Kept at default |
| `input_train` | list | List of paths for training data. Alternative to `input` and `train_ratio`. | — | :x: |
| `input_test` | list | List of paths for test data. Alternative to `input` and `train_ratio`. | — | :x: |
| `input_validation` | list | List of paths for validation data. Alternative to `input` and `train_ratio`. | — | :x: |
| `train_workers` | int | Number of parallel workers for loading training data. Higher values increase RAM usage. | `10` | :white_check_mark: |
| `test_workers` | int | Number of parallel workers for loading test data. | `4` | :white_check_mark: |
| `fast_chunk_loading` | bool | If `true`, uses optimized chunk loading (recommended). | `true` | Kept at default |
| `pc_min` | int | (Optional) Minimum piece count filter for positions. | `0` | :x: |
| `pc_max` | int | (Optional) Maximum piece count filter for positions. | `6` | :x: |

## Training Settings (`training`)

### General Training

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `precision` | string | Floating-point precision. `"half"` (FP16) is faster and uses less memory, `"single"` (FP32) is more stable. | `"half"` | :white_check_mark: |
| `batch_size` | int | Total batch size across all GPUs. | `2048` | :white_check_mark: |
| `num_batch_splits` | int | Split batch for gradient accumulation. Effective batch = `batch_size`, but memory usage ≈ `batch_size / num_batch_splits`. | `1` | :white_check_mark: |
| `total_steps` | int | Total number of training steps. | `200_000` | :white_check_mark: |
| `warmup_steps` | int | Number of steps for linear learning rate warmup from 0 to initial LR. | `1000` | :white_check_mark: |
| `shuffle_size` | int | Size of the shuffle buffer. Larger = better randomization but more RAM. | `500_000` | :white_check_mark: |
| `mask_legal_moves` | bool | If `true`, masks illegal moves in policy output (recommended). | `true` | Kept at default |
| `check_numerics` | bool | If `true`, checks for NaN/Inf during training. Useful for debugging but slower. | `false` | :white_check_mark: for debugging |
| `max_grad_norm` | float | Maximum gradient norm for clipping. Prevents exploding gradients. | `10.0` | Kept at default |

### Checkpointing

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `checkpoint_steps` | int | Save a checkpoint every N steps. | `10_000` | :white_check_mark: |
| `disable_checkpoints` | bool | If `true`, disables all checkpointing (useful for parameter search). | `false` | :white_check_mark: in hyperparameter search |
| `disable_pb_checkpointing` | bool | If `true`, disables protobuf (`.pb.gz`) checkpoint saving. Protobuf checkpoints are required to load weights into the LC0 engine for inference/play. | `false` | :white_check_mark: in hyperparameter search |
| `path` | string | Directory where network weights are saved. | `"networks"` | Kept at default |

### Evaluation & Logging

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `test_steps` | int | Run evaluation on test set every N steps and log metrics to TensorBoard. | `10_000` | :white_check_mark: |
| `num_test_positions` | int | Number of positions to evaluate during testing. Number of evaluation batches: `num_test_positions` / (`batch_size` / `num_batch_splits`) | `65_536` | :white_check_mark: |
| `train_avg_report_steps` | int | Print averaged training metrics to the terminal every N steps. | `1000` | :white_check_mark: |
| `validation_steps` | int | (Optional) Run validation every N steps if validation dataset is configured. | `5000` | :x: |

### Stochastic Weight Averaging (SWA)

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `swa` | bool | Enable Stochastic Weight Averaging (implemented as exponential moving average of weights) for better generalization. | `true` | :white_check_mark:, disabled in hyperparameter search |
| `swa_output` | bool | If `true`, outputs SWA-averaged weights in addition to regular weights. | `true` | :white_check_mark:, disabled in hyperparameter search |
| `swa_max_n` | int | Maximum number of weight sets to average in SWA. | `10` | :white_check_mark:, disabled in hyperparameter search |
| `swa_steps` | int | Update SWA average every N steps. | `100` | :white_check_mark:, disabled in hyperparameter search |

### Learning Rate Schedule

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `lr_values` | list | Learning rate values. First value is initial LR, subsequent values are used after corresponding boundaries. | `[0.0005, 0.00025, 0.0001]` | :white_check_mark: |
| `lr_boundaries` | list | Step numbers at which to switch to next LR value. Must have `len(lr_values) - 1` entries. | `[55_000, 110_000]` | :white_check_mark: |

### Optimizer Settings

> **Note:** Only SGD, RMSprop, AdamW, and Nadam have been tested in this work. **Nadam** caused NaN values in multi-GPU setups. Only AdamW currently supports weight decay.

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `optimizer` | string | Optimizer type: `"sgd"`, `"adam"`, `"adamw"`, `"nadam"`, `"rmsprop"`, `"adabelief"`. | `"adamw"` | :white_check_mark: |
| `beta_1` | float | Exponential decay rate for first moment (Adam/AdamW/Nadam). | `0.9` | Kept at default |
| `beta_2` | float | Exponential decay rate for second moment (Adam/AdamW/Nadam). | `0.999` | Kept at default |
| `epsilon` | float | Small constant for numerical stability (avoids division by zero). | `1e-6` | Kept at default |
| `weight_decay` | float | Weight decay coefficient (only for AdamW). | `0.01` | :white_check_mark: |
| `sparse` | bool | Use sparse updates (experimental). | `false` | Kept at default |


### Unused parameters (Legacy)

> **Note:** These parameters appear to be legacy code. These settings are not relevant for transformer training.

| Parameter | Used/Adapted in this work |
|-----------|------|
| `renorm` |  :x: |
| `renorm_max_r` | :x: |
| `renorm_max_d` |  :x: |
| `value_focus_min` | :x: |
| `value_focus_slope` | :x: |
| `checkpoint_activations` | :x: |
| `lookahead_optimizer` | :x: |



## Loss Weights (`training.loss_weights`)

All loss weights control the relative importance of each prediction head in the total loss.
Refer to the appendix of the paper: [https://arxiv.org/abs/2409.12272](https://arxiv.org/abs/2409.12272).

> **Note:** The L2 value loss, value error loss, and categorical value loss from the paper are each split into two parallel outputs in this implementation
> - `value_q` and `value_st` (short-term) are parallel scalar outputs from the value head (long-term and short-term value).
> - `value_q_err` and `value_st_err` are parallel error predictions for those two value outputs.
> - `value_q_cat` and `value_st_cat` are parallel categorical value outputs (only used in the 240M model).


| Parameter | Type | Description | Paper Value | Used/Adapted in this work |
|-----------|------|-------------|-------------|-------------------|
| `policy` | float | Weight for policy loss. | `1.0` | Paper value used |
| `policy_soft` | float | Weight for soft policy loss (temperature-scaled policy). Higher because loss magnitude is smaller. | `8.0` | Paper value used |
| `policy_optimistic_st` | float | Weight for optimistic short-term policy. | `0.0` | :x: |
| `policy_opponent` | float | Weight for opponent policy prediction. | `0.0` | :x: |
| `policy_next` | float | Weight for next-move policy prediction. | `0.0` | :x: |
| `value_winner` | float | Weight for WDL (Win/Draw/Loss) prediction. | `1.0` | Paper value used |
| `value_q` | float | Weight for Q-value (L2 regression). | `1.0` | Paper value used |
| `value_st` | float | Weight for short-term value (L2 regression). | `1.0` | Paper value used |
| `value_q_err` | float | Weight for Q-value uncertainty estimation. | `1.0` | Paper value used |
| `value_st_err` | float | Weight for short-term value uncertainty estimation. | `1.0` | Paper value used |
| `value_q_cat` | float | Weight for categorical Q-value (only 240M model). | `0.0` (6M) / `0.1` (240M) | Paper value used |
| `value_st_cat` | float | Weight for categorical short-term value (only 240M model). | `0.0` (6M) / `0.1` (240M) | Paper value used |
| `moves_left` | float | Weight for moves-left prediction. | `1.0` | Kept at default |
| `reg` | float | Weight for L2 regularization (currently not implemented by authors, used AdamW instead). | `0.0` | :x: |
| `future` | float | Weight for future position prediction. | `0.0` | :x: |



## Model Architecture (`model`)

### Transformer Dimensions
All values in this table are taken directly from the paper ([https://arxiv.org/abs/2409.12272](https://arxiv.org/abs/2409.12272)). Dropout is set as in Vaswani et al. ("Attention is All You Need"), since the authors did not specify dropout settings.

| Parameter | Type | Description | 6M Value | 240M Value |
|-----------|------|-------------|----------|------------|
| `embedding_size` | int | Size of input embeddings. | `256` | `1024` | 
| `encoder_layers` | int | Number of transformer encoder layers. | `8` | `15` | 
| `encoder_heads` | int | Number of attention heads. Should divide `encoder_d_model`. | `8` | `32` | 
| `encoder_d_model` | int | Dimension of Q, K, V vectors in attention. | `256` | `1024` | 
| `encoder_dff` | int | Hidden dimension in feed-forward layers. | `256` | `4096` | 
| `policy_embedding_size` | int | Embedding size for policy head. | `256` | `1024` | 
| `policy_d_model` | int | Dimension for policy attention. | `256` | `1024` | 
| `policy_d_aux` | int | Dimension for auxiliary policy layers. | `256` | `1024` | 
| `value_embedding_size` | int | Embedding size for value head. | `32` | `32` | 
| `moves_left_embedding_size` | int | Embedding size for moves-left head. | `32` | `32` | 
| `dropout_rate` | float | Dropout rate during training (0.0 = disabled). | `0.1` | `0.1` | 

### Embedding
| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `embedding_style` | string | Embedding architecture: `"new"` (recommended) or `"old"`. | `"new"` | Kept at default |
| `embedding_dense_sz` | int | Controls the intermediate preprocess feature size before concatenation and final embedding projection. | `32` | Kept at default |
| `input_type` | string | Input encoding type. `"classic"` for standard board representation. | `"classic"` | Kept at default |

### Position Encoding (PE)
**Note:** Relative Position Encoding (RPE) is used by default in this work. Absolute Position Encoding (`use_absolute_pe`) was only used in an ablation experiment. If `use_absolute_pe` is set to `true`, all RPE options (`use_rpe_q`, `use_rpe_k`, `use_rpe_v`) must be set to `false`. Conversely, if any RPE option is `true`, `use_absolute_pe` must be `false`.

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `use_rpe_q` | bool | Use relative position encoding for queries. | `true` | :white_check_mark: |
| `use_rpe_k` | bool | Use relative position encoding for keys. | `true` | :white_check_mark: |
| `use_rpe_v` | bool | Use relative position encoding for values. | `true` | :white_check_mark: |
| `use_absolute_pe` | bool | Use absolute position encoding. | `false` | Only used in ablation experiment. |

### Smolgen

**Note:** The following parameters were not mentioned in the paper. Therefore, this feature is disabled.

| Parameter | Used/Adapted in this work |
|-----------|------|
| `use_smolgen` | :x:, set to `false` |
| `smolgen_hidden_channels` | :x: |
| `smolgen_hidden_sz` | :x: |
| `smolgen_gen_sz` | :x: |
| `smolgen_activation` | :x: |

### Output Heads

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `value` | string | Value output type: `"wdl"` (Win/Draw/Loss) or `"scalar"`. | `"wdl"` | Used 'wdl' as in the paper |
| `moves_left` | string | Moves-left head version. | `"v1"` | Kept at default |
| `value_st` | bool | Enable short-term value head. | `true` | Used as in the paper |
| `value_q` | bool | Enable Q-value head. | `true` | Used as in the paper |
| `soft_policy` | bool | Enable soft policy head. | `true` | Used as in the paper |
| `soft_policy_temperature` | float | Temperature for soft policy. | `4.0` | Kept at default as paper did not mention a value |
| `categorical_value_buckets` | int | Number of buckets for categorical value (240M only). | `32` | Kept at default |
| `policy_optimistic_st` | bool | Enable optimistic short-term policy head. | `false` | :x: |
| `policy_opponent` | bool | Enable opponent policy prediction head. | `false` | :x: |
| `policy_next` | bool | Enable next-move policy prediction head. | `false` | :x: |

### Architecture Ablations

**Note:** Omitting QKV biases and using RMSNorm follow the paper; logit gating is disabled because it is not mentioned in the paper.

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `omit_qkv_biases` | bool | Remove biases from Q/K/V projections. | `true` | Kept at default |
| `encoder_rms_norm` | bool | Use RMSNorm instead of LayerNorm. Faster without quality degradation. | `true` | Kept at default |
| `use_logit_gating` | bool | Use logit gating mechanism. | `false` | :x: |

### Quantization 
**Note:** The following parameters were not mentioned in the paper. Therefore, this feature is disabled.

| Parameter | Used/Adapted in this work |
|-----------|------|
| `quantize_activations` | :x: |
| `quantize_weights` | :x: |
| `quantize_activation_bits` | :x: |
| `quantize_weight_bits` | :x: |
| `quantize_channels` | :x: |
| `rep_quant` | :x: |

### Debug / Analysis

| Parameter | Type | Description | Example/Default | Used/Adapted in this work |
|-----------|------|-------------|---------|-------------------|
| `return_attn_wts` | bool | Return attention weights (for visualization/analysis). | `true` | :white_check_mark:, this feature is needed to visualize the attention maps. |
| `return_activations` | bool | Return intermediate activations (for analysis). | `false` | :x:, used default |
