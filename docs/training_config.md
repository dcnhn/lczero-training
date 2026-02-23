# Training Configuration Reference

Training is configured through YAML files located in `tf/configs/`. Below is a comprehensive reference of all available parameters.

## General Settings

| Parameter | Type | Description | Example/Default | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `name` | string | Unique identifier for the training run. Used for checkpoint directories and TensorBoard logs. | `"my-model"` | :white_check_mark: |
| `gpu` | string | GPU configuration. `"none"` for CPU, `0` for single GPU, `"0,1,2,3"` for specific GPUs, `"all"` for all available. | `"0,1,2,3"` | :white_check_mark: |

## Dataset Settings (`dataset`)

| Parameter | Type | Description | Example/Default | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `num_chunks` | int | Maximum number of chunk files to load. Set high and use `allow_less_chunks: true` to load all available. | `500_000_000` | Kept at default |
| `allow_less_chunks` | bool | If `true`, training proceeds even if fewer chunks than `num_chunks` are found. | `true` | Kept at default |
| `train_ratio` | float | Fraction of data used for training (remainder used for testing). | `0.95` | Kept at default |
| `sort_type` | string | How to sort chunk files before selecting `num_chunks`. `"mtime"` = by modification time (newest first), `"name"` = alphabetically, `"number"` = by game number in filename. "Latest" files first. | `"name"` | Kept at default |
| `input` | list | List of paths (relative or absolute) to directories containing `.gz` chunk files. | See example above | :white_check_mark: |
| `train_workers` | int | Number of parallel workers for loading training data. Higher values increase RAM usage. | `10` | :white_check_mark: |
| `test_workers` | int | Number of parallel workers for loading test data. | `4` | :white_check_mark: |
| `fast_chunk_loading` | bool | If `true`, uses optimized chunk loading (recommended). | `true` | Kept at default |
| `pc_min` | int | (Optional) Minimum piece count filter for positions. | `0` | :x: |
| `pc_max` | int | (Optional) Maximum piece count filter for positions. | `6` | :x: |

## Training Settings (`training`)

### General Training

| Parameter | Type | Description | Example/Default | Used in this work |
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

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `checkpoint_steps` | int | Save a checkpoint every N steps. | `10_000` | |
| `disable_checkpoints` | bool | If `true`, disables all checkpointing (useful for parameter search). | `false` | |
| `disable_pb_checkpointing` | bool | If `true`, disables protobuf checkpoint saving. | `false` | |
| `path` | string | Directory where network weights are saved. | `"networks"` | |

### Evaluation & Logging

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `test_steps` | int | Run evaluation on test set every N steps. | `10_000` | |
| `num_test_positions` | int | Number of positions to evaluate during testing. | `65_536` | |
| `train_avg_report_steps` | int | Log training metrics every N steps. | `1000` | |
| `validation_steps` | int | (Optional) Run validation every N steps if validation dataset is configured. | `5000` | |

### Stochastic Weight Averaging (SWA)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `swa` | bool | Enable Stochastic Weight Averaging for better generalization. | `true` | |
| `swa_output` | bool | If `true`, outputs SWA-averaged weights in addition to regular weights. | `true` | |
| `swa_max_n` | int | Maximum number of models to average in SWA. | `10` | |
| `swa_steps` | int | Update SWA average every N steps. | `100` | |

### Learning Rate Schedule

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `lr_values` | list | Learning rate values. First value is initial LR, subsequent values are used after corresponding boundaries. | `[0.0005, 0.00025, 0.0001]` | |
| `lr_boundaries` | list | Step numbers at which to switch to next LR value. Must have `len(lr_values) - 1` entries. | `[55_000, 110_000]` | |

### Optimizer Settings

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `optimizer` | string | Optimizer type: `"sgd"`, `"adam"`, `"adamw"`, `"nadam"`, `"rmsprop"`, `"adabelief"`. | `"adamw"` | |
| `beta_1` | float | Exponential decay rate for first moment (Adam/AdamW/Nadam). | `0.9` | |
| `beta_2` | float | Exponential decay rate for second moment (Adam/AdamW/Nadam). | `0.999` | |
| `epsilon` | float | Small constant for numerical stability. | `1e-6` | |
| `weight_decay` | float | Weight decay coefficient (only for AdamW). | `0.01` | |
| `sparse` | bool | Use sparse updates (experimental). | `false` | |
| `lookahead_optimizer` | bool | Wrap optimizer with Lookahead for smoother convergence. | `false` | |

### Batch Normalization / Renorm

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `renorm` | bool | Use batch renormalization instead of standard batch norm. | `true` | |
| `renorm_max_r` | float | Maximum ratio for renorm correction. | `1.0` | |
| `renorm_max_d` | float | Maximum difference for renorm correction. | `0.0` | |

### Value Focus (Curriculum Learning)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `value_focus_min` | float | Minimum value focus weight. | `1.0` | |
| `value_focus_slope` | float | Slope for value focus increase over training. | `0.0` | |

### Advanced

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `checkpoint_activations` | bool | Checkpoint activations to save memory (trades compute for memory). | `false` | |

## Loss Weights (`training.loss_weights:`)

All loss weights control the relative importance of each prediction head in the total loss.

| Parameter | Type | Description | Paper Value | Used in this work |
|-----------|------|-------------|-------------|-------------------|
| `policy` | float | Weight for hard policy loss (cross-entropy with search policy). | `1.0` | |
| `policy_soft` | float | Weight for soft policy loss (KL divergence with temperature-scaled policy). Higher because loss magnitude is smaller. | `8.0` | |
| `policy_optimistic_st` | float | Weight for optimistic short-term policy. | `0.0` | |
| `policy_opponent` | float | Weight for opponent policy prediction. | `0.0` | |
| `policy_next` | float | Weight for next-move policy prediction. | `0.0` | |
| `value_winner` | float | Weight for WDL (Win/Draw/Loss) prediction. | `1.0` | |
| `value_q` | float | Weight for Q-value (L2 regression). | `1.0` | |
| `value_st` | float | Weight for short-term value (L2 regression). | `1.0` | |
| `value_q_err` | float | Weight for Q-value uncertainty estimation. | `1.0` | |
| `value_st_err` | float | Weight for short-term value uncertainty estimation. | `1.0` | |
| `value_q_cat` | float | Weight for categorical Q-value (only 240M model). | `0.0` (6M) / `0.1` (240M) | |
| `value_st_cat` | float | Weight for categorical short-term value (only 240M model). | `0.0` (6M) / `0.1` (240M) | |
| `moves_left` | float | Weight for moves-left prediction. | `1.0` | |
| `reg` | float | Weight for L2 regularization (currently not implemented). | `0.0` | |
| `future` | float | Weight for future position prediction. | `0.0` | |

## Model Architecture (`model:`)

### Transformer Dimensions

| Parameter | Type | Description | 6M Value | 240M Value | Used in this work |
|-----------|------|-------------|----------|------------|-------------------|
| `embedding_size` | int | Size of input embeddings. | `256` | `1024` | |
| `encoder_layers` | int | Number of transformer encoder layers. | `8` | `15` | |
| `encoder_heads` | int | Number of attention heads. Should divide `encoder_d_model`. | `8` | `32` | |
| `encoder_d_model` | int | Dimension of Q, K, V vectors in attention. | `256` | `1024` | |
| `encoder_dff` | int | Hidden dimension in feed-forward layers. | `256` | `4096` | |
| `policy_embedding_size` | int | Embedding size for policy head. | `256` | `1024` | |
| `policy_d_model` | int | Dimension for policy attention. | `256` | `1024` | |
| `policy_d_aux` | int | Dimension for auxiliary policy layers. | `256` | `1024` | |
| `value_embedding_size` | int | Embedding size for value head. | `32` | `32` | |
| `moves_left_embedding_size` | int | Embedding size for moves-left head. | `32` | `32` | |
| `dropout_rate` | float | Dropout rate during training (0.0 = disabled). | `0.0` | `0.0` | |

### Embedding

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `embedding_style` | string | Embedding architecture: `"new"` (recommended) or `"old"`. | `"new"` | |
| `embedding_dense_sz` | int | Dense layer size in embedding. | `32` | |
| `input_type` | string | Input encoding type. `"classic"` for standard board representation. | `"classic"` | |

### Position Encoding (RPE)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `use_rpe_q` | bool | Use Relative Position Encoding for queries. | `true` | |
| `use_rpe_k` | bool | Use Relative Position Encoding for keys. | `true` | |
| `use_rpe_v` | bool | Use Relative Position Encoding for values. | `true` | |

### Smolgen (Dynamic Attention Generation)

Smolgen is a mechanism that dynamically generates attention weights based on the current board position. It was used in earlier BT-series models (BT2/BT3/BT4) but is **not used** in the paper's architecture, which relies on RPE instead. Smolgen and RPE address different aspects and are not mutually exclusive.

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `use_smolgen` | bool | Enable Smolgen dynamic attention generation. | `false` | |
| `smolgen_hidden_channels` | int | Hidden channels in Smolgen. | `16` | |
| `smolgen_hidden_sz` | int | Hidden size in Smolgen. | `64` | |
| `smolgen_gen_sz` | int | Generator size in Smolgen. | `64` | |
| `smolgen_activation` | string | Activation function for Smolgen. | `"swish"` | |

### Output Heads

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `value` | string | Value output type: `"wdl"` (Win/Draw/Loss) or `"scalar"`. | `"wdl"` | |
| `moves_left` | string | Moves-left head version. | `"v1"` | |
| `value_st` | bool | Enable short-term value head. | `true` | |
| `value_q` | bool | Enable Q-value head. | `true` | |
| `soft_policy` | bool | Enable soft policy head. | `true` | |
| `soft_policy_temperature` | float | Temperature for soft policy. | `4.0` | |
| `categorical_value_buckets` | int | Number of buckets for categorical value (240M only). | `32` | |
| `policy_optimistic_st` | bool | Enable optimistic short-term policy head. | `false` | |
| `policy_opponent` | bool | Enable opponent policy prediction head. | `false` | |
| `policy_next` | bool | Enable next-move policy prediction head. | `false` | |

### Architecture Ablations

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `omit_qkv_biases` | bool | Remove biases from Q/K/V projections. Increases speed ~10% without quality loss. | `true` | |
| `encoder_rms_norm` | bool | Use RMSNorm instead of LayerNorm. Faster without quality degradation. | `true` | |
| `use_logit_gating` | bool | Use logit gating mechanism. | `false` | |

### Quantization (Experimental)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `quantize_activations` | bool | Quantize activations for inference. | `false` | |
| `quantize_weights` | bool | Quantize weights for inference. | `false` | |
| `quantize_activation_bits` | int | Bit width for activation quantization. | `8` | |
| `quantize_weight_bits` | int | Bit width for weight quantization. | `8` | |
| `quantize_channels` | bool | Per-channel quantization. | `false` | |
| `rep_quant` | bool | Representation quantization. | `false` | |

### Debug / Analysis

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `return_attn_wts` | bool | Return attention weights (for visualization/analysis). | `true` | |
| `return_activations` | bool | Return intermediate activations (for analysis). | `false` | |
