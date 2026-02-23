<!-- # Introduction

This repository is a fork of https://github.com/daniel-monroe/lczero-training. Its primary goal is to reproduce and study the results presented in the paper *“Mastering Chess with a Transformer Model”* (https://arxiv.org/abs/2409.12272).

The project adapts the original Leela Chess Zero training pipeline to support transformer-based architectures. It is intended as a research-oriented codebase for exploring training dynamics, architectural choices, and reproducibility rather than as a polished end-user application. 
<br>
The results obtained within the scope of this project are **not** presented in this repository. Instead, they are described in detail in a separate report, which is not published here. -->

# Introduction

This repository is a fork of https://github.com/daniel-monroe/lczero-training. Its primary goal is to reproduce and study the results presented in the paper *“Mastering Chess with a Transformer Model”* (https://arxiv.org/abs/2409.12272).

The project adapts the original Leela Chess Zero training pipeline to support transformer-based architectures. It is intended as a research-oriented codebase for exploring training dynamics, architectural choices, and reproducibility. 

This work was carried out in the context of the “Practical Work in AI (Master)” course within the Artificial Intelligence master’s program at Johannes Kepler University Linz (JKU). The results obtained within the scope of this project are **not** presented in this repository. Instead, they are described in detail in a separate report, which is not published here.


# Setup


## Prerequisites
- This repository is designed to run on **Linux**.
- If you are working on **Windows**, you must have **Windows Subsystem for Linux (WSL)** installed and configured.
- A working **conda** or **mamba** installation (e.g. Miniconda or Mambaforge).
- **For training with GPU:** An NVIDIA GPU with a compatible NVIDIA driver installed.
  - Must support **CUDA ≥ 12.2**
- All scripts/commands assume your terminal is opened at the **root directory of this repository**
- [Recommended] The Protocol Buffers compiler (`protoc`) is installed:
  ```bash
  sudo apt update
  sudo apt install -y protobuf-compiler

### Why Protocol Buffers Compilation Is Required

The `.proto` files located in `libs/lczero-common/proto/` (`chunk.proto` and `net.proto`) define message schemas-structured templates that specify how data is organized and serialized for communication (see the [Protocol Buffers documentation](https://protobuf.dev/overview/))

These schemas are not directly usable by Python. They must first be compiled into Python modules using `protoc`. The generated Python files provide:

- **Typed message classes**
- **Efficient binary serialization and deserialization**
- **A single, well-defined structure for exchanging data between components**

Without compiling these `.proto` files, the corresponding Python imports will fail and any code relying on these message definitions will not run.

**Note:**
The repository **already** includes pre-compiled Python files in `tf/proto/`. If you modify any `.proto` files, you must run `init.sh` to regenerate the corresponding Python files in `tf/proto/` so that the changes are reflected in the Python code. **As provided, the repository is fully self-contained and can be used without running any compilation steps.**


## Setup of Conda Environment and Repository

Create a new conda environment with the required Python version:

```bash
conda create --name <ENV_NAME> python=3.10
conda activate <ENV_NAME>
```

Once the environment is activated, clone the repository and navigate into its root directory:
```bash
git clone https://github.com/dcnhn/lczero-training.git
cd lczero-training
```

After cloning, install the Python dependencies.
For a system with **NVIDIA GPU**:
```bash
pip install -r ./tf/requirements.txt
```
Otherwise:
```bash
pip install -r ./tf/requirements_cpu.txt
```

## Verify Environment Setup
Run the following script to verify that the environment is correctly configured:
```bash
# For GPU
python tf/train.py --cfg tf/configs/debug.yaml  --output ./tmp/debug.txt

# For CPU
python tf/train.py --cfg tf/configs/debug_cpu.yaml  --output ./tmp/debug.txt
```
This command runs a small-scale debug training to validate that all dependencies, configurations, and runtime components are working correctly.

A successful run should complete without errors and produce output similar to the following:
![Successful debug training run](doc/finished_debug_train.png)

# Training

> **Note:** This repository includes a small sample dataset in `playground_data/` for testing purposes only. These files are used to verify that the environment is set up correctly (see [Verify Environment Setup](#verify-environment-setup)). Once setup is confirmed, this folder can be safely deleted to free up space.

## Data preparation
In order to start a training session you first need to download training data from https://storage.lczero.org/files/training_data/.
The **LCZero database** contains multiple versions of the training data format, reflecting changes and improvements over time.<br>
⚠️ **Recommendation (as of 2026-02-17):** use training data generated in **2024 or later**, as older datasets may rely on deprecated formats or lack newer features expected by the current training pipeline.

### [OPTIONAL] Automated Data Fetching and Download

For convenience, a script is provided that can fetch training data from:

- `https://storage.lczero.org/files/training_data/`
- or `https://storage.lczero.org/files/training_data/<SUB_DIR>/`

The script performs the following tasks:

- Filters out all data generated **before 2024**
- Generates a **csv-file** containing metadata
- Generates a **txt-file** (default name: *lczero_largest_10_tars.txt*) containing only URLs, suitable for use with `wget`

For example:
```bash
python lc0_data_scraper.py 
  --lczero-url https://storage.lczero.org/files/training_data/test91/
```

| Option         | Description                                           | Type    | Default                                                   |
|----------------|-------------------------------------------------------|---------|-----------------------------------------------------------|
| `--lczero-url` | Base URL to scrape for LCZero training data            | string  | `https://storage.lczero.org/files/training_data/`         |
| `--save-top`   | Number of the largest `.tar` files (by size) to save   | integer | `10`                                                      |


To download the data, you can execute the following command (assuming the default file name):
```bash
wget -i lczero_largest_10_tars.txt -c
```
Feel free to modify the text file before starting the download process.

After downloading, extract the `.tar` archives to access the training chunks.


### Data Preprocessing

#### Data Format

The LCZero training data exists in multiple format versions. This documentation focuses on **V6**, which is the most recent and recommended format.

The training data is processed by `tf/chunkparser.py`, which converts raw V6 data into a 5-element tuple: `(planes, probs, winner, best_q, plies_left)`.

When interpreted as NumPy arrays, each training example has the following structure:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `planes` | `(112, 64)` | float32 | Board state as 112 feature planes, each 8×8 (flattened to 64). The original 104 planes are augmented with 8 additional planes for castling rights, side to move, rule 50 count, and board edge detection. |
| `probs` | `(1858,)` | float32 | Policy probabilities for all possible moves (corresponds to `float probabilities[1858]` in the V6TrainingData C++ struct). |
| `winner` | `(3,)` | float32 | Game outcome from the current player's perspective: win, draw, loss probabilities. |
| `best_q` | `(3,)` | float32 | Position value after search (Q-value), also as win, draw, loss probabilities. |
| `plies_left` | scalar | float32 | Estimated number of plies remaining until game end. |

For more details, see the [official training tuple documentation](https://github.com/LeelaChessZero/lczero-training/blob/master/docs/training_tuple.md).

#### Rescoring

The raw training data from LCZero is stored in **V6 format**, but this training pipeline requires **V7 format**. The `rescore_file()` function in `tf/chunkparser.py` performs this conversion by computing additional training targets.

**Why rescoring is needed:**

The original V6 data contains `root_q` and `root_d` for each position. These are the Q-value (expected game outcome) and draw probability computed by the **MCTS search** at the root of the search tree. While these values incorporate search information, they are computed **independently for each position** without considering the evaluations of subsequent positions in the same game.

The transformer architecture benefits from *temporally smoothed* value targets that incorporate information from future positions along the game trajectory. This helps the model learn more stable and consistent value estimates.

**What rescoring adds:**

The rescoring process applies an **exponential moving average (EMA)** to the Q-values and draw probabilities across the game trajectory, computed **backwards from the game end**:

- `st_q` (short-term Q): EMA of Q-values with α = 1 - 1/6 (≈ 0.833)
- `st_d` (short-term D): EMA of draw probabilities with the same α

With α ≈ 0.833, the EMA places **more weight on future positions** (closer to the game end):
- The current position's value gets weight (1 - α) ≈ **17%**
- The accumulated value from future positions gets weight α ≈ **83%**

For example, if position 20 has `root_q = 0.3` but the following positions 21–25 all have `root_q ≈ 0.5`, the smoothed `st_q` for position 20 will be higher than 0.3 because it incorporates information from future positions.

These smoothed targets provide a more robust training signal by reducing noise from individual position evaluations and incorporating future game outcomes into the value targets.

To rescore your training data, use the `tf/rescore_files.py` script on your downloaded `.gz` chunk files before training:
```bash
python tf/rescore_files.py --cfg tf/configs/<CONFIG>.yaml
```

The script parses the YAML configuration file and reads all paths listed under `dataset.input`. These paths can be specified as:
- **Relative paths** (relative to the repository root)
- **Absolute paths**

You can list multiple directories containing training chunks:

![Data path configuration in YAML](docs/rescore_data_paths.png)

The script will recursively scan all specified directories for `.gz` chunk files and process them. This means you only need to configure your data paths once in the YAML file, and the same configuration can be used for both rescoring and training.

#### Verifying Data Format

After rescoring (or if you're unsure about your data format), you can verify that all chunk files are in the correct V7/V7B format:

```bash
python tf/check_files_V7B.py --cfg tf/configs/<CONFIG>.yaml
```

This script reads the same `dataset.input` paths from your YAML configuration and checks each `.gz` file's header to confirm it uses the V7 or V7B format. Any files with incompatible formats will be listed, allowing you to identify and fix issues before training.



## Training Configuration

Training is configured through YAML files located in `tf/configs/`. Below is a comprehensive reference of all available parameters.

### General Settings

| Parameter | Type | Description | Example/Default | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `name` | string | Unique identifier for the training run. Used for checkpoint directories and TensorBoard logs. | `"my-model"` | :white_check_mark: |
| `gpu` | string | GPU configuration. `"none"` for CPU, `0` for single GPU, `"0,1,2,3"` for specific GPUs, `"all"` for all available. | `"0,1,2,3"` | :white_check_mark: |

### Dataset Settings (`dataset`)

| Parameter | Type | Description | Example/Default | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `num_chunks` | int | Maximum number of chunk files to load. Set high and use `allow_less_chunks: true` to load all available. | `500_000_000` | Kept at default |
| `allow_less_chunks` | bool | If `true`, training proceeds even if fewer chunks than `num_chunks` are found. | `true` | Kept at default |
| `train_ratio` | float | Fraction of data used for training (remainder used for testing). | `0.95` | Kept at default |
| `sort_type` | string | How to sort chunk files before selecting `num_chunks`. `"mtime"` = by modification time (newest first), `"name"` = alphabetically, `"number"` = by game number in filename. "Latest" files first. | `"name"` | Kept at default |
| `input` | list | List of paths (relative or absolute) to directories containing `.gz` chunk files. | See example below | :white_check_mark: |
| `train_workers` | int | Number of parallel workers for loading training data. Higher values increase RAM usage. | `10` | :white_check_mark: |
| `test_workers` | int | Number of parallel workers for loading test data. | `4` | :white_check_mark: |
| `fast_chunk_loading` | bool | If `true`, uses optimized chunk loading (recommended). | `true` | Kept at default |
| `pc_min` | int | (Optional) Minimum piece count filter for positions. | `0` | Kept at default |
| `pc_max` | int | (Optional) Maximum piece count filter for positions. | `6` | Kept at default |

### Training Settings (`training`)

#### General Training

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

#### Checkpointing

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `checkpoint_steps` | int | Save a checkpoint every N steps. | `10_000` | |
| `disable_checkpoints` | bool | If `true`, disables all checkpointing (useful for parameter search). | `false` | |
| `disable_pb_checkpointing` | bool | If `true`, disables protobuf checkpoint saving. | `false` | |
| `path` | string | Directory where network weights are saved. | `"networks"` | |

#### Evaluation & Logging

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `test_steps` | int | Run evaluation on test set every N steps. | `10_000` | |
| `num_test_positions` | int | Number of positions to evaluate during testing. | `65_536` | |
| `train_avg_report_steps` | int | Log training metrics every N steps. | `1000` | |
| `validation_steps` | int | (Optional) Run validation every N steps if validation dataset is configured. | `5000` | |

#### Stochastic Weight Averaging (SWA)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `swa` | bool | Enable Stochastic Weight Averaging for better generalization. | `true` | |
| `swa_output` | bool | If `true`, outputs SWA-averaged weights in addition to regular weights. | `true` | |
| `swa_max_n` | int | Maximum number of models to average in SWA. | `10` | |
| `swa_steps` | int | Update SWA average every N steps. | `100` | |

#### Learning Rate Schedule

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `lr_values` | list | Learning rate values. First value is initial LR, subsequent values are used after corresponding boundaries. | `[0.0005, 0.00025, 0.0001]` | |
| `lr_boundaries` | list | Step numbers at which to switch to next LR value. Must have `len(lr_values) - 1` entries. | `[55_000, 110_000]` | |

#### Optimizer Settings

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `optimizer` | string | Optimizer type: `"sgd"`, `"adam"`, `"adamw"`, `"nadam"`, `"rmsprop"`, `"adabelief"`. | `"adamw"` | |
| `beta_1` | float | Exponential decay rate for first moment (Adam/AdamW/Nadam). | `0.9` | |
| `beta_2` | float | Exponential decay rate for second moment (Adam/AdamW/Nadam). | `0.999` | |
| `epsilon` | float | Small constant for numerical stability. | `1e-6` | |
| `weight_decay` | float | Weight decay coefficient (only for AdamW). | `0.01` | |
| `sparse` | bool | Use sparse updates (experimental). | `false` | |
| `lookahead_optimizer` | bool | Wrap optimizer with Lookahead for smoother convergence. | `false` | |

#### Batch Normalization / Renorm

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `renorm` | bool | Use batch renormalization instead of standard batch norm. | `true` | |
| `renorm_max_r` | float | Maximum ratio for renorm correction. | `1.0` | |
| `renorm_max_d` | float | Maximum difference for renorm correction. | `0.0` | |

#### Value Focus (Curriculum Learning)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `value_focus_min` | float | Minimum value focus weight. | `1.0` | |
| `value_focus_slope` | float | Slope for value focus increase over training. | `0.0` | |

#### Advanced

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `checkpoint_activations` | bool | Checkpoint activations to save memory (trades compute for memory). | `false` | |

### Loss Weights (`training.loss_weights:`)

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

### Model Architecture (`model:`)

#### Transformer Dimensions

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

#### Embedding

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `embedding_style` | string | Embedding architecture: `"new"` (recommended) or `"old"`. | `"new"` | |
| `embedding_dense_sz` | int | Dense layer size in embedding. | `32` | |
| `input_type` | string | Input encoding type. `"classic"` for standard board representation. | `"classic"` | |

#### Position Encoding (RPE)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `use_rpe_q` | bool | Use Relative Position Encoding for queries. | `true` | |
| `use_rpe_k` | bool | Use Relative Position Encoding for keys. | `true` | |
| `use_rpe_v` | bool | Use Relative Position Encoding for values. | `true` | |

#### Smolgen (Dynamic Attention Generation)

Smolgen is a mechanism that dynamically generates attention weights based on the current board position. It was used in earlier BT-series models (BT2/BT3/BT4) but is **not used** in the paper's architecture, which relies on RPE instead. Smolgen and RPE address different aspects and are not mutually exclusive.

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `use_smolgen` | bool | Enable Smolgen dynamic attention generation. | `false` | |
| `smolgen_hidden_channels` | int | Hidden channels in Smolgen. | `16` | |
| `smolgen_hidden_sz` | int | Hidden size in Smolgen. | `64` | |
| `smolgen_gen_sz` | int | Generator size in Smolgen. | `64` | |
| `smolgen_activation` | string | Activation function for Smolgen. | `"swish"` | |

#### Output Heads

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

#### Architecture Ablations

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `omit_qkv_biases` | bool | Remove biases from Q/K/V projections. Increases speed ~10% without quality loss. | `true` | |
| `encoder_rms_norm` | bool | Use RMSNorm instead of LayerNorm. Faster without quality degradation. | `true` | |
| `use_logit_gating` | bool | Use logit gating mechanism. | `false` | |

#### Quantization (Experimental)

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `quantize_activations` | bool | Quantize activations for inference. | `false` | |
| `quantize_weights` | bool | Quantize weights for inference. | `false` | |
| `quantize_activation_bits` | int | Bit width for activation quantization. | `8` | |
| `quantize_weight_bits` | int | Bit width for weight quantization. | `8` | |
| `quantize_channels` | bool | Per-channel quantization. | `false` | |
| `rep_quant` | bool | Representation quantization. | `false` | |

#### Debug / Analysis

| Parameter | Type | Description | Example | Used in this work |
|-----------|------|-------------|---------|-------------------|
| `return_attn_wts` | bool | Return attention weights (for visualization/analysis). | `true` | |
| `return_activations` | bool | Return intermediate activations (for analysis). | `false` | |


## Training Process
TODO: Describe how training is started


## Tensorboard
TODO: Explain how a board is started


# Evaluation

## Creating an Agent
TODO: Check out how to create an agent and document here.

## Puzze-solving Evaluation
TODO: Check out `https://github.com/google-deepmind/searchless_chess`



<!-- # Quickstart

## Installation

This repo has been tested with tensorflow versions 2.9-2.15. If you want to use the newest tensorflow version (which handles all the cuda stuff for you), run `pip install -r requirements.txt`. Otherwise, you'll need to install tensorflow-addons, which requires tensorflow<=2.12. To use an older version, run `pip install -r trequirements_tf_2_9.txt` Note that only tensorflow versions <2.11 support Windows natively. Run ./init.sh to compile the protobuf files.

## Data preparation

In order to start a training session you first need to download training data from https://storage.lczero.org/files/training_data/. Several chunks/games are packed into a tar file, and each tar file contains an hour worth of chunks. Preparing data requires the following steps:

```
wget https://storage.lczero.org/files/training_data/training-run1--20200711-2017.tar
tar -xzf training-run1--20200711-2017.tar
```

## Training pipeline

Now that the data is in the right format one can configure a training pipeline. This configuration is achieved through a yaml file, see `training/tf/configs/example.yaml`:

The configuration is pretty self explanatory, if you're new to training I suggest looking at the [machine learning glossary](https://developers.google.com/machine-learning/glossary/) by google. Now you can invoke training with the following command:

```bash
./train.py --cfg configs/example.yaml --output /tmp/mymodel.txt
```

This will initialize the pipeline and start training a new neural network. You can view progress by invoking tensorboard:

```bash
tensorboard --logdir leelalogs
```

If you now point your browser at localhost:6006 you'll see the trainingprogress as the trainingsteps pass by. Have fun!

## Restoring models

The training pipeline will automatically restore from a previous model if it exists in your `training:path` as configured by your yaml config. For initializing from a raw `weights.txt` file you can use `training/tf/net_to_model.py`, this will create a checkpoint for you.

## Supervised training

Generating trainingdata from pgn files is currently broken and has low priority, feel free to create a PR.



# What's new here?

I've added a few features and modules to the training pipeline. Please direct any questions to the Leela Discord server.


## Architectural improvements
Replacing the smolgen layer used by the second, third, and fourth models in the BT series, we're now using learned relative position encodings.

## Quality of life
There are three quality of life improvements: a progress bar, new metrics, and pure attention code

Progress bar: A simple progress bar implemented in the Python `rich` module displays the current steps (including part-steps if the batches are split) and the expected time to completion.

Pure attention: The pipeline no longer contains any code from the original ResNet architecture. This makes for clearer yamls and code. The protobuf has been updated to support smolgen, input gating, and the square relu activation function.

## More metrics

I've added train value accuracy and train policy accuracy for the sake of completeness and to help detect overfitting. The speed difference is negligible. There are also three new losses metrics to evaluate policy. The cross entropy we are currently using is probably still the best for training, though we could try instead to turn the task into a classification problem, effectively using one-hot vectors at the targets' best moves, though this would run the risk of overfitting.

Thresholded policy accuracies: the thresholded policy accuracy @x% is the percent of moves for which the net has policy at least x% at the move the target thinks is best.

Reducible policy loss is the amount of policy loss we can reduce, i.e., the policy loss minus the entropy of the policy target.

The search policy loss is designed to loosely describe how long it would take to find the best move in the average position. It is implemented as the average of the multiplicative inverses of the network's policies at the targets' top moves, or one over the harmonic mean of those values. This is not too accurate since the search algorithm will often give up on moves the network does not like unless they provide returns that the network can immediately recognize. -->



