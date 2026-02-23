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
![Successful debug training run](docs/finished_debug_train.png)

# Training

> **Note:** This repository includes a small sample dataset in `playground_data/` for testing purposes only. These files are used to verify that the environment is set up correctly (see [Verify Environment Setup](#verify-environment-setup)). Once setup is confirmed, this folder can be safely deleted to free up space.

## Data preparation
In order to start a training session you first need to download training data from https://storage.lczero.org/files/training_data/.
The **LCZero database** contains multiple versions of the training data format, reflecting changes and improvements over time.<br>
> ⚠️ **Recommendation (as of 2026-02-17):** use training data generated in **2024 or later**, as older datasets may rely on deprecated formats or lack newer features expected by the current training pipeline.
> ⚠️ **Important:** Data preprocessing is a **required** step. Training will fail with a "memory layout mismatch" error if you attempt to use the data directly without performing the preprocessing.

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



## Training Configurations

Training is configured through YAML files located in `tf/configs/`. For a detailed specification of all available parameters, see the **[Training Configuration Reference](docs/training_config.md)**.

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



