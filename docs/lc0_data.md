# Terms used:

- **Chunk/Game:** A single training game, individual .gz file.
- **Chunk source:** A file (.tar or .gz) containing multiple chunks.
- **Frame/Record/Position:** A single training position inside a chunk.
- **Training tensor:** A single batch of inputs/outputs encoded in NN format for one training step.

Incoming data comes as chunks, but for the training we need frames from different games.

For further information about LCZero, see the [official documentation](https://github.com/LeelaChessZero/lczero-training/tree/master/docs).


# Data Format

The LCZero training data exists in multiple format versions. This documentation focuses on **V6**, which is the most recent and recommended format.

The training data is processed by `tf/chunkparser.py`, which converts raw V6 data into a 5-element tuple: `(planes, probs, winner, best_q, plies_left)`.

When interpreted as NumPy arrays, each training example has the following structure:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `planes` | `(112, 64)` | float32 | Board state as 112 feature planes, each 8×8 (flattened to 64). The original 104 planes are augmented with 8 additional planes for castling rights, side to move, rule 50 count, and board edge detection. |
| `probs` | `(1858,)` | float32 | Policy probabilities for all possible moves (corresponds to `float probabilities[1858]` in the [`V6TrainingData` C++ struct](https://github.com/LeelaChessZero/lc0/blob/13474cce7b83e7497f11b2d3439284a32573f5ce/src/trainingdata/trainingdata_v6.h)). |
| `winner` | `(3,)` | float32 | Game outcome from the current player's perspective: win, draw, loss probabilities. |
| `best_q` | `(3,)` | float32 | Position value after search (Q-value), also as win, draw, loss probabilities. |
| `plies_left` | scalar | float32 | Estimated number of plies remaining until game end. |

For more details, see the [official training tuple documentation](https://github.com/LeelaChessZero/lczero-training/blob/master/docs/training_tuple.md).

# Rescoring

The raw training data from LCZero is stored in **V6 format**, but this training pipeline requires a custom **V7 format**. The `rescore_file()` function in `tf/chunkparser.py` performs this conversion by computing additional training targets.

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