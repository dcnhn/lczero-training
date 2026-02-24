# Logged Training Metrics on Tensorboard
> Note: Not all logged signals/metrics are described in this table. Only the most relevant and commonly used metrics are listed below.

| Plot Name | Description |
|------------|------------|
| Policy Accuracy | Overall top-1 policy prediction accuracy across all positions. |
| Policy Entropy | Entropy of the predicted policy distribution (measures confidence/sharpness). |
| Policy Loss | Primary supervised policy loss (e.g., cross-entropy). |
| Thresholded Policy Accuracy @ 1  | Proportion of positions where P(GT move)>1% |
| Thresholded Policy Accuracy @ 2  | Proportion of positions where P(GT move)>2% |
| Thresholded Policy Accuracy @ 5  | Proportion of positions where P(GT move)>5% |
| Thresholded Policy Accuracy @ 10 | Proportion of positions where P(GT move)>10% |
| Value Accuracy | Classification accuracy of the value head (win/draw/loss). |
| MSE Loss | Mean Squared Error loss (for value regression target). |
| Value Winner Loss | Loss for predicting the final game outcome (winner head). |
| Value Q Loss | Loss for Q-value prediction head. |
| Value ST Loss | Combined short-term value loss. |
| Value ST Err Loss | Short-term value regression error loss. |
| Value Err L | Value prediction error loss (e.g., regression error). |
| Value Cat L | Categorical value loss component, only relevant for the 240M model. |
| Value ST Cat Loss | Short-term categorical value loss, only relevant for the 240M model. |
| Moves Left Loss | Loss for predicting the number of moves remaining (if modeled). |
