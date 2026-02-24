## High-Level Overview of Training Components

The training process in LCZero is organized around three main components:

- **train.py script:** This script serves as the main entry point for training. It loads the configuration file, prepares the training and test datasets, and orchestrates the overall training workflow.

- **TFProcess class:** This class encapsulates the core training logic. It is responsible for building and initializing the neural network, setting up the training loop, managing optimizers, logging (e.g., TensorBoard), and saving checkpoints. TFProcess also handles multi-GPU training and ensures the execution of training steps.

- **ChunkParser class:** ChunkParser (and its helper classes) is responsible for loading and preprocessing the training data. It reads the compressed chunk files, extracts individual training positions (frames), and assembles them into batches for training. The data is shuffled, filtered, and formatted as needed for the neural network.
