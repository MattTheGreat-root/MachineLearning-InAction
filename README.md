# MachineLearning-InAction

A hands-on learning repository exploring core machine learning concepts, computer vision, natural language processing, and financial modeling through practical notebooks and scripts.

## Overview

This repository is a personal collection of experiments and implementations designed to bridge theory and practice. It covers foundational ML algorithms built from scratch, image processing techniques, stock market volatility prediction, character-level language modeling, and basic PyTorch workflows.
<br/>
Each notebook is written in a self-contained, educational style with explanations, code, and visualizations, making it suitable for reviewing concepts or using as a quick reference.

## Modules

### Basics
Implements fundamental machine learning concepts from scratch to build intuition.
- **Bias vs. Variance**: Demonstrates the bias-variance trade-off using KNN with varying k values.
- **KNN-Ensemble**: A custom dataset and KNN ensemble implementation.
- **Perceptron**: Simple Perceptron algorithm for binary classification.
- **Regularization**: L1 and L2 regularization applied to linear models to control overfitting.

### ImageProcessing
Explores image manipulation and basic computer vision tasks using OpenCV and custom logic.
- **Alpha channel** handling and compositing.
- **Blob detection** to identify regions of interest.
- **Contrast adjustment** and histogram manipulation.
- **Event-based processing** for interactive editing.
- **Opening & cropping** images programmatically.
- **SIFT** feature extraction and matching.
- **Simple OCR** pipeline for digit/character recognition.
- **Skin tracking** using color thresholds.
- **Thresholding** techniques (global, adaptive, Otsu).
- **Webcam capture** and real-time processing.

### MarketVolatility
Predicts stock market volatility using machine learning. Contains a complete pipeline:
- Fetches historical price data (default: AAPL, 2014–2024).
- Engineers volatility-related features.
- Trains a Random Forest regressor with GridSearchCV for hyperparameter tuning.
- Provides both a Python script (`main.py`) and an identical Jupyter notebook for exploratory analysis. <br/>
Checkout the local README for more details.

### NLP
- **c-rnn.ipynb**: Character-level recurrent neural network trained on textual data (e.g., Shakespeare). Demonstrates text generation using a simple RNN built from scratch (no deep learning frameworks aside from numpy) to understand sequence modeling.

### Torch
Introduction to PyTorch with two foundational model types.
- **Linear Regression**: A minimal PyTorch training loop for linear regression, covering tensors, gradients, and optimization.
- **Classifier**: A feed-forward neural network for classification, illustrating dataset loading, training/validation loops, and evaluation.

<br/>
The repository is still under active development and may contain incomplete implementations.