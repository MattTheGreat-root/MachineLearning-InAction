# MachineLearning-InAction

A personal learning repository covering machine learning, image processing with OpenCV, and natural language processing.

## Overview

This repository is a collection of Jupyter notebooks and Python scripts that explore a variety of topics, from foundational algorithms to practical applications. It serves as a hands-on playground to learn and experiment with different techniques.

## Repository Structure

MachineLearning-InAction/
├── Basics/ # Core ML algorithms from scratch
│ ├── BiasVsVariance/ # Bias-variance trade-off & KNN
│ ├── KNN-Ensemble/ # KNN ensemble with custom dataset
│ ├── Perceptron/ # Simple Perceptron implementation
│ └── Regularization/ # Regularization techniques (L1, L2)
├── ImageProcessing/ # Image processing & computer vision
│ ├── alphaChannel.ipynb # Alpha channel handling
│ ├── blobdetection.ipynb # Blob detection
│ ├── contrast.ipynb # Contrast adjustment
│ ├── event.ipynb # Event-based processing
│ ├── open&crop.ipynb # Image opening & cropping
│ ├── sift.ipynb # Scale-Invariant Feature Transform
│ ├── simpleOCR.ipynb # Simple Optical Character Recognition
│ ├── skintracker.ipynb # Skin tracking
│ ├── thresholding.ipynb # Image thresholding
│ └── webcam.ipynb # Webcam capture & processing
├── MarketVolatility/ # Stock volatility prediction
│ ├── data.py # Data fetching (AAPL, 2014-2024)
│ ├── main.py # Training script (RandomForest + GridSearchCV)
│ ├── volatility.ipynb # Notebook version of the model
│ └── requirements.txt # Dependencies
├── NLP/ # Natural Language Processing
│ └── c-rnn.ipynb # Character-level RNN
├── Torch/ # PyTorch experiments
│ ├── LinearRegression.ipynb # Linear Regression with PyTorch
│ └── classifier.ipynb # Classifier model
└── .gitignore

## Usage
Basics: Open any notebook in the Basics/ folder to see implementations of bias-variance, KNN ensembles, perceptron, and regularization.

Image Processing: Open notebooks in ImageProcessing/ to learn how to manipulate images, detect features, threshold, and perform OCR.

Market Volatility:

Run data.py to download stock data (default: AAPL 2014-2024).
Run main.py to train a RandomForest regressor with GridSearchCV.
Alternatively, open volatility.ipynb for an interactive version.
NLP: Open c-rnn.ipynb to train a character-level RNN on a Shakespeare dataset.

Torch: Open LinearRegression.ipynb or classifier.ipynb to see basic PyTorch workflows.

## Notes
The repository is still under active development and may contain incomplete implementations.

Some modules (e.g., MarketVolatility) include a local README with more details.