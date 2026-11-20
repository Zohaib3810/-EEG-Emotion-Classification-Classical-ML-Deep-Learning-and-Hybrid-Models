# -EEG-Emotion-Classification-Classical-ML-Deep-Learning-and-Hybrid-Models
This repository contains an EEG-based emotion recognition system classifying brain signals into five emotional states: Funny, Sad, Scary, Relaxed, and Enjoyed.
## Project Overview

EEG (Electroencephalography) records electrical activity of the brain. Because EEG data is complex, non-stationary, and highly subjective, classifying emotional states from raw signals is challenging. This project addresses the challenge using three distinct modeling pipelines:

1. **Feature Engineering + Classical ML**: Extracts statistical time-domain and frequency-domain (FFT) features from multi-channel inputs and applies a Random Forest Classifier.
2. **Deep Learning on Raw Signals (1D-CNN)**: Feeds raw multi-channel time-series data directly into a 1D Convolutional Neural Network to automatically learn spatial-temporal representations.
3. **Multi-Branch Hybrid Model**: A deep neural network architecture that combines both extracted statistical/spectral features and raw sequence inputs using parallel branches, concatenating them prior to final classification.

## Repository Structure

- `eeg_dataset/`: Directory containing the multi-channel EEG recordings in CSV format, structured by advertising/content type and emotional classes.
- `notebooks/`: Jupyter/Google Colab notebooks showcasing data extraction, model training, evaluation metrics, and dimensionality reduction.
- `README.md`: Project documentation and guides.

## Dataset

The dataset consists of multi-channel EEG time-series signals. Since raw file shapes vary due to recording duration and channel count (e.g., 20,000 timesteps across 22 channels, or 180,000 timesteps across 24 channels), the pipeline includes preprocessing steps to identify and group consistent shapes for training sequence-based models.

The emotions are mapped as follows:
- Funny: 0
- Sad: 1
- Scary: 2
- Relaxed: 3
- Enjoyed: 4

## Feature Engineering

For the classical Machine Learning branch and the hybrid branch, features are extracted for each EEG channel independently:
- **Time-Domain**: Mean, Standard Deviation, Minimum, and Maximum of the signal amplitude.
- **Frequency-Domain**: Fourier Transform (FFT) average magnitude and standard deviation to capture spectral power distribution.

## Models and Architectures

### 1. Random Forest Classifier
- Input: 144-dimensional engineered feature vector.
- Model: Scikit-learn Random Forest Classifier.

### 2. 1D Convolutional Neural Network
- Input: Raw EEG signal array with shape `(samples, timesteps, channels)`.
- Layers: Conv1D, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Dense.

### 3. Multi-Branch Hybrid Model
- Branch A: Dense layers processing engineered features.
- Branch B: Conv1D and Global Pooling processing raw signal sequences.
- Merge: Concatenation layer joining both representation branches, followed by dense dropout regularization and a final Softmax output layer.

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib scipy
```

### Data Processing & Training

To load the dataset, run feature extraction, and train the pipelines, execute the notebook cells sequentially. The data ingestion script automatically filters files with mismatched dimensions to ensure stable tensor operations during deep learning training.

```python
# Run the extraction and preprocessing pipeline
import numpy as np
# (Ensure your dataset zip file is uploaded and extracted as 'eeg_dataset')
```

## Evaluation Metrics

Models are validated using standard classification metrics:
- Accuracy
- Classification Report (Precision, Recall, F1-Score per emotion)
- Confusion Matrices
- Principal Component Analysis (PCA) to visualize 2D clusters of extracted features across different emotion classes.
