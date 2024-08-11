# ECG Anomaly Detection using Temporal Convolutional Networks (TCNs) and Darts

## Overview

This project aims to detect anomalies in ECG signals using Temporal Convolutional Networks (TCNs) with the Darts library. The ECG data is sourced from the ECG5000 dataset, which is preprocessed and scaled before being used to train a TCN-based model for identifying abnormal patterns in the signals.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Anomaly Detection](#anomaly-detection)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction

This project uses the ECG5000 dataset containing labeled ECG signals (normal and abnormal). The objective is to develop and train a Temporal Convolutional Network (TCN) to identify anomalies in ECG signals effectively. The model's performance is evaluated using various metrics and visualizations.

## Requirements

The following Python libraries are required:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `darts`
- `pytorch_lightning`
- `scikit-learn`
- `torch`

Install the necessary libraries using:

bash
pip install numpy pandas matplotlib seaborn darts pytorch_lightning scikit-learn torch

##Data Preparation
Loading Data:
The data is loaded from ECG5000_TRAIN.txt and ECG5000_TEST.txt. The files are combined and saved as Combined_data.csv.

python
Copy code
import pandas as pd

train_data = pd.read_csv('ECG5000_TRAIN.txt', header=None, delimiter=' ')
test_data = pd.read_csv('ECG5000_TEST.txt', header=None, delimiter=' ')
combined_data = pd.concat([train_data, test_data], axis=0)
combined_data.to_csv('Combined_data.csv', index=False)

Data Extraction:
Normal and abnormal ECG signals are extracted and saved into normal_data.csv and abnormal_data.csv.

python
Copy code
normal_data = combined_data[combined_data[0] == 0]
abnormal_data = combined_data[combined_data[0] == 1]
normal_data.to_csv('normal_data.csv', index=False)
abnormal_data.to_csv('abnormal_data.csv', index=False)
Scaling:
RobustScaler is used to scale the training, validation, and test data.

python
Copy code
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(combined_data.iloc[:, 1:])
Time Series Conversion:
The scaled data is converted into TimeSeries objects using Darts.

python
Copy code
from darts import TimeSeries

series = TimeSeries.from_dataframe(pd.DataFrame(scaled_data, columns=['value']))
Model Training
Model Definition:
Define the Temporal Convolutional Network (TCN) model.

python
Copy code
from darts.models import TCNModel

model = TCNModel(
    input_chunk_length=50,
    output_chunk_length=30,
    kernel_size=3,
    num_filters=32,
    num_layers=3,
    dropout=0.2
)
Training:
Train the model using the Trainer class from pytorch_lightning.

python
Copy code
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=50, early_stop_callback=True)
trainer.fit(model, train_dataloader, val_dataloader)
Callback:
Custom callback for logging training and validation losses.

python
Copy code
from pytorch_lightning.callbacks import Callback

class LossLoggingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Log losses here
        pass
Anomaly Detection
Model Fitting:
Fit the TCN model using ForecastingAnomalyModel with a NormScorer.

python
Copy code
from darts.models import ForecastingAnomalyModel
from darts.metrics import NormScorer

anomaly_model = ForecastingAnomalyModel(model, scorer=NormScorer())
anomaly_model.fit(train_series)
Scoring:
Calculate anomaly scores and detect anomalies based on z-scores.

python
Copy code
scores = anomaly_model.score(test_series)
anomalies = scores > 3  # Example threshold
Visualization:
Plot anomaly scores and detected anomalies.

python
Copy code
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_series.time_index, test_series.values, label='ECG Signal')
plt.plot(scores.time_index, scores.values, label='Anomaly Scores')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Detected Anomalies')
plt.legend()
plt.show()
Evaluation
Metrics:
Compute MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

python
Copy code
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(true_values, predicted_values)
rmse = mean_squared_error(true_values, predicted_values, squared=False)
Anomaly Detection:
Assess performance by plotting detected anomalies and calculating z-scores.

python
Copy code
# Example plot
plt.figure(figsize=(12, 6))
plt.plot(test_series.time_index, test_series.values, label='ECG Signal')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Detected Anomalies')
plt.legend()
plt.show()
Results
The results include:

Visualizations: Normal and abnormal ECG signals, histograms and density plots of signal amplitudes.
Anomaly Scores: Plots of anomaly scores and detected anomalies on test data.
Training/Validation Losses: Plots showing training and validation losses over epochs.
