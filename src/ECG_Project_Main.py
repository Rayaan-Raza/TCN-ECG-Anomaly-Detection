# %% [markdown]
# # Libraries

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.ad import ForecastingAnomalyModel, KMeansScorer, NormScorer
from darts.ad.detectors import QuantileDetector
from darts.models import TCNModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.statistics import check_seasonality
from darts.metrics import mse
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim import Adam

# %% [markdown]
# # Loading Data

# %%
df_1 = pd.read_csv("D:/internship project/ECG5000_TRAIN.txt", delimiter='\s+', header=None)
df_2 = pd.read_csv("D:/internship project/ECG5000_TEST.txt", delimiter='\s+', header=None)

# %%
df = pd.concat([df_1, df_2], ignore_index=True)
df.to_csv('D:/internship project/Combined_data.csv', index=False, header=False)
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()
df.dropna(inplace=True)

# %% [markdown]
# # Extracting Normal, Abnormal Data

# %%
normal_data = df.loc[df[0] == 1]
abnormal_data = df.loc[df[0] != 1]
normal_data.to_csv('D:/internship project/normal data.csv', index=False)
abnormal_data.to_csv('D:/internship project/abnormal data.csv', index=False)

# %% [markdown]
# # Plots

# %%
ecg_signals = df.iloc[:, 1:]

plt.figure(figsize=(20, 30))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('ECG Signals')
plt.show()

# %%
ecg_signals = normal_data.iloc[:, 1:]

plt.figure(figsize=(20, 30))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Normal ECG Signals')
plt.show()

# %%
# Assuming normal_data contains the normal ECG signals
normal_signals = normal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_normal_signal = normal_signals.iloc[random.randint(0, len(normal_signals) - 1)]

# Assuming abnormal_data contains the abnormal ECG signals
abnormal_signals = abnormal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_abnormal_signal = abnormal_signals.iloc[random.randint(0, len(abnormal_signals) - 1)]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot the normal ECG signal
axs[0].plot(selected_normal_signal, label='Normal ECG Signal', color='blue')
axs[0].set_xlabel('Time (Sample Points)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Normal ECG Signal')
axs[0].legend()

# Plot the abnormal ECG signal
axs[1].plot(selected_abnormal_signal, label='Abnormal ECG Signal', color='red')
axs[1].set_xlabel('Time (Sample Points)')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Abnormal ECG Signal')
axs[1].legend()

# Display the plot
plt.show()


# %%
# Assuming ecg_signals contains the normal data
normal_signals = normal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_normal_signal = normal_signals.iloc[random.randint(0, len(normal_signals) - 1)]

# Assuming ecg_signals contains the abnormal data
abnormal_signals = abnormal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_abnormal_signal = abnormal_signals.iloc[random.randint(0, len(abnormal_signals) - 1)]

# Plotting both normal and abnormal signals on the same plot
plt.figure(figsize=(10, 5))
plt.plot(selected_normal_signal, label='Normal ECG Signal', color='blue')
plt.plot(selected_abnormal_signal, label='Abnormal ECG Signal', color='red')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Comparison of Normal and Abnormal ECG Signals')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.hist(normal_data, bins=75, alpha=0.7, label='Normal ECG Signal')
plt.hist(abnormal_data, bins=75, alpha=0.7, label='Abnormal ECG Signal')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.title('Histogram of ECG Signal Amplitudes')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 5))

# Plot density for normal ECG signals without labels in the legend
sns.kdeplot(normal_data, fill=True, alpha=0.5, color='blue', legend=False)

# Plot density for abnormal ECG signals without labels in the legend
sns.kdeplot(abnormal_data, fill=True, alpha=0.5, color='red', legend=False)

plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of ECG Signal Amplitudes')
plt.show()

# %%
# Combine all columns of normal and abnormal data into single series
normal_combined = normal_data.values.flatten()
abnormal_combined = abnormal_data.values.flatten()

# Plotting the density plots
plt.figure(figsize=(10, 5))
sns.kdeplot(normal_combined, label='Normal ECG Signal', fill=True, alpha=0.5, color='blue')
sns.kdeplot(abnormal_combined, label='Abnormal ECG Signal', fill=True, alpha=0.5, color='red')
plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of ECG Signal Amplitudes')
plt.legend()
plt.show()


# %%
plt.figure(figsize=(10, 5))
sns.heatmap(normal_data.iloc[:10, :], cmap='viridis')
plt.xlabel('Time (Sample Points)')
plt.ylabel('ECG Signal Index')
plt.title('Heatmap of Normal ECG Signals')
plt.show()

# %%
# Create a DataFrame for plotting
boxplot_data = pd.DataFrame({
    'Amplitude': pd.concat([selected_normal_signal, selected_abnormal_signal]),
    'Type': ['Normal'] * len(selected_normal_signal) + ['Abnormal'] * len(selected_abnormal_signal)
})

plt.figure(figsize=(20, 20))

# Basic Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x='Type', y='Amplitude', data=boxplot_data, palette='Set2')
plt.title('Basic Boxplot of ECG Signal Amplitudes')

# Detailed Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x='Type', y='Amplitude', data=boxplot_data, palette='Set2', showfliers=False)
sns.swarmplot(x='Type', y='Amplitude', data=boxplot_data, color='k', alpha=0.5, dodge=True)

# Add mean markers
mean_values = boxplot_data.groupby('Type')['Amplitude'].mean()
for i, mean in enumerate(mean_values):
    plt.scatter(x=i, y=mean, color='red', marker='D', s=100, label='Mean' if i == 0 else "", zorder=10)

plt.title('Detailed Boxplot of ECG Signal Amplitudes')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # Remove Label column

# %%
normal_data.drop(normal_data.columns[0], axis=1, inplace=True)
abnormal_data.drop(abnormal_data.columns[0], axis=1, inplace=True)

# %%
normal_data.head()

# %%
abnormal_data.head()

# %% [markdown]
# # Train, Validation and Test Split

# %%
# Split into training and remaining (test + validation)
train_data, temp_data = train_test_split(normal_data, test_size=0.3, random_state=42)

# Split the remaining data into test and validation
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(f"Training Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {val_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# %% [markdown]
# # Scaling

# %%
# Initialize RobustScaler
scaler = RobustScaler()

# Fit on training data and transform train, validation, and test sets
train_features_scaled = scaler.fit_transform(train_data)
val_features_scaled = scaler.transform(val_data)
test_features_scaled = scaler.transform(test_data)

# Convert scaled features back to DataFrames if needed
train_data_scaled = pd.DataFrame(train_features_scaled, columns=train_data.columns)
val_data_scaled = pd.DataFrame(val_features_scaled, columns=val_data.columns)
test_data_scaled = pd.DataFrame(test_features_scaled, columns=test_data.columns)

# %% [markdown]
# # Remove Index

# %%
# Remove index

train_data_scaled.reset_index(drop=True, inplace=True)
val_data_scaled.reset_index(drop=True, inplace=True)
test_data_scaled.reset_index(drop=True, inplace=True)

# %%
train_data_scaled.head()

# %%
val_data_scaled.head()

# %%
test_data_scaled.head()

# %%
#%% Printing infos
train_data_scaled.info()
val_data_scaled.info()
test_data_scaled.info()

# %% [markdown]
# # Adding TimeSeries

# %%
train_series = TimeSeries.from_dataframe(train_data_scaled)
val_series = TimeSeries.from_dataframe(val_data_scaled)
test_series = TimeSeries.from_dataframe(test_data_scaled)
train_series = train_series.astype(np.float32)
val_series = val_series.astype(np.float32)
test_series = test_series.astype(np.float32)

# %%
train_series.head(5)

# %%
val_series.head(5)

# %%
test_series.head(5)

# %% [markdown]
# # Model Architecture

# %%
class LossLoggingCallback(Callback):
    def __init__(self):
        self.metrics = {"epochs": [], "train_loss": [], "val_loss": []}
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            train_loss = train_loss.item()
            self.train_losses.append(train_loss)
            print(f"Train epoch end: recorded train loss {train_loss}")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", None)
        print("Validation Epoch End Callback Triggered")  # Debugging Line
        if val_loss is not None:
            val_loss = val_loss.item()
            self.val_losses.append(val_loss)
            print(f"Validation epoch end: recorded validation loss {val_loss}")

            # Append new metrics
            epoch = trainer.current_epoch
            self.metrics["epochs"].append(epoch)
            self.metrics["train_loss"].append(self.train_losses[-1] if self.train_losses else None)
            self.metrics["val_loss"].append(val_loss)


# Define the TCN model
ecg_model = TCNModel(
    input_chunk_length=50,
    output_chunk_length=30,
    kernel_size=3,
    num_filters=32,
    num_layers=3,
    dropout=0.2,
    optimizer_cls=Adam,
    optimizer_kwargs={"lr": 0.001},
    random_state=42
)

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=10,          # Number of epochs to wait for improvement
    mode="max"           # Mode should be 'min' for loss metrics
)

# Create the loss logging callback instance
loss_callback = LossLoggingCallback()

# Initialize the trainer with callbacks
trainer = Trainer(
    callbacks=[early_stopping_callback, loss_callback],
    max_epochs=100,
    logger=True,
    enable_progress_bar=True
)

# %%
ecg_model.fit(train_series, val_series=val_series, trainer=trainer)

# %%
# Save the model
ecg_model.save("ecg_model_darts.pth")

# Print debugging statement for successful save
print("Model saved successfully as 'ecg_model_darts.pth'")

# %%
# Load the model
loaded_ecg_model = TCNModel.load("ecg_model_darts.pth")

# Print debugging statement for successful load
if loaded_ecg_model:
    print("Model loaded successfully from 'ecg_model_darts.pth'")
else:
    print("Failed to load the model from 'ecg_model_darts.pth'")

# %%
# Make sure to have matching lengths for epochs, train_losses, and val_losses
epochs = range(len(loss_callback.train_losses))

# Ensure that the lengths match
num_train_epochs = len(loss_callback.train_losses)
num_val_epochs = len(loss_callback.val_losses)

# Adjust val_losses if it has more entries than train_losses
if num_val_epochs > num_train_epochs:
    loss_callback.val_losses = loss_callback.val_losses[:num_train_epochs]

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_callback.train_losses, label='Train Loss', marker='o')
plt.plot(epochs, loss_callback.val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Forecasting Anomaly Model

# %%
abnormal_features_scaled = scaler.transform(abnormal_data)
abnormal_data_scaled = pd.DataFrame(abnormal_features_scaled, columns=test_data.columns)

# %%
abnormal_series = TimeSeries.from_dataframe(abnormal_data_scaled)
abnormal_series = abnormal_series.astype(np.float32)

# %%
# instantiate the anomaly model with: one fitted model, and 3 scorers
anomaly_model = ForecastingAnomalyModel(
    model=ecg_model,
    scorer=[
        NormScorer(ord=1),
    ],
)

# %%
START = 0.1
anomaly_model.fit(train_series, start=START, allow_model_training=False, verbose=True, scorer=NormScorer)

# %%
# Calculate anomaly scores for the validation or test series
anomaly_scores, model_forecasting = anomaly_model.score(
    test_series, start=START, return_model_prediction=True, verbose=True
)
pred_start = model_forecasting.start_time()

# %%
# Extract anomaly scores from the result
anomaly_scores

# %%
# Extract the time index and values from the TimeSeries object
time_index = anomaly_scores.time_index
scores = anomaly_scores.values()

# Plot the anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(time_index, scores, label="Anomaly Scores")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores')
plt.legend()
plt.show()

# %%
mae(model_forecasting, test_series)

# %%
rmse(model_forecasting, test_series)

# %%
# Step 1: Calculate anomaly scores on the validation data
val_anomaly_scores, val_model_forecasting = anomaly_model.score(
    val_series, start=START, return_model_prediction=True, verbose=True
)
pred_start = model_forecasting.start_time()

# %%
val_anomaly_scores

# %%
# Convert to numpy arrays if they are not already
time_index = np.array(time_index)
scores = np.array(scores)

# Calculate z-scores
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score

# Define the threshold for anomaly detection
threshold = 3
anomalies = z_scores > threshold
anomalies

# %%
# Print the types and shapes of the variables
print(f"Type of time_index: {type(time_index)}")
print(f"Shape of time_index: {np.shape(time_index)}")
print(f"Type of scores: {type(scores)}")
print(f"Shape of scores: {np.shape(scores)}")
print(f"Type of anomalies: {type(anomalies)}")
print(f"Shape of anomalies: {np.shape(anomalies)}")

# Convert to numpy arrays if they are not already
if not isinstance(time_index, np.ndarray):
    time_index = np.array(time_index)
if not isinstance(scores, np.ndarray):
    scores = np.array(scores)
if not isinstance(anomalies, np.ndarray):
    anomalies = np.array(anomalies)

# Verify the shapes after conversion
print(f"Converted type of time_index: {type(time_index)}")
print(f"Shape of time_index after conversion: {np.shape(time_index)}")
print(f"Converted type of scores: {type(scores)}")
print(f"Shape of scores after conversion: {np.shape(scores)}")
print(f"Converted type of anomalies: {type(anomalies)}")
print(f"Shape of anomalies after conversion: {np.shape(anomalies)}")

# %%
# Flatten the scores and anomalies arrays to 1D
scores = scores.flatten()
anomalies = anomalies.flatten()

# Plot the anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(time_index, scores, label="Anomaly Scores")
plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores with Detected Anomalies')
plt.legend()
plt.show()

# %%
# Print the indices of detected anomalies
print("Anomaly indices:", np.where(anomalies)[0])

# %%
# Define the chunk size
chunk_size = 50  # Example chunk size

# %%
# Function to calculate anomaly scores for each chunk
def calculate_anomaly_scores_by_chunk(time_index, scores, chunk_size, threshold=3):
    num_chunks = len(time_index) // chunk_size
    if len(time_index) % chunk_size != 0:
        num_chunks += 1

    all_anomalies = []
    all_scores = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(time_index))
        
        # Extract chunk
        chunk_time_index = time_index[start_idx:end_idx]
        chunk_scores = scores[start_idx:end_idx]
        
        # Compute mean and std deviation for the chunk
        mean_score = np.mean(chunk_scores)
        std_dev_score = np.std(chunk_scores)
        
        # Compute z-scores and detect anomalies
        z_scores = (chunk_scores - mean_score) / std_dev_score
        anomalies = z_scores > threshold
        
        # Append results
        all_anomalies.append(anomalies)
        all_scores.append(chunk_scores)
        
    # Combine results
    combined_anomalies = np.concatenate(all_anomalies)
    combined_scores = np.concatenate(all_scores)
    
    return combined_anomalies, combined_scores

# %%
# Calculate anomaly scores by chunk
chunk_anomalies, chunk_scores = calculate_anomaly_scores_by_chunk(time_index, scores, chunk_size)


# %%
# Plot the anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(time_index, chunk_scores, label="Anomaly Scores")
plt.scatter(time_index[chunk_anomalies], chunk_scores[chunk_anomalies], color='red', label="Detected Anomalies")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores with Detected Anomalies (by Chunks)')
plt.legend()
plt.show()

# %%
# Print the indices of detected anomalies
print("Anomaly indices:", np.where(chunk_anomalies)[0])

# %%
# Example data: Replace these with your actual data
time_index = np.arange(len(scores))  # Assuming time index is sequential
normal_data = np.random.normal(0, 1, len(scores))  # Replace with actual normal ECG data
anomalous_data = np.random.normal(0, 1, len(scores))  # Replace with actual anomalous ECG data


# %%
# Function to plot ECG data with anomalies
def plot_ecg_with_anomalies(time_index, normal_data, anomalous_data, scores, anomalies, threshold=3):
    plt.figure(figsize=(14, 7))
    
    # Plot normal ECG data
    plt.plot(time_index, normal_data, label="Normal ECG Data", color='blue', alpha=0.5)
    
    # Plot anomalous ECG data
    plt.plot(time_index, anomalous_data, label="Anomalous ECG Data", color='orange', alpha=0.5)
    
    # Plot anomaly scores
    plt.plot(time_index, scores, label="Anomaly Scores", color='green', linestyle='--', alpha=0.7)
    
    # Highlight detected anomalies
    plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies", marker='x')
    
    plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ECG Data with Detected Anomalies')
    plt.legend()
    plt.show()

# %%
# Example usage
plot_ecg_with_anomalies(time_index, normal_data, anomalous_data, scores, chunk_anomalies)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Example data: Replace these with your actual data
time_index = np.arange(len(scores))  # Assuming time index is sequential
ecg_data = np.random.normal(0, 1, len(scores))  # Replace with your actual ECG data
chunk_start = 100  # Define start index for the chunk
chunk_end = 200  # Define end index for the chunk

# Define the threshold for anomalies (e.g., z-score > 3)
threshold = 3

# Compute z-scores (assuming you already have scores and anomalies)
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score
anomalies = z_scores > threshold

# Extract the chunk of data
chunk_time_index = time_index[chunk_start:chunk_end]
chunk_ecg_data = ecg_data[chunk_start:chunk_end]
chunk_scores = scores[chunk_start:chunk_end]
chunk_anomalies = anomalies[chunk_start:chunk_end]

# Function to plot a chunk of ECG data with normal and anomalous segments
def plot_ecg_chunk_with_normal_and_anomalous_segments(time_index, ecg_data, scores, anomalies, threshold):
    plt.figure(figsize=(14, 7))
    
    # Plot segments
    current_color = 'green'
    for i in range(len(time_index) - 1):
        # Switch color when an anomaly is detected
        if anomalies[i]:
            plt.plot(time_index[i:i+2], ecg_data[i:i+2], color='red', alpha=0.8)
        else:
            plt.plot(time_index[i:i+2], ecg_data[i:i+2], color=current_color, alpha=0.8)
    
    # Plot anomaly scores
    plt.plot(time_index, scores, label="Anomaly Scores", color='blue', linestyle='--', alpha=0.7)
    
    # Highlight detected anomalies
    plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies", marker='x')
    
    plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
    plt.xlabel('Time')
    plt.ylabel('ECG Value')
    plt.title('ECG Chunk with Normal and Anomalous Segments')
    plt.legend()
    plt.show()

# Plot the chunk
plot_ecg_chunk_with_normal_and_anomalous_segments(chunk_time_index, chunk_ecg_data, chunk_scores, chunk_anomalies, threshold)


# %%
print("Time Index Chunk:", chunk_time_index)
print("ECG Data Chunk:", chunk_ecg_data)
print("Anomalies Chunk:", chunk_anomalies)

# %%
if np.any(chunk_anomalies):
    print("Anomalies detected in chunk.")
else:
    print("No anomalies detected in chunk.")


# %%
plt.figure(figsize=(14, 7))
for i in range(len(chunk_time_index) - 1):
    plt.plot(chunk_time_index[i:i+2], chunk_ecg_data[i:i+2], color='red' if chunk_anomalies[i] else 'green', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('ECG Value')
plt.title('ECG Chunk with Normal and Anomalous Segments')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Example data: Replace these with your actual data
time_index = np.arange(len(scores))  # Assuming time index is sequential
ecg_data = np.random.normal(0, 1, len(scores))  # Replace with your actual ECG data

# Define the threshold for anomalies (e.g., z-score > 3)
threshold = 1

# Compute z-scores (assuming you already have scores and anomalies)
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score
anomalies = z_scores > threshold

# Function to plot ECG data with normal and anomalous segments
def plot_ecg_with_anomalies(time_index, ecg_data, anomalies):
    plt.figure(figsize=(14, 7))
    
    # Plot the entire ECG data first
    plt.plot(time_index, ecg_data, color='green', alpha=0.3)  # Plot all data in light green for context
    
    # Overlay the anomalous segments in red
    start_idx = 0
    while start_idx < len(time_index):
        end_idx = start_idx
        while end_idx < len(time_index) and anomalies[end_idx] == anomalies[start_idx]:
            end_idx += 1
        
        # Plot the segment
        if anomalies[start_idx]:
            plt.plot(time_index[start_idx:end_idx], ecg_data[start_idx:end_idx], color='red', alpha=0.8)
        
        # Move to the next segment
        start_idx = end_idx
    
    plt.xlabel('Time')
    plt.ylabel('ECG Value')
    plt.title('ECG Data with Normal and Anomalous Segments')
    plt.show()

# Plot the ECG data
plot_ecg_with_anomalies(time_index, ecg_data, anomalies)




