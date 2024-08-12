ECG Anomaly Detection using Temporal Convolutional Networks (TCNs) and Darts
================================

This project aims to detect anomalies in ECG signals using Temporal Convolutional Networks (TCNs) with the Darts library. The ECG data is sourced from the ECG5000 dataset, which is preprocessed and scaled before being used to train a TCN-based model for identifying abnormal patterns in the signals.
The system consists of the following components:

1. **Data Loading**: Loads the ECG data from a CSV file and preprocesses it.
2. **Model Training**: Trains a TCN model on the normal ECG data to learn the patterns and trends.
3. **Anomaly Detection**: Uses the trained model to detect anomalies in the ECG data.
4. **Visualization**: Visualizes the anomaly scores and detected anomalies.

**Code Structure**
-------------------

The code is organized into the following sections:

### Libraries

* Import necessary libraries, including:
	+ Darts: A Python library for time series forecasting and anomaly detection.
	+ Pandas: A Python library for data manipulation and analysis.
	+ NumPy: A Python library for numerical computing.
	+ Matplotlib: A Python library for data visualization.
	+ Seaborn: A Python library for data visualization based on Matplotlib.

### Data Loading

* Load the ECG data from a CSV file named `Combined_data.csv`.
* Preprocess the data by:
	+ Removing the label column.
	+ Scaling the data using a RobustScaler to reduce the effect of outliers.

### Model Training

* Define a TCN model with the following architecture:
	+ Input chunk length: 50
	+ Output chunk length: 30
	+ Kernel size: 3
	+ Number of filters: 32
	+ Number of layers: 3
	+ Dropout rate: 0.2
* Train the model on the normal ECG data using the Adam optimizer and a learning rate of 0.001.
* Define an early stopping callback to stop training when the validation loss stops improving.

### Anomaly Detection

* Define an anomaly detection model using the trained TCN model and a NormScorer.
* Calculate anomaly scores for the test data using the anomaly detection model.
* Detect anomalies by thresholding the anomaly scores using a z-score threshold of 3.

### Visualization

* Visualize the anomaly scores and detected anomalies using Matplotlib and Seaborn.
* Plot the ECG data with normal and anomalous segments using different colors.
* Highlight detected anomalies using red markers.

**Functions**
-------------

### `calculate_anomaly_scores_by_chunk`

* Calculate anomaly scores for each chunk of data.
* Compute z-scores and detect anomalies for each chunk.
* Return the anomaly scores and detected anomalies for each chunk.

### `plot_ecg_with_anomalies`

* Plot the ECG data with normal and anomalous segments.
* Highlight detected anomalies using red markers.
* Return the plot.

### `plot_ecg_chunk_with_normal_and_anomalous_segments`

* Plot a chunk of ECG data with normal and anomalous segments.
* Highlight detected anomalies using red markers.
* Return the plot.

**Results**
----------

The code produces the following results:

* Anomaly scores for the test data.
* Detected anomalies in the test data.
* Visualizations of the anomaly scores and detected anomalies.


**Future Work**
--------------

* Improve the performance of the anomaly detection model by tuning the hyperparameters.
* Experiment with different anomaly detection algorithms and techniques.
* Integrate the anomaly detection system with a real-time ECG monitoring system.
