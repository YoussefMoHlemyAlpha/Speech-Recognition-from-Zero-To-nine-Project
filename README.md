# Speech Commands Recognition Project
This project is a neural network-based approach to recognizing spoken commands using the Google Speech Commands dataset. The project aims to classify spoken digits (zero to nine) using extracted MFCC (Mel-frequency cepstral coefficients) features and data augmentation techniques.

# Project Objective
The goal of this project is to develop a model that can recognize and classify single-digit spoken commands using an artificial neural network (ANN) model. The model is designed to handle background noise and minor pitch variations using data augmentation techniques.

# Dataset Information
The dataset used in this project is the Google Speech Commands dataset, which contains thousands of .wav files representing various spoken words, including digits. The dataset is downloaded using the Kaggle API.

# Data Preprocessing Steps
#### Data Augmentation: Augmentation techniques include adding random noise and pitch shifting to make the model more robust to variations in audio signals.
#### Feature Extraction: MFCC features are extracted from the audio files to provide meaningful representations of the audio signals.
#### Data Encoding: The labels are encoded into categorical form for multi-class classification(0 : 9).
#### Data Splitting: The dataset is split into training (80%), validation (10%), and test sets (10%).
# Methodology :
## 1. Data Acquisition
The dataset used in this project is the Google Speech Commands dataset, which contains thousands of .wav files of spoken words, including the digits zero through nine. The dataset is downloaded from Kaggle using the Kaggle API.

## 2. Data Preprocessing
#### Audio Loading: Each audio file is loaded using the librosa library, which enables flexible loading and manipulation of audio files.
#### Data Augmentation: To enhance model robustness and address potential overfitting, data augmentation is applied to each audio signal.
#### Random Noise: Random Gaussian noise is added to the audio signals to simulate various recording conditions.
#### Pitch Shifting: The pitch of the audio is shifted to help the model generalize across different voice pitches.
#### Feature Extraction: Mel-frequency cepstral coefficients (MFCCs) are extracted from the raw audio. MFCCs are widely used in audio processing as they effectively capture the characteristics of the human voice, making them suitable features for speech recognition tasks.
##  3. Data Encoding and Splitting
Label Encoding: The spoken words (target classes) are encoded into integer labels and then converted into a one-hot encoded format, as required for multi-class classification.
##### Dataset Splitting: The dataset is split into training, validation, and test sets, with an 80-10-10 ratio to ensure sufficient data for model training while reserving some data for validation and testing.
## 4. Feature Engineering
#### Padding/Truncation: Since audio files vary in length, the extracted MFCCs are either truncated or zero-padded to a fixed length. This ensures consistency in input dimensions for the model.
#### Feature Stacking: Augmented MFCC features (original, noisy, and pitch-shifted) are combined, providing more feature-rich data for training.
## 5. Model Design
Architecture: A feed-forward artificial neural network (ANN) is used due to its simplicity and effectiveness for MFCC-based audio classification. The network comprises fully connected layers with ReLU activation functions and is structured as follows:
#### Input Layer: Accepts flattened MFCC features.
#### Hidden Layers: Dense layers with Batch Normalization and Dropout for regularization.
#### Output Layer: Softmax activation to predict probabilities for each class.
#### Regularization: Dropout layers are incorporated to reduce overfitting, and Batch Normalization is applied for faster convergence and stable training.
## 6. Model Training
#### Compilation: The model is compiled with adam optimizer and categorical_crossentropy loss function, suitable for multi-class classification.
#### Callbacks: To optimize training, callbacks are used:
#### Early Stopping: Monitors validation loss and stops training if it does not improve for a set number of epochs.
#### Learning Rate Reduction: Reduces the learning rate if the validation loss plateaus, helping the model converge to a better minimum.
## 7. Model Evaluation
#### Metrics: The model's performance is assessed on the test set using accuracy, precision, recall, and F1 score to capture overall and per-class performance.
#### Confusion Matrix: A confusion matrix is generated to analyze the model’s performance for each class, identifying any misclassifications and understanding the model’s strengths and weaknesses.
## 8. Visualization of Training and Evaluation
Training Curves: Plots for training and validation accuracy and loss are generated to visualize model learning and monitor for overfitting or underfitting.
Confusion Matrix Visualization: The confusion matrix is visualized using a heatmap to analyze the model’s predictive accuracy across the different command classes.
# Code Instructions :
## Prerequisites :
Ensure the following Python libraries are installed:

1-os

2-numpy

3-pandas

4-librosa

5-matplotlib

6-scikit-learn

7-tensorflow

8-seaborn

9-kaggle

You can install the required libraries with:
by Copying These codes 
pip install numpy pandas librosa matplotlib scikit-learn tensorflow seaborn kaggle
Running the Code
Set Up Kaggle API: Make sure your Kaggle API credentials are stored as environment variables (KAGGLE_USERNAME and KAGGLE_KEY) to download the dataset.
Download and Preprocess Dataset:
The code automatically downloads and unzips the dataset to the specified folder.

# Run the Model:
Train the model and evaluate its performance on the test set. The model outputs metrics, including accuracy, precision, recall, and F1 score.

# Visualizations:
Training and validation accuracy and loss are visualized to monitor the model's learning process.
A confusion matrix is generated for a more detailed look at the classification performance.
![image](https://github.com/user-attachments/assets/31e70956-5446-429a-b6c5-e7da0c366f28)
![image](https://github.com/user-attachments/assets/3b71a0e2-c37d-44d6-b768-ee23e49ac0a2)


# Evaluation Metrics
The project uses the following metrics for model evaluation:

1- Accuracy.

2- Precision.

3- Recall.

4- F1 Score.

The metrics are calculated on the test set after training the model.

# Dependencies and Installation Instructions
Install the dependencies listed above, and then run the code in a Python environment. For best results, use a Jupyter notebook or an IDE that supports visualizations, such as VSCode or PyCharm.
