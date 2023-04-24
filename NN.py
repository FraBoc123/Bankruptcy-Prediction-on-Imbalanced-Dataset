import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, classification_report, precision_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

df=pd.read_csv("data.csv")
df.head()

# Dividing Data and Labels
y = df['Bankrupt?']
X = df.drop(['Bankrupt?'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)



# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import additional libraries
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import LeakyReLU

# Apply SMOTE to balance the dataset
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Create the neural network with more hidden layers and LeakyReLU activation
model = Sequential()
# model.add(Dense(128, input_shape=(X_train_resampled.shape[1],)))
# model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
# model.add(Dense(64))
# model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
# model.add(Dense(32))
# model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# Replaced LeakyReLU with ReLU activation functions
model.add(Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate class weights
class_weights = {0: 1., 1: (len(y_train_resampled) - sum(y_train_resampled)) / sum(y_train_resampled)}

# Train the model with the resampled data
history = model.fit(X_train_resampled, y_train_resampled, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], class_weight = class_weights)

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype('int32')

# Print the recall score and classification report
print("Recall:", recall_score(y_test, y_pred))
print("Classification Report Neural Network:")
print(classification_report(y_test, y_pred))

import numpy as np
from sklearn.metrics import f1_score

def evaluate_threshold_f1(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba > threshold).astype('int32')
    f1 = f1_score(y_true, y_pred)
    return f1

thresholds = np.linspace(0.1, 0.9, 9)  # You can adjust the range and step size
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    f1 = evaluate_threshold_f1(y_test, y_pred_proba, threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

y_pred = (y_pred_proba > best_threshold).astype('int32')

#
# # Define a function to calculate recall and precision for a given threshold
# def evaluate_threshold(y_true, y_pred_proba, threshold):
#     y_pred = (y_pred_proba > threshold).astype('int32')
#     recall = recall_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     return recall, precision
#
# # Test different threshold values
# thresholds = np.linspace(0.1, 0.9, 9)  # You can adjust the range and step size
# results = []
#
# for threshold in thresholds:
#     recall, precision = evaluate_threshold(y_test, y_pred_proba, threshold)
#     results.append({'threshold': threshold, 'recall': recall, 'precision': precision})
#
# # Print the results
# results_df = pd.DataFrame(results)
# print(results_df)
#
#
# def plot_recall_precision(y_true, y_pred_proba):
#     thresholds = [x / 100 for x in range(10, 90, 5)]
#     recalls = []
#     precisions = []
#
#     for threshold in thresholds:
#         y_pred = (y_pred_proba > threshold).astype('int32')
#         recall = recall_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred)
#         recalls.append(recall)
#         precisions.append(precision)
#
#     plt.plot(thresholds, recalls, marker='o', label='Recall')
#     plt.plot(thresholds, precisions, marker='x', label='Precision')
#     plt.xlabel('Threshold')
#     plt.ylabel('Score')
#     plt.title('Recall and Precision vs. Threshold')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
# # Use the function to plot the results
plot_recall_precision(y_test, y_pred_proba)