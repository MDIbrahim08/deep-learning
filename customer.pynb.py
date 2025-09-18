import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
# NOTE: The file path 'C:/Users/user/Desktop/genai/Churn_Modelling.csv' is specific to a local machine
# For Kaggle, the correct path is likely '../input/path_to_file/Churn_Modelling.csv'
df = pd.read_csv('C:/Users/ISHAQ/Desktop/Deep-Learning-main/Churn_Modelling.csv')

# Drop unnecessary columns
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

# Separate features (X) and target (y)
X = df.drop(columns=['Exited'])
y = df['Exited'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=X_train_scaled.shape[1])) # Use ReLU for hidden layers
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Use sigmoid for binary classification output

# Compile the model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, batch_size=50, epochs=100, verbose=1, validation_split=0.2)

# Make predictions
y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy of the model: {accuracy}")

# Plotting the loss and accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()