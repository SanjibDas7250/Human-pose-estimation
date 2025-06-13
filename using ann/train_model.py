import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import joblib

# 1. Load dataset
df = pd.read_csv("pose_dataset/keypoints.csv", header=None)

# 2. Separate features and labels
X = df.iloc[:, :-1].values  # Keypoints
y = df.iloc[:, -1].values   # Action labels

# 3. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 4. Save label encoder for inference
joblib.dump(label_encoder, "pose_label_encoder.pkl")

# 5. Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 6. Build classification model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

# 8. Save model
model.save("pose_action_classifier.h5")
print("Model and label encoder saved.")

# 9. Save X_test/y_test for later evaluation
X_test = X_val
y_test = y_val
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# 10. Optional: Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training History")
plt.grid(True)
plt.tight_layout()
plt.show()
