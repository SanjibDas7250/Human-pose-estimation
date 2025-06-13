import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error

# 1. Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# 2. Expand input to match model input shape (if needed)
expected_shape = 132
if X_test.shape[1] != expected_shape:
    factor = expected_shape // X_test.shape[1]
    if expected_shape % X_test.shape[1] != 0:
        raise ValueError(f"Cannot tile input shape {X_test.shape} to match model input {expected_shape}")
    print(f"[INFO] Expanding X_test from {X_test.shape[1]} to {expected_shape}")
    X_test = np.tile(X_test, factor)

# Final input to model
X = X_test

# Convert one-hot labels to class indices
y_true = np.argmax(y_test, axis=1)

# 3. Load label encoder and class names
label_encoder = joblib.load("pose_label_encoder.pkl")
class_names = [str(cls) for cls in label_encoder.classes_]

# 4. Load the trained model
model = load_model("pose_cnn_model.h5")

# 5. Predict on test data
y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)

# 6. Evaluation
acc = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)

print("Accuracy:", acc)
#Mean Square root
mse = mean_squared_error(y_true, y_pred)
print("Test MSE:", mse)
#Classification Report
print("\nClassification Report:\n", report)
# 7. Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
