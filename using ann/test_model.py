import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load label encoder
label_encoder = joblib.load("pose_label_encoder.pkl")
class_names = [str(c) for c in label_encoder.classes_]

# Load trained model
model = load_model("pose_cnn_model.h5")

# Adjust input shape if needed (model expects 132 features)
expected_input_shape = model.input_shape[-1]
actual_input_shape = X_test.shape[1]

if actual_input_shape < expected_input_shape:
    # Pad with zeros
    padding = expected_input_shape - actual_input_shape
    X_test = np.pad(X_test, ((0, 0), (0, padding)), mode='constant')
elif actual_input_shape > expected_input_shape:
    # Truncate extra features
    X_test = X_test[:, :expected_input_shape]

# Predict
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluation
acc = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)

print("Accuracy:", acc)
print("\nClassification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
