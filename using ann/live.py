import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSizePolicy, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import QTimer, Qt
from keras.models import load_model # For loading the Keras model
import joblib # For loading the label encoder
import tensorflow as tf
# Load trained model and label encoder
# IMPORTANT: Ensure 'pose_cnn_model.h5' and 'pose_label_encoder.pkl' are in the same directory
# as this script, or provide the full path to them.
try:
    model = load_model("pose_cnn_model.h5")
    label_encoder = joblib.load("pose_label_encoder.pkl")
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    print("Please ensure 'pose_cnn_model.h5' and 'pose_label_encoder.pkl' are in the correct path.")
    # Exit or handle gracefully if model/encoder are critical
    sys.exit(1)


# MediaPipe initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.0, min_tracking_confidence=0.0) # Set to 0.0 to ensure detection always attempts
mp_drawing = mp.solutions.drawing_utils

# Function to extract 132 landmark features (from your main code)
def extract_landmarks(results):
    """
    Extracts 132 landmark features (x, y, z, visibility) from MediaPipe pose results.
    Returns None if no pose landmarks are detected.
    """
    if not results.pose_landmarks:
        return None
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(landmarks, dtype=np.float32)


class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Action Detection")
        self.setGeometry(100, 100, 1280, 720)
        self.setMinimumSize(1000, 600)
        self.is_dark_mode = False
        self.light_palette = self.palette() # Store default light palette

        # Video capture and timer for frame updates
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video stream. Please check camera connection.")
            sys.exit(1) # Exit if camera cannot be opened
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Variables for FPS calculation and action tracking (from your main code)
        self.prev_time = 0
        self.last_actions = [] # Stores last two actions for transition display

        self.initUI()
        # Apply initial theme (light mode) and update button styles
        self.apply_theme(dark=False)
        self.update_button_style(start=True, dark_mode=False) # Ensure start button has correct initial style
        self.update_live_feed_style(video_active=self.timer.isActive()) # Initial style for video feed border

    def create_label(self, text, font_size, bold=True, color="#222", bg="#ffffff"):
        """Helper function to create styled QLabel."""
        label = QLabel(text)
        font = QFont("Segoe UI", font_size)
        font.setBold(bold)
        label.setFont(font)
        label.setStyleSheet(f"""
            background-color: {bg};
            color: {color};
            padding: 10px;
            border-radius: 8px;
        """)
        label.setAlignment(Qt.AlignCenter)
        return label

    def initUI(self):
        """Initializes the main UI components and layout."""
        # Video feed display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setText("Camera Feed (Click Start)") # Initial text

        # Title for the live feed
        self.live_feed_title = QLabel("Live Feed")
        self.live_feed_title.setAlignment(Qt.AlignCenter)

        # Layout for video feed and its title
        self.video_layout = QVBoxLayout()
        self.video_layout.addWidget(self.live_feed_title)
        self.video_layout.addWidget(self.video_label)

        # "Detect Gesture" button (placeholder, as actual detection is continuous)
        self.detect_button = QPushButton("Action Prediction Active")
        self.detect_button.setEnabled(False) # Disable as prediction is continuous
        # Initial style will be set by update_detect_button_style in apply_theme

        # Label to display the predicted gesture/action
        self.gesture_label = QLabel("Waiting for pose...")
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.gesture_label.setStyleSheet("""
            background-color: white;
            color: #000;
            padding: 14px;
            border-radius: 10px;
        """)

        # Label to display confidence of the prediction
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.confidence_label.setStyleSheet(
            "background-color: #555; color: white; padding: 10px; border-radius: 8px;"
        )

        # Group box and label for displaying action transitions
        self.transition_box = QGroupBox("Transition")
        transition_layout = QVBoxLayout()
        self.transition_label = QLabel("-")
        self.transition_label.setAlignment(Qt.AlignCenter)
        self.transition_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.transition_label.setStyleSheet(
            "background-color: #fff0db; color: black; padding: 10px; border-radius: 8px;"
        )
        transition_layout.addWidget(self.transition_label)
        self.transition_box.setLayout(transition_layout)
        self.transition_box.setFixedHeight(100) # Fixed height for the transition box

        # Label to display Frames Per Second (FPS)
        self.fps_label = self.create_label("FPS: -", 14, bold=True)

        # Start/Stop button for the video feed and prediction
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_start_stop)

        # Theme toggle button (Light/Dark mode)
        self.theme_toggle_button = QPushButton("", checkable=True)
        self.theme_toggle_button.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        self.update_theme_button_style(self.is_dark_mode) # Set initial style based on mode

        # Right-side layout for all control and info widgets
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detect_button)
        right_layout.addWidget(self.gesture_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addWidget(self.transition_box)
        right_layout.addWidget(self.fps_label)
        right_layout.addWidget(self.theme_toggle_button)
        right_layout.addStretch() # Pushes widgets to the top
        right_layout.addWidget(self.start_button)

        # Container widget for the right layout to control its width
        info_container = QWidget()
        info_container.setLayout(right_layout)
        info_container.setFixedWidth(300)

        # Main horizontal layout combining video and info sections
        main_layout = QHBoxLayout()
        main_layout.addLayout(self.video_layout)
        main_layout.addWidget(info_container)
        self.setLayout(main_layout)


    def update_theme_button_style(self, dark_mode):
        """Updates the text and style of the theme toggle button."""
        if dark_mode:
            self.theme_toggle_button.setText("☀️ Switch to Light") # White sun emoji
            self.theme_toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #333;
                    color: white;
                    padding: 12px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 6px;
                }
                QPushButton:checked {
                    background-color: #555;
                }
            """)
        else:
            self.theme_toggle_button.setText("🌙 Switch to Dark") # Black moon emoji
            self.theme_toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: black;
                    color: white;
                    padding: 12px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 6px;
                }
                QPushButton:checked {
                    background-color: #ccc;
                }
            """)

    def update_detect_button_style(self, dark_mode):
        """Updates the style of the 'Action Prediction Active' button."""
        if dark_mode:
            bg_color = "#febe7e" # Inversed from light mode
            text_color = "#25161b" # Inversed from light mode
        else:
            bg_color = "#25161b" # User requested color for light mode
            text_color = "#febe7e" # User requested color for light mode

        self.detect_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                padding: 14px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
        """)

    def toggle_theme(self):
        """Toggles between light and dark themes."""
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme(dark=self.is_dark_mode)
        self.update_theme_button_style(self.is_dark_mode) # Update button style after theme change

    def apply_theme(self, dark=False):
        """Applies the selected theme to the application."""
        if dark:
            self.setStyleSheet("""
                QWidget {
                    background-color: #1e1e1e;
                    color: #f0f0f0;
                }
                QLabel {
                    background-color: #2e2e2e;
                    color: #ffffff;
                }
                QGroupBox {
                    border: 1px solid #555;
                    margin-top: 10px;
                    color: #f0f0f0; /* Text color for GroupBox title */
                }
            """)
            # Specific styles for transition box and label in dark mode
            self.transition_box.setStyleSheet("color: #f0f0f0; border: 1px solid #555;")
            self.transition_label.setStyleSheet(
                "background-color: #333; color: #f0f0f0; padding: 10px; border-radius: 8px;"
            )
        else:
            self.setStyleSheet("") # Clear stylesheet for light mode
            self.setPalette(self.light_palette) # Revert to default light palette
            # Specific styles for transition box and label in light mode
            self.transition_box.setStyleSheet("color: black; border: 1px solid #ccc;")
            self.transition_label.setStyleSheet(
                "background-color: #fff0db; color: black; padding: 10px; border-radius: 8px;"
            )
        self.update_live_feed_style(video_active=self.timer.isActive()) # Update video border color
        self.update_detect_button_style(dark) # Update detect button style
        self.update_button_style(start=not self.timer.isActive(), dark_mode=dark) # Update start/stop button style

    def update_live_feed_style(self, video_active):
        """Updates the border and title style of the live video feed."""
        if video_active:
            color = "#00C5FF" if self.is_dark_mode else "red"
        else:
            color = "#888" # Grey when inactive

        self.video_label.setStyleSheet(
            f"border: 4px solid {color}; border-radius: 16px; background-color: black;"
        )
        self.live_feed_title.setStyleSheet(
            f"background-color: {color}; color: white; font-size: 16px; font-weight: bold; padding: 6px;"
        )

    def update_button_style(self, start=True, dark_mode=False):
        """Updates the style of the Start/Stop button based on its state and current theme."""
        if start: # Start button state
            if dark_mode:
                bg_color = "#f3cba5" # Inversed from light mode
                text_color = "#453953" # Inversed from light mode
            else:
                bg_color = "#453953" # User requested color for light mode
                text_color = "#f3cba5" # User requested color for light mode
            self.start_button.setText("Start")
        else: # Stop button state
            bg_color = "#c82333" # Red for stop, consistent regardless of theme
            text_color = "white"
            self.start_button.setText("Stop")

        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                padding: 16px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: {'#36b10f' if start else '#a81224'}; /* Keep hover effect consistent */
            }}
        """)

    def toggle_start_stop(self):
        """Starts or stops the video feed and prediction timer."""
        if self.timer.isActive():
            self.timer.stop()
            self.video_label.setText("Camera Feed (Stopped)") # Display text when stopped
        else:
            self.timer.start(20) # Start timer to update every 20ms (approx 50 FPS)
            self.video_label.setText("") # Clear text when started
        self.update_button_style(start=not self.timer.isActive(), dark_mode=self.is_dark_mode)
        self.update_live_feed_style(video_active=self.timer.isActive())


    def update_frame(self):
        """
        Reads a frame from the camera, processes it using MediaPipe,
        predicts action with the loaded model, and updates the UI.
        This method integrates the core logic from your 'Main Code'.
        """
        success, frame = self.cap.read()
        if not success:
            print("Failed to read frame from camera.")
            # Optionally stop the timer if camera fails
            self.toggle_start_stop()
            return

        # Flip the frame horizontally for mirror effect (common for webcams)
        image = cv2.flip(frame, 1)
        
        # Create a temporary RGB image for MediaPipe processing.
        # MediaPipe's pose.process expects RGB.
        rgb_for_mediapipe = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_for_mediapipe)

        # Draw landmarks on the original BGR 'image'.F
        # mp_drawing.draw_landmarks modifies 'image' in-place.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the final 'image' (which now has landmarks and is BGR) to RGB
        # before displaying it with PyQt's QImage, which expects RGB.
        final_display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract landmarks and predict action (from your main code)
        landmarks = extract_landmarks(results)
        action_label = "No Pose Detected"
        confidence = 0

        if landmarks is not None and landmarks.shape[0] == 132:
            input_data = np.expand_dims(landmarks, axis=0)
            try:
                prediction = model.predict(input_data, verbose=0)
                class_id = np.argmax(prediction)
                confidence = int(np.max(prediction) * 100)
                action_label = label_encoder.inverse_transform([class_id])[0]
            except Exception as e:
                print(f"Error during model prediction: {e}")
                action_label = "Prediction Error"
                confidence = 0

            # Update last actions (no duplicates next to each other, from main code)
            if not self.last_actions or self.last_actions[-1] != action_label:
                self.last_actions.append(action_label)
                if len(self.last_actions) > 2:
                    self.last_actions.pop(0)
        else:
            action_label = "No Pose Detected"
            confidence = 0
            self.last_actions = [] # Clear last actions if no pose detected

        # FPS calculation (from your main code)
        curr_time = time.time()
        fps = int(1 / (curr_time - self.prev_time)) if self.prev_time != 0 else 0
        self.prev_time = curr_time

        # --- Update UI elements with processed data ---

        # Update Gesture/Action Label
        self.gesture_label.setText(action_label)

        # Update Confidence Label
        confidence_text = f"Confidence: {confidence}%"
        if confidence > 75: # Using percentage directly
            color = "#15c912" # Green
        elif confidence > 60:
            color = "#dde00d" # Yellow
        else:
            color = "#c94810" # Orange/Red
        self.confidence_label.setStyleSheet(
            f"background-color: white; color: {color}; padding: 10px; border-radius: 8px;"
        )
        self.confidence_label.setText(confidence_text)

        # Update Transition Label
        if len(self.last_actions) >= 2:
            self.transition_label.setText(
                f"{self.last_actions[-2]} → {self.last_actions[-1]}"
            )
        else:
            self.transition_label.setText("-")

        # Update FPS Label
        fps_color = "#d46617" if fps < 15 else "#4ad417" # Orange if low, green if good
        self.fps_label.setStyleSheet(f"""
            background-color: white;
            color: {fps_color};
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
        """)
        self.fps_label.setText(f"FPS: {int(fps)}")

        # Convert OpenCV image to QImage and display in QLabel
        h, w, ch = final_display_image.shape # Use the correctly converted RGB image
        qt_image = QImage(final_display_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        """Handles closing the application, releasing camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PoseApp()
    win.show()
    sys.exit(app.exec_())
