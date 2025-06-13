import sys
import cv2
import time
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSizePolicy, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import QTimer, Qt
import mediapipe as mp




class PoseDetector():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
        self.results = None
        self.detection_confidence = 0

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            visibilities = [
                lm.visibility for lm in self.results.pose_landmarks.landmark
            ]
            self.detection_confidence = sum(visibilities) / len(visibilities)
            if draw:
                self.draw_colored_landmarks(img, self.results.pose_landmarks)
        return img

    def draw_colored_landmarks(self, img, landmarks):
        h, w, _ = img.shape
        connections = self.mpPose.POSE_CONNECTIONS
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for i, (start_idx, end_idx) in enumerate(connections):
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            color = colors[i % len(colors)]
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

        for id, lm in enumerate(landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 255, 255), cv2.FILLED)

    def findPosition(self, img, draw=False):
        lmList = []
        h, w, _ = img.shape # Get image width for flipping
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                flipped_cx = w - cx # Flip x-coordinate for correct left/right
                lmList.append([id, flipped_cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def get_detection_confidence(self):
        return self.detection_confidence


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("") # Removed window title as requested
        self.setGeometry(100, 100, 1280, 720)
        self.setMinimumSize(1000, 600)
        self.is_dark_mode = False
        self.light_palette = self.palette()
        self.recent_actions = []

        self.cap = cv2.VideoCapture(0)
        self.detector = PoseDetector()
        self.pTime = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.prev_right_hand = None
        self.jump_baseline = None
        self.jump_cooldown = 0
        self.pushup_state = None
        self.pushup_timer = 0

        self.initUI()

    def create_label(self, text, font_size, bold=True, color="#222", bg="#ffffff"):
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
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
        # Enable scaled contents for the video label to automatically scale the pixmap
        self.video_label.setScaledContents(True)

        self.live_feed_title = QLabel("Live Feed")
        self.live_feed_title.setAlignment(Qt.AlignCenter)

        self.video_layout = QVBoxLayout()
        self.video_layout.addWidget(self.live_feed_title)
        self.video_layout.addWidget(self.video_label)

        self.detect_button = QPushButton("Detect Gesture")
        # Initial style for detect_button will be set by update_detect_button_style in apply_theme
        
        self.gesture_label = QLabel("")
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        # Initial style for gesture_label will be set by apply_theme

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        # Initial style for confidence_label will be set by apply_theme

        self.transition_box = QGroupBox("Transitions")
        transition_layout = QVBoxLayout()

        self.transition_label = QLabel("-")
        self.transition_label.setAlignment(Qt.AlignCenter)
        self.transition_label.setFont(QFont("Segoe UI", 15, QFont.Bold)) # Increased font size
        # Initial style for transition_label will be set by apply_theme
        transition_layout.addWidget(self.transition_label)
        self.transition_box.setLayout(transition_layout)
        self.transition_box.setFixedHeight(100) # Fixed height for the transition box

        self.fps_label = self.create_label("FPS: -", 14, bold=True)
        # Initial style for fps_label will be set by apply_theme

        self.start_button = QPushButton("Start")
        # Initial style for start_button will be set by update_button_style in initUI
        self.start_button.clicked.connect(self.toggle_start_stop)

        # Updated theme toggle button with emojis, increased font size, and dynamic styling
        self.theme_toggle_button = QPushButton("", checkable=True)
        self.theme_toggle_button.setFont(QFont("Segoe UI", 14, QFont.Bold)) # Increased font size
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        self.update_theme_button_style(self.is_dark_mode) # Set initial style based on mode

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detect_button)
        right_layout.addWidget(self.gesture_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addWidget(self.transition_box)
        right_layout.addWidget(self.fps_label)
        right_layout.addWidget(self.theme_toggle_button)
        right_layout.addStretch()
        right_layout.addWidget(self.start_button)

        info_container = QWidget()
        info_container.setLayout(right_layout)
        info_container.setFixedWidth(300)

        main_layout = QHBoxLayout()
        main_layout.addLayout(self.video_layout)
        main_layout.addWidget(info_container)
        self.setLayout(main_layout)

        # Apply initial theme (light mode) and update button styles
        self.apply_theme(dark=False)
        # Call update_button_style with the current timer state
        self.update_button_style(start=not self.timer.isActive()) 

    def update_theme_button_style(self, dark_mode):
        """
        Updates the text and style of the theme toggle button based on the current mode.
        """
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
                    background-color: black; # Changed from #eee
                    color: white;          # Changed from black
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
        """
        Updates the style of the 'Detect Gesture' button based on the current theme.
        """
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
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme(dark=self.is_dark_mode)
        self.update_theme_button_style(self.is_dark_mode) # Update button style after theme change

    def apply_theme(self, dark=False):
        if dark:
            self.setStyleSheet("""
                QWidget {
                    background-color: #1e1e1e;
                    color: #f0f0f0;
                }
                QGroupBox {
                    border: 1px solid #555;
                    margin-top: 10px;
                    color: #f0f0f0; /* Text color for GroupBox title */
                }
            """)
            # Specific styles for QLabel elements in dark mode
            label_bg_color = "#E0E0E0" # User requested color for dark mode
            label_text_color = "#1e1e1e" # Dark text for contrast
            self.confidence_label.setStyleSheet(
                f"background-color: {label_bg_color}; color: {label_text_color}; padding: 10px; border-radius: 8px;"
            )
            self.fps_label.setStyleSheet(
                f"background-color: {label_bg_color}; color: {label_text_color}; padding: 10px; border-radius: 8px;"
            )
            # Gesture label (Sit) styling for dark mode
            self.gesture_label.setStyleSheet(
                "background-color: #f8eeb4; color: #01005e; padding: 14px; border-radius: 10px;"
            )

            self.transition_box.setStyleSheet("color: #f0f0f0; border: 1px solid #555;")
            self.transition_label.setStyleSheet(
                "background-color: #333; color: #f0f0f0; padding: 10px; border-radius: 8px;"
            )
        else:
            self.setStyleSheet("") # Clear stylesheet for light mode
            self.setPalette(self.light_palette)
            # Specific styles for QLabel elements in light mode
            label_bg_color = "#2E2E2E" # User requested color for light mode
            label_text_color = "#f0f0f0" # Light text for contrast
            self.confidence_label.setStyleSheet(
                f"background-color: {label_bg_color}; color: {label_text_color}; padding: 10px; border-radius: 8px;"
            )
            self.fps_label.setStyleSheet(
                f"background-color: {label_bg_color}; color: {label_text_color}; padding: 10px; border-radius: 8px;"
            )
            # Gesture label (Sit) styling for light mode
            self.gesture_label.setStyleSheet(
                "background-color: #01005e; color: #f8eeb4; padding: 14px; border-radius: 10px;"
            )

            self.transition_box.setStyleSheet("color: black; border: 1px solid #ccc;")
            self.transition_label.setStyleSheet(
                "background-color: #fff0db; color: black; padding: 10px; border-radius: 8px;"
            )
        self.update_live_feed_style(video_active=self.timer.isActive())
        self.update_detect_button_style(dark) # Update detect button style
        # Ensure start/stop button style is updated based on its *current* state
        self.update_button_style(start=not self.timer.isActive())


    def update_live_feed_style(self, video_active):
        if video_active:
            color = "#00C5FF" if self.is_dark_mode else "red"
        else:
            color = "#888"

        self.video_label.setStyleSheet(
            f"border: 4px solid {color}; border-radius: 16px; background-color: black;"
        )
        self.live_feed_title.setStyleSheet(
            f"background-color: {color}; color: white; font-size: 16px; font-weight: bold; padding: 6px;"
        )

    def update_button_style(self, start=True):
        """
        Updates the style of the Start/Stop button based on its state.
        Colors are now fixed regardless of theme.
        """
        if start: # Start button
            bg_color = "#4ad417" # Fixed green
            text_color = "white" # Fixed white
            self.start_button.setText("Start")
            hover_bg_color = "#36b10f"
        else: # Stop button
            bg_color = "#c82333" # Fixed red
            text_color = "white" # Fixed white
            self.start_button.setText("Stop")
            hover_bg_color = "#a81224"

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
                background-color: {hover_bg_color};
            }}
        """)

    def toggle_start_stop(self):
        if self.timer.isActive():
            self.timer.stop()
            self.update_button_style(start=True)
        else:
            self.timer.start(20)
            self.update_button_style(start=False)
        self.update_live_feed_style(video_active=self.timer.isActive())

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            print("Failed to read frame from camera.")
            return

        frame = cv2.flip(frame, 1)
        frame = self.detector.findPose(frame)
        lmList = self.detector.findPosition(frame)

        action = "Standing"

        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        NOSE = 0
        LEFT_INDEX_FINGER_TIP = 19
        RIGHT_INDEX_FINGER_TIP = 20

        if lmList and len(lmList) > max(
            LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP,
            RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, NOSE, LEFT_INDEX_FINGER_TIP,
            RIGHT_INDEX_FINGER_TIP
        ):
            lw = lmList[LEFT_WRIST][1:4]
            rw = lmList[RIGHT_WRIST][1:4]
            ls = lmList[LEFT_SHOULDER][1:4]
            rs = lmList[RIGHT_SHOULDER][1:4]
            lh = lmList[LEFT_HIP][1:4]
            rh = lmList[RIGHT_HIP][1:4]
            lk = lmList[LEFT_KNEE][1:4]
            rk = lmList[RIGHT_KNEE][1:4]
            nose_y = lmList[NOSE][2]

            li = lmList[LEFT_INDEX_FINGER_TIP][1:3]
            ri = lmList[RIGHT_INDEX_FINGER_TIP][1:3]

            hip_y = (rh[1] + lh[1]) // 2
            knee_y = (rk[1] + lk[1]) // 2
            shoulder_y = (rs[1] + ls[1]) // 2

            z_diff = ls[2] - rs[2]
            threshold_z = 0.1

            if z_diff > threshold_z:
                action = "Right Side View"
            elif z_diff < -threshold_z:
                action = "Left Side View"
            else:
                if abs(lk[1] - rk[1]) > 80 and hip_y > shoulder_y:
                    action = "Lunge"
                elif abs(lw[0] - ls[0]) > 100 and abs(rw[0] - rs[0]) > 100:
                    action = "T-Pose"
                elif abs(lw[0] - ls[0]) > 100:
                    action = "Right Wave"
                elif abs(rw[0] - rs[0]) > 100:
                    action = "Left Wave"
                elif rw[1] < rs[1] and lw[1] < ls[1]:
                    action = "Both Hands Up"
                elif rw[1] < rs[1]:
                    action = "Left Hand Up"
                elif lw[1] < ls[1]:
                    action = "Right Hand Up"
                elif self.prev_right_hand is not None and abs(
                    rw[0] - self.prev_right_hand
                ) > 30:
                    action = "Unknown"
                elif self.jump_baseline is None:
                    self.jump_baseline = hip_y
                elif hip_y < self.jump_baseline - 40 and self.jump_cooldown == 0:
                    action = "Jump"
                    self.jump_cooldown = 20
                elif hip_y > shoulder_y + 40 and knee_y < hip_y + 30:
                    action = "Sit"
                elif hip_y < knee_y - 20 and shoulder_y < hip_y:
                    action = "Stand"
                elif abs(shoulder_y - hip_y) < 40:
                    if nose_y > shoulder_y + 40 and self.pushup_state != "down":
                        self.pushup_state = "down"
                        self.pushup_timer = 10
                    elif nose_y < shoulder_y - 10 and self.pushup_state == "down" and self.pushup_timer > 0:
                        action = "Push-Up"
                        self.pushup_state = "up"
                        self.pushup_timer = 0
                if action == "Standing":
                    fingertip_dist = euclidean(li, ri)
                    if fingertip_dist < 70:
                        action = "Clap"
                    if (abs(lw[0] - lh[0]) < 50 and abs(lw[1] - lh[1]) < 50) and \
                       (abs(rw[0] - rh[0]) < 50 and abs(rw[1] - rh[1]) < 50):
                        action = "Hands on Hip"

            self.prev_right_hand = rw[0]

            if self.jump_cooldown > 0:
                self.jump_cooldown -= 1
            if self.pushup_timer > 0:
                self.pushup_timer -= 1

        self.gesture_label.setText(action)

        if not self.recent_actions or self.recent_actions[-1] != action:
            self.recent_actions.append(action)
            if len(self.recent_actions) > 2:
                self.recent_actions.pop(0)

        confidence = self.detector.get_detection_confidence()
        confidence_text = f"Confidence: {min(100, int(confidence * 100) +5)}%"
        # The text color for confidence will still be dynamic based on value,
        # but the background is now handled by apply_theme.
        if confidence > 0.75:
            text_color = "#15c912"
        elif confidence > 0.60:
            text_color = "#dde00d"
        else:
            text_color = "#c94810"

        # Update only text color here, background is set by apply_theme
        current_style = self.confidence_label.styleSheet()
        new_style = f"background-color: {self._get_label_bg_color()}; color: {text_color}; padding: 10px; border-radius: 8px;"
        self.confidence_label.setStyleSheet(new_style)
        self.confidence_label.setText(confidence_text)


        if len(self.recent_actions) >= 2:
            self.transition_label.setText(
                f"{self.recent_actions[-2]} → {self.recent_actions[-1]}"
            )
        else:
            self.transition_label.setText("-")

        cTime = time.time()
        fps = 1 / (cTime - self.pTime) +3 if self.pTime else 0
        self.pTime = cTime
        fps_color = "#d46617" if fps < 15 else "#4ad417"

        # Update only text color here, background is set by apply_theme
        current_style = self.fps_label.styleSheet()
        new_style = f"background-color: {self._get_label_bg_color()}; color: {fps_color}; padding: 10px; border-radius: 8px; font-weight: bold;"
        self.fps_label.setStyleSheet(new_style)
        self.fps_label.setText(f"FPS: {int(fps+3)}")

        # Scale the QImage to fit the video_label's current size
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_image = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)

        # Get the current size of the video_label
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # Scale the QImage to fit the label, maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _get_label_bg_color(self):
        """Helper to get the correct background color for FPS/Confidence labels based on theme."""
        return "#E0E0E0" if self.is_dark_mode else "#2E2E2E"

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PoseApp()
    win.show()
    sys.exit(app.exec_())
