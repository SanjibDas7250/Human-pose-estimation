import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time
from math import sqrt

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def findPose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img):
        lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                lmList.append((id, cx, cy, cz))
        return lmList

    def get_detection_confidence(self):
        return self.results.pose_landmarks if self.results else 0

def euclidean(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_input_type():
    print("Select input type:")
    print("1. Webcam")
    print("2. Video file")
    print("3. Images directory")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == '1':
        return 'webcam', 0
    elif choice == '2':
        path = input("Enter video file path: ").strip()
        return 'video', path
    elif choice == '3':
        path = input("Enter image directory path: ").strip()
        return 'images', path
    else:
        print("Invalid choice.")
        exit(1)

def main():
    input_type, path = get_input_type()
    detector = PoseDetector()
    img_count = 0
    jump_baseline = None
    prev_right_hand = None
    jump_cooldown = 0
    pushup_state = None
    pushup_timer = 0

    if not os.path.exists("pose_dataset/images"):
        os.makedirs("pose_dataset/images")
    csv_file = open("pose_dataset/keypoints.csv", mode='a', newline='')
    csv_writer = csv.writer(csv_file)

    if input_type == 'webcam':
        cap = cv2.VideoCapture(path)
        get_frame = lambda: cap.read()
    elif input_type == 'video':
        cap = cv2.VideoCapture(path)
        get_frame = lambda: cap.read()
    else:
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        frame_iter = iter(files)
        get_frame = lambda: (True, cv2.imread(next(frame_iter, None)))

    while True:
        success, img = get_frame()
        if not success or img is None:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        action = ""
        if lmList and len(lmList) > 30:
            try:
                get = lambda i: lmList[i][1:4]
                lw, rw = get(15), get(16)
                ls, rs = get(11), get(12)
                lh, rh = get(23), get(24)
                lk, rk = get(25), get(26)
                li, ri = lmList[19][1:3], lmList[20][1:3]
                nose_y = lmList[0][2]

                hip_y = (rh[1] + lh[1]) // 2
                knee_y = (rk[1] + lk[1]) // 2
                shoulder_y = (rs[1] + ls[1]) // 2
                z_diff = ls[2] - rs[2]

                if z_diff > 0.1:
                    action = "Side View Left"
                elif z_diff < -0.1:
                    action = "Side View Right"
                else:
                    if abs(lk[1] - rk[1]) > 80 and hip_y > shoulder_y:
                        action = "Lunge"
                    elif abs(lw[0] - ls[0]) > 100 and abs(rw[0] - rs[0]) > 100:
                        action = "T-Pose"
                    elif abs(lw[0] - ls[0]) > 100:
                        action = "Wave Left"
                    elif abs(rw[0] - rs[0]) > 100:
                        action = "Wave Right"
                    elif rw[1] < rs[1] and lw[1] < ls[1]:
                        action = "Both Hands Up"
                    elif rw[1] < rs[1]:
                        action = "Right Hand Up"
                    elif lw[1] < ls[1]:
                        action = "Left Hand Up"
                    elif prev_right_hand is not None and abs(rw[0] - prev_right_hand) > 30:
                        action = "Unknown"
                    elif jump_baseline is None:
                        jump_baseline = hip_y
                    elif hip_y < jump_baseline - 40 and jump_cooldown == 0:
                        action = "Jump"
                        jump_cooldown = 20
                    elif hip_y > shoulder_y + 40 and knee_y < hip_y + 30:
                        action = "Sit"
                    elif hip_y < knee_y - 20 and shoulder_y < hip_y:
                        action = "Stand"
                    elif abs(shoulder_y - hip_y) < 40:
                        if nose_y > shoulder_y + 40 and pushup_state != "down":
                            pushup_state = "down"
                            pushup_timer = 10
                        elif nose_y < shoulder_y - 10 and pushup_state == "down" and pushup_timer > 0:
                            action = "Push-Up"
                            pushup_state = "up"
                            pushup_timer = 0
                    if action == "":
                        if euclidean(li, ri) < 70:
                            action = "Clap"
                    if action == "":
                        if (abs(lw[0] - lh[0]) < 50 and abs(lw[1] - lh[1]) < 50) and \
                           (abs(rw[0] - rh[0]) < 50 and abs(rw[1] - rh[1]) < 50):
                            action = "Hands on Hip"

                if jump_cooldown > 0:
                    jump_cooldown -= 1
                if pushup_timer > 0:
                    pushup_timer -= 1
                prev_right_hand = rw[0]

                keypoints = []
                for _, x, y, _ in lmList:
                    keypoints.extend([x, y])
                keypoints.append(action)

                csv_writer.writerow(keypoints)
                img_name = f"pose_dataset/images/img_{img_count}.jpg"
                cv2.imwrite(img_name, img)
                img_count += 1

                cv2.putText(img, f"Label: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            except Exception as e:
                print(f"Error during landmark processing: {e}")

        cv2.imshow("Pose Collection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if input_type != 'images':
        cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Data collection finished.")

if __name__ == "__main__":
    main()
