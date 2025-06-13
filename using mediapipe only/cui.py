import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.results = None
        self.detection_confidence = 0

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            visibilities = [lm.visibility for lm in self.results.pose_landmarks.landmark]
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
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def get_detection_confidence(self):
        return self.detection_confidence

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    pTime = 0
    prev_right_hand = None
    jump_baseline = None
    jump_cooldown = 0
    pushup_state = None
    pushup_timer = 0
    recent_actions = []

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        action = ""

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

        if lmList and len(lmList) > max(LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER,
                                       LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                                       NOSE, LEFT_INDEX_FINGER_TIP, RIGHT_INDEX_FINGER_TIP):

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
                action = "Side View Left"
            elif z_diff < -threshold_z:
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
                elif action == "":
                    fingertip_dist = euclidean(li, ri)
                    if fingertip_dist < 70:
                        action = "Clap"
                if action == "":
                    if (abs(lw[0] - lh[0]) < 50 and abs(lw[1] - lh[1]) < 50) and \
                       (abs(rw[0] - rh[0]) < 50 and abs(rw[1] - rh[1]) < 50):
                        action = "Hands on Hip"

            prev_right_hand = rw[0]
            if jump_cooldown > 0:
                jump_cooldown -= 1
            if pushup_timer > 0:
                pushup_timer -= 1

        if action:
            cv2.putText(img, action, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if len(recent_actions) == 0 or recent_actions[-1] != action:
                recent_actions.append(action)
                if len(recent_actions) > 2:
                    recent_actions.pop(0)

        for i, a in enumerate(recent_actions):
            cv2.putText(img, f"{a}", (10, 200 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        confidence = detector.get_detection_confidence()
        cv2.putText(img, f"Confidence: {int(confidence * 100)}%", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Pose Detection", img)

        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty("Pose Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)

if __name__ == "__main__":
    main()
