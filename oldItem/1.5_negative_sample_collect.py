import cv2
import mediapipe as mp
import csv
import os
from config import labels  # 讀取標籤列表，但這裡負樣本不用labels裡的正樣本標籤

NEGATIVE_LABEL = "negative"  # 自訂負樣本標籤

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 以追加模式開啟csv，若檔案不存在會新建（避免覆蓋）
csv_file = open("gesture_data.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

recording = False  # 按 r 開始/停止錄製

print("按 r 開始/停止錄製負樣本資料，按 q 離開程式")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data_row = []
            for lm in hand_landmarks.landmark:
                data_row.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(data_row)

    while len(landmarks_list) < 2:
        landmarks_list.append([0.0] * 63)

    combined_features = landmarks_list[0] + landmarks_list[1]

    if recording:
        # 負樣本標籤加在最後
        combined_features.append(NEGATIVE_LABEL)
        csv_writer.writerow(combined_features)

    # 顯示錄製狀態
    cv2.putText(frame, f"Recording Negative Samples: {'ON' if recording else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (255, 0, 0), 2)

    cv2.imshow("Negative Sample Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        recording = not recording
        print(f"{'開始' if recording else '停止'}錄製負樣本資料")

csv_file.close()
cap.release()
cv2.destroyAllWindows()
