import cv2
import mediapipe as mp
import csv
from config import labels  # 從 config 讀取 labels

label_index = 0  # 預設標籤索引

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

csv_file = open("gesture_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

recording = False  # 用 r 開始/停止錄製

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks_list = []
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data_row = []
            for lm in hand_landmarks.landmark:
                data_row.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(data_row)

        while len(landmarks_list) < 2:
            landmarks_list.append([0.0] * 63)

        combined_features = landmarks_list[0] + landmarks_list[1]

        if recording:
            combined_features.append(labels[label_index])
            csv_writer.writerow(combined_features)

    # 顯示目前手勢標籤與錄製狀態
    cv2.putText(frame, f"Label: {labels[label_index]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Recording: {'ON' if recording else 'OFF'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if recording else (255, 0, 0), 2)

    cv2.imshow("MediaPipe Hands Double Hand", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in range(ord('1'), ord('1') + len(labels)):  # 動態按鍵控制切標籤
        label_index = key - ord('1')
    elif key == ord('r'):  # 切換連續錄製
        recording = not recording
    elif key == 32:  # 空白鍵 32，單次紀錄當前手勢資料
        if result.multi_hand_landmarks:
            # 直接寫入 CSV，附加標籤
            single_record = combined_features + [labels[label_index]]
            csv_writer.writerow(single_record)
            print(f"Single record saved: Label={labels[label_index]}")

csv_file.close()
cap.release()
cv2.destroyAllWindows()
