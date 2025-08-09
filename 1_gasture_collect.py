import cv2
import mediapipe as mp
import csv
from config import labels  # 你的 label 字串列表

label_index = 0  # 預設標籤索引

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 改成 append 模式，若檔案不存在就建立，已存在就追加
csv_file = open("gesture_data.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

recording = False  # 用 'r' 鍵開始/停止錄製

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

        # 如果偵測到的手少於2隻，補成兩隻（避免欄位不齊）
        while len(landmarks_list) < 2:
            landmarks_list.append([0.0] * 63)

        combined_features = landmarks_list[0] + landmarks_list[1]

        if recording:
            # 在寫入資料列時，在最後加入對應標籤
            combined_features.append(labels[label_index])
            csv_writer.writerow(combined_features)
            csv_file.flush()  # 寫入即時生效

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
    elif key == ord('r'):  # 切換錄製狀態
        recording = not recording
    elif key == 32:  # 空白鍵 (32)，單次錄製當前手勢資料
        if result.multi_hand_landmarks:
            single_record = combined_features + [labels[label_index]]
            csv_writer.writerow(single_record)
            csv_file.flush()
            print(f"Single record saved: Label={labels[label_index]}")

csv_file.close()
cap.release()
cv2.destroyAllWindows()
