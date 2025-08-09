import cv2
import mediapipe as mp
import csv

# 你可以自行修改此處標籤名稱，代表該影片所屬手語
label = "rabbit"  # 這邊改成你影片對應的標籤名稱

# 影片檔名 (與標籤對應)
video_path = "rabbit7.mp4"  # 改成你想處理的影片檔名

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils

# 輸出 CSV 檔案，以 append 模式新增數據
csv_file = open("gesture_data.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("影片讀取完成或錯誤，總共處理幀數:", frame_count)
        break

    frame_count += 1
    
    # 將 BGR 轉為 RGB 供 MediaPipe 使用
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks_list = []

        for hand_landmarks in results.multi_hand_landmarks:
            data_row = []
            for lm in hand_landmarks.landmark:
                data_row.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(data_row)
        
        # 若偵測少於兩隻手，補零
        while len(landmarks_list) < 2:
            landmarks_list.append([0.0] * 63)

        combined_features = landmarks_list[0] + landmarks_list[1]
        # 附加標籤（最後一欄）
        combined_features.append(label)

        # 自動寫入CSV
        csv_writer.writerow(combined_features)

    # 加上可視化骨架與示意
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # 顯示影片+骨架
    cv2.imshow("Video Gesture Recording", frame)

    # 你可以按 'q' 鍵提前結束錄製，也可自動跑完整個影片
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("使用者提前結束")
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
