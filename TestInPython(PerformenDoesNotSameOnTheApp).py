import cv2
import mediapipe as mp
import numpy as np
import coremltools as ct
import sys

def main():
    model_path = "signLanguage.mlmodel"
    try:
        model = ct.models.MLModel(model_path)
    except Exception as e:
        print(f"無法載入 Core ML 模型: {e}")
        sys.exit(1)
    print("Core ML 模型載入成功")

    input_names = [i.name for i in model.input_description._fd_spec]
    print(f"模型輸入名稱：{input_names}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.80,
        min_tracking_confidence=0.80
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機，請確認授權")
        sys.exit(1)
    print("攝影機開啟成功")

    confidence_threshold = 0.95

    stable_frame_threshold = 5  # 連續幀數閾值（例如5幀）
    consecutive_count = 0
    last_prediction = None
    stable_prediction = "No confident prediction"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("讀取畫面失敗，結束")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data_row = []
                for lm in hand_landmarks.landmark:
                    data_row.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(data_row)

            while len(landmarks_list) < 2:
                landmarks_list.append([0.0] * 63)

            combined_features = landmarks_list[0] + landmarks_list[1]

            input_name = input_names[0]
            input_data = np.array(combined_features, dtype=np.float32).reshape(1, -1)

            try:
                outputs = model.predict({input_name: input_data})
                class_probs = outputs.get('classProbability', None)
                if class_probs is not None:
                    max_prob = max(class_probs.values()) if isinstance(class_probs, dict) else np.max(class_probs)
                    predicted_label = outputs.get('classLabel', 'Unknown')

                    if max_prob >= confidence_threshold:
                        # 若與上一幀結果相同，累計計數器，否則重置計數器
                        if predicted_label == last_prediction:
                            consecutive_count += 1
                        else:
                            consecutive_count = 1
                            last_prediction = predicted_label

                        # 連續到達閾值，認定手勢穩定
                        if consecutive_count >= stable_frame_threshold:
                            stable_prediction = predicted_label
                    else:
                        # 信心不足，重置計數器與結果
                        consecutive_count = 0
                        stable_prediction = "Confidence too low"
                        last_prediction = None
                else:
                    stable_prediction = outputs.get('classLabel', 'Unknown')
                    consecutive_count = 0
                    last_prediction = None

                print(f"模型預測（穩定）: {stable_prediction}")

            except Exception as e:
                print(f"模型推論失敗: {e}")
                stable_prediction = "Error"
                consecutive_count = 0
                last_prediction = None

            cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "No hands detected.", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            consecutive_count = 0
            last_prediction = None
            stable_prediction = "No hands detected"

        cv2.imshow("Double Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("結束程式")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
