from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import coremltools as ct

app = Flask(__name__)

# 載入 Core ML 模型
MODEL_PATH = "signLanguage.mlmodel"
model = ct.models.MLModel(MODEL_PATH)
input_name = list(model.input_description._fd_spec)[0].name

# MediaPipe Hands 設定：static_image_mode=False 啟用連續追蹤
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 關鍵：開啟追蹤模式提升精度
    max_num_hands=2,
    min_detection_confidence=0.80,
    min_tracking_confidence=0.80
)

# 全局變數示範（單用戶或單線程情況適用）
confidence_threshold = 0.95
stable_frame_threshold = 5
consecutive_count = 0
last_prediction = None
stable_prediction = "No confident prediction"

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global consecutive_count, last_prediction, stable_prediction

    if 'frame' not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files['frame']
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # 建議中的前處理：鏡像(水平翻轉)保持和 Swift 端一致
    img = cv2.flip(img, 1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            data_row = []
            for lm in hand_landmarks.landmark:
                data_row.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(data_row)

        while len(landmarks_list) < 2:
            landmarks_list.append([0.0] * 63)

        combined_features = landmarks_list[0] + landmarks_list[1]
        input_data = np.array(combined_features, dtype=np.float32).reshape(1, -1)

        try:
            outputs = model.predict({input_name: input_data})

            class_probs = outputs.get('classProbability', None)
            predicted_label = outputs.get('classLabel', 'Unknown')

            if class_probs is not None:
                max_prob = max(class_probs.values()) if isinstance(class_probs, dict) else np.max(class_probs)

                if max_prob >= confidence_threshold:
                    if predicted_label == last_prediction:
                        consecutive_count += 1
                    else:
                        consecutive_count = 1
                        last_prediction = predicted_label

                    if consecutive_count >= stable_frame_threshold:
                        stable_prediction = predicted_label
                else:
                    consecutive_count = 0
                    stable_prediction = "Confidence too low"
                    last_prediction = None
            else:
                consecutive_count = 0
                stable_prediction = predicted_label
                last_prediction = None

        except Exception as e:
            stable_prediction = "Error"
            consecutive_count = 0
            last_prediction = None
    else:
        stable_prediction = "No hands detected"
        consecutive_count = 0
        last_prediction = None

    return jsonify({"result": stable_prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
