import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from config import labels  # 從共用設定檔引入標籤清單

data = pd.read_csv("gesture_data.csv", header=None)
X = data.iloc[:, :-1].values  # 特徵為126維
y = data.iloc[:, -1].values    # 標籤

print("csv 標籤集中:", set(y))
print("labels 清單:", labels)

# 建立映射
label_to_index = {label: idx for idx, label in enumerate(labels)}
y_encoded = np.array([label_to_index[label] for label in y])

num_classes = len(labels)
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, input_shape=(126,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

model.save("gesture_model.h5")
