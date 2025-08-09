import coremltools as ct
from config import labels  # 從共用設定檔引入

class_labels = labels
input_features = ct.TensorType(shape=(1, 126))  # 雙手特徵長度

mlmodel = ct.convert(
    "gesture_model.h5",
    source="tensorflow",
    convert_to="neuralnetwork",
    inputs=[input_features],
    classifier_config=ct.ClassifierConfig(class_labels)
)

mlmodel.save("signLanguage.mlmodel")
print("雙手模型已成功轉換並加入分類標籤，儲存為 signLanguage.mlmodel")
