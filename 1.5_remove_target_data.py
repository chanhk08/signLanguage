import pandas as pd

def remove_negative_samples(csv_path="gesture_data.csv", target_label="negative"): #target_label 移除有關 label 已紀錄的 data*****
    # 讀取CSV檔案
    df = pd.read_csv(csv_path, header=None)

    # 假設標籤在最後一欄，過濾掉標籤等於負樣本標籤的行
    df_filtered = df[df.iloc[:, -1] != target_label]

    # 將過濾後的資料寫回CSV，覆蓋原檔案，且不寫入索引
    df_filtered.to_csv(csv_path, index=False, header=False)

    print(f"已移除 '{target_label}' 標籤的資料列，更新檔案：{csv_path}")

if __name__ == "__main__":
    remove_negative_samples()
