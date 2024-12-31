import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate


def load_training_data(data_file: str):
    """載入訓練資料"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到資料檔案: {data_file}")

    # 讀取CSV檔案
    df = pd.read_csv(data_file)

    # 確保資料足夠
    if len(df) < 10:
        raise ValueError("資料量不足，至少需要10筆記錄")

    # 轉換時間戳為datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


def prepare_features(df: pd.DataFrame):
    """準備特徵資料"""
    # 對檔案路徑進行編碼
    label_encoder = LabelEncoder()
    df["file_path_encoded"] = label_encoder.fit_transform(df["file_path"])

    # 基本時間特徵
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["second_sin"] = np.sin(2 * np.pi * df["second"] / 60)
    df["second_cos"] = np.cos(2 * np.pi * df["second"] / 60)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # 檔案路徑特徵
    df["folder_depth"] = df["file_path"].apply(lambda x: len(x.split(os.sep)) - 1)
    df["folder_name"] = df["file_path"].apply(lambda x: os.path.dirname(x))
    df["folder_encoded"] = label_encoder.fit_transform(df["folder_name"])
    df["file_extension"] = df["file_path"].apply(lambda x: os.path.splitext(x)[1])
    df["file_extension_encoded"] = label_encoder.fit_transform(df["file_extension"])

    # 時間差特徵
    df["time_diff"] = df["datetime"].diff().dt.total_seconds()
    df["time_diff"] = df["time_diff"].fillna(0)

    # 建立特徵矩陣
    features = df[
        [
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "second_sin",
            "second_cos",
            "weekday_sin",
            "weekday_cos",
            "file_path_encoded",
            "folder_encoded",
            "folder_depth",
            "file_extension_encoded",
            "time_diff",
        ]
    ].values

    # 標準化特徵
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 準備訓練資料
    X = features[:-1]  # 除了最後一條記錄
    next_files = df["file_path"].values[1:]  # 從第二條記錄開始

    return X, next_files, label_encoder, scaler


def train_models(X, next_files):
    """訓練模型"""
    # 分割訓練集和測試集
    X_train, X_test, files_train, files_test = train_test_split(
        X, next_files, test_size=0.2, random_state=42
    )

    print(f"\n資料集大小:")
    print(
        tabulate(
            [
                ["訓練集", len(X_train)],
                ["測試集", len(X_test)],
            ],
            headers=["資料集", "樣本數"],
            tablefmt="grid",
        )
    )

    # 訓練檔案預測模型
    print("\n訓練檔案預測模型...")
    file_predictor = RandomForestClassifier(
        n_estimators=200,  # 增加樹的數量
        max_depth=15,  # 增加樹的深度
        min_samples_split=5,  # 減少分裂所需的最小樣本數
        min_samples_leaf=2,  # 減少葉節點的最小樣本數
        class_weight="balanced",  # 使用平衡權重
        bootstrap=True,  # 使用bootstrap採樣
        random_state=42,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1,
    )
    file_predictor.fit(X_train, files_train)

    # 計算分類模型的評估指標
    file_pred = file_predictor.predict(X_test)
    file_accuracy = file_predictor.score(X_test, files_test)
    file_precision = precision_score(files_test, file_pred, average="weighted")
    file_recall = recall_score(files_test, file_pred, average="weighted")
    file_f1 = f1_score(files_test, file_pred, average="weighted")
    conf_matrix = confusion_matrix(files_test, file_pred)

    # 打印特徵重要性
    feature_names = [
        "小時(sin)",
        "小時(cos)",
        "分鐘(sin)",
        "分鐘(cos)",
        "秒(sin)",
        "秒(cos)",
        "星期(sin)",
        "星期(cos)",
        "檔案路徑編碼",
        "資料夾編碼",
        "資料夾深度",
        "副檔名編碼",
        "時間差",
    ]

    print("\n檔案預測模型 - 特徵重要性:")
    feature_importance = [
        [name, f"{importance:.4f}"]
        for name, importance in zip(feature_names, file_predictor.feature_importances_)
    ]
    print(tabulate(feature_importance, headers=["特徵", "重要性"], tablefmt="grid"))

    metrics = {
        "file": {
            "accuracy": file_accuracy,
            "precision": file_precision,
            "recall": file_recall,
            "f1": file_f1,
            "confusion_matrix": conf_matrix,
        }
    }

    return file_predictor, metrics


def save_models(file_predictor, label_encoder, scaler, model_file: str):
    """儲存訓練好的模型"""
    models = {
        "version": "3.0",  # 更新版本標記
        "file_predictor": file_predictor,
        "label_encoder": label_encoder,
        "scaler": scaler,  # 新增 scaler
    }

    with open(model_file, "wb") as f:
        pickle.dump(models, f)


def main():
    data_file = "file_changes.csv"
    model_file = "ml_model.pkl"

    try:
        print("開始載入訓練資料...")
        df = load_training_data(data_file)
        print(f"成功載入 {len(df)} 筆資料")

        print("\n準備特徵資料...")
        X, next_files, label_encoder, scaler = prepare_features(df)
        print("特徵資料準備完成")

        print("\n開始訓練模型...")
        file_predictor, metrics = train_models(X, next_files)

        # 檔案預測模型結果表格
        file_metrics = [
            ["準確率", f"{metrics['file']['accuracy']:.2%}"],
            ["精確度", f"{metrics['file']['precision']:.2%}"],
            ["召回率", f"{metrics['file']['recall']:.2%}"],
            ["F1分數", f"{metrics['file']['f1']:.2%}"],
        ]
        print("\n檔案預測模型評估結果:")
        print(tabulate(file_metrics, headers=["指標", "數值"], tablefmt="grid"))

        print("\n儲存模型...")
        save_models(file_predictor, label_encoder, scaler, model_file)
        print(f"模型已儲存至 {model_file}")

    except Exception as e:
        print(f"訓練過程發生錯誤: {e}")


if __name__ == "__main__":
    main()
