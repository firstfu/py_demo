import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate


class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """時間特徵轉換器"""

    def __init__(self):
        self.time_features = ["hour", "minute", "second", "weekday"]
        self.periods = {"hour": 24, "minute": 60, "second": 60, "weekday": 7}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = np.zeros((X.shape[0], len(self.time_features) * 2))
        for i, feature in enumerate(self.time_features):
            values = X[feature].values
            period = self.periods[feature]
            result[:, i * 2] = np.sin(2 * np.pi * values / period)
            result[:, i * 2 + 1] = np.cos(2 * np.pi * values / period)
        return result

    def get_feature_names_out(self, input_features=None):
        feature_names = []
        for feature in self.time_features:
            feature_names.extend([f"{feature}_sin", f"{feature}_cos"])
        return feature_names


class PathFeatureTransformer(BaseEstimator, TransformerMixin):
    """路徑特徵轉換器"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.folder_encoder = LabelEncoder()
        self.extension_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X["file_path"])
        folder_names = X["file_path"].apply(os.path.dirname)
        self.folder_encoder.fit(folder_names)
        extensions = X["file_path"].apply(lambda x: os.path.splitext(x)[1])
        self.extension_encoder.fit(extensions)
        return self

    def transform(self, X):
        file_path_encoded = self.label_encoder.transform(X["file_path"])
        folder_names = X["file_path"].apply(os.path.dirname)
        folder_encoded = self.folder_encoder.transform(folder_names)
        folder_depth = X["file_path"].apply(lambda x: len(x.split(os.sep)) - 1)
        extensions = X["file_path"].apply(lambda x: os.path.splitext(x)[1])
        extension_encoded = self.extension_encoder.transform(extensions)

        return np.column_stack(
            [file_path_encoded, folder_encoded, folder_depth, extension_encoded]
        )

    def get_feature_names_out(self, input_features=None):
        return [
            "file_path_encoded",
            "folder_encoded",
            "folder_depth",
            "extension_encoded",
        ]


def load_training_data(data_file: str):
    """載入訓練資料"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到資料檔案: {data_file}")

    df = pd.read_csv(data_file)
    if len(df) < 10:
        raise ValueError("資料量不足，至少需要10筆記錄")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def create_pipeline():
    """建立訓練 Pipeline"""
    # 特徵轉換器
    time_features = TimeFeatureTransformer()
    path_features = PathFeatureTransformer()

    # 特徵處理 Pipeline
    feature_pipeline = ColumnTransformer(
        transformers=[
            ("time_features", time_features, ["hour", "minute", "second", "weekday"]),
            ("path_features", path_features, ["file_path"]),
        ],
        remainder="drop",
    )

    # 完整訓練 Pipeline
    pipeline = Pipeline(
        [
            ("features", feature_pipeline),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1,
                    verbose=1,
                ),
            ),
        ]
    )

    return pipeline


def train_model(df: pd.DataFrame):
    """訓練模型"""
    # 修改資料準備方式
    X = df.copy()  # 複製整個 DataFrame

    # 創建目標標籤：使用下一個檔案作為預測目標
    next_files = df["file_path"].shift(-1)  # 向上移動一位
    X = X[:-1]  # 移除最後一筆記錄，因為它沒有對應的下一個檔案
    y = next_files[:-1].values  # 移除最後一個 NaN 值

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    # 建立並訓練 Pipeline
    print("\n訓練檔案預測模型...")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # 計算評估指標
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": pipeline.score(X_test, y_test),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    # 打印特徵重要性
    feature_names = (
        pipeline.named_steps["features"]
        .named_transformers_["time_features"]
        .get_feature_names_out()
        + pipeline.named_steps["features"]
        .named_transformers_["path_features"]
        .get_feature_names_out()
    )

    print("\n檔案預測模型 - 特徵重要性:")
    feature_importance = [
        [name, f"{importance:.4f}"]
        for name, importance in zip(
            feature_names, pipeline.named_steps["classifier"].feature_importances_
        )
    ]
    print(tabulate(feature_importance, headers=["特徵", "重要性"], tablefmt="grid"))

    return pipeline, metrics


def save_model(pipeline, model_file: str):
    """儲存訓練好的模型"""
    models = {"version": "4.0", "pipeline": pipeline}

    with open(model_file, "wb") as f:
        pickle.dump(models, f)


def main():
    data_file = "file_changes.csv"
    model_file = "ml_model.pkl"

    try:
        print("開始載入訓練資料...")
        df = load_training_data(data_file)
        print(f"成功載入 {len(df)} 筆資料")

        print("\n開始訓練模型...")
        pipeline, metrics = train_model(df)

        # 顯示評估結果
        evaluation_metrics = [
            ["準確率", f"{metrics['accuracy']:.2%}"],
            ["精確度", f"{metrics['precision']:.2%}"],
            ["召回率", f"{metrics['recall']:.2%}"],
            ["F1分數", f"{metrics['f1']:.2%}"],
        ]
        print("\n檔案預測模型評估結果:")
        print(tabulate(evaluation_metrics, headers=["指標", "數值"], tablefmt="grid"))

        print("\n儲存模型...")
        save_model(pipeline, model_file)
        print(f"模型已儲存至 {model_file}")

    except Exception as e:
        print(f"訓練過程發生錯誤: {e}")


if __name__ == "__main__":
    main()
