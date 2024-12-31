import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class FileChangeRecord:
    def __init__(self):
        self.changes = []
        self.label_encoder = LabelEncoder()
        self.data_file = "file_changes.csv"
        self.init_data_file()
        self.migrate_old_data()

    def init_data_file(self):
        """初始化資料記錄檔案"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w", encoding="utf-8") as f:
                f.write(
                    "timestamp,file_path,change_type,hour,minute,second,weekday,folder_name,folder_depth\n"
                )

    def migrate_old_data(self):
        """遷移舊版本的資料到新格式"""
        if not os.path.exists(self.data_file):
            return

        try:
            # 讀取現有資料，並指定欄位名稱
            columns = [
                "timestamp",
                "file_path",
                "change_type",
                "hour",
                "minute",
                "second",
                "weekday",
                "folder_name",
                "folder_depth",
            ]

            # 檢查檔案是否為空
            if os.path.getsize(self.data_file) == 0:
                # 如果是空檔案，直接寫入標題
                with open(self.data_file, "w", encoding="utf-8") as f:
                    f.write(",".join(columns) + "\n")
                return

            # 嘗試讀取第一行判斷是否有標題
            with open(self.data_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()

            if not first_line.startswith("timestamp,file_path"):
                # 如果沒有標題列，讀取資料時指定欄位名稱
                df = pd.read_csv(self.data_file, names=columns)

                # 備份舊檔案
                backup_file = f"{self.data_file}.backup"
                os.rename(self.data_file, backup_file)

                # 使用新格式儲存（包含標題）
                df.to_csv(self.data_file, index=False)
                print(f"已新增標題列。舊資料已備份至 {backup_file}")
            else:
                # 如果有標題列，正常讀取
                df = pd.read_csv(self.data_file)

            # 檢查是否需要新增欄位
            if "folder_name" not in df.columns or "folder_depth" not in df.columns:
                print("檢測到舊版本資料，開始遷移...")

                # 新增必要的欄位
                df["folder_name"] = df["file_path"].apply(os.path.dirname)
                df["folder_depth"] = df["file_path"].apply(
                    lambda x: len(x.split(os.sep)) - 1
                )

                # 使用新格式儲存
                df.to_csv(self.data_file, index=False)
                print("資料遷移完成")

        except Exception as e:
            print(f"資料遷移過程發生錯誤: {e}")

    def add_record(self, file_path: str, change_type: str):
        """添加一條變更記錄"""
        now = datetime.now()
        folder_name = os.path.dirname(file_path)
        folder_depth = len(file_path.split(os.sep)) - 1

        record = {
            "timestamp": now.timestamp(),
            "file_path": file_path,
            "change_type": change_type,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "weekday": now.weekday(),
            "folder_name": folder_name,
            "folder_depth": folder_depth,
        }
        self.changes.append(record)

        # 寫入CSV檔案
        with open(self.data_file, "a", encoding="utf-8") as f:
            f.write(
                f"{record['timestamp']},{record['file_path']},{record['change_type']},"
                f"{record['hour']},{record['minute']},"
                f"{record['second']},{record['weekday']},{record['folder_name']},"
                f"{record['folder_depth']}\n"
            )


class MLPredictor:
    def __init__(self):
        self.records = FileChangeRecord()
        self.prediction_log_file = "prediction_results.csv"
        self.model_file = "ml_model.pkl"
        self.init_prediction_log()
        self.load_model()

    def init_prediction_log(self):
        """初始化預測記錄檔案"""
        if not os.path.exists(self.prediction_log_file):
            with open(self.prediction_log_file, "w", encoding="utf-8") as f:
                f.write("timestamp,predicted_file,actual_file,file_correct\n")

    def load_model(self):
        """載入訓練好的模型"""
        self.is_trained = False
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, "rb") as f:
                    models = pickle.load(f)

                    # 檢查模型版本
                    if isinstance(models, dict) and "version" in models:
                        if models["version"] >= "2.0":  # 新版本模型
                            self.file_predictor = models["file_predictor"]
                            self.records.label_encoder = models["label_encoder"]
                            self.is_trained = True
                        else:
                            print("檢測到舊版本模型，請重新訓練")
                    else:
                        print("無法確認模型版本，請重新訓練")
            except Exception as e:
                print(f"載入模型時發生錯誤: {e}")

    def log_prediction_result(
        self, prediction: Dict[str, any], actual_change: Dict[str, any]
    ):
        """記錄預測結果"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predicted_file = prediction.get("predicted_file", "")

        # 獲取實際變更資訊
        actual_file = ""
        if actual_change and "message" not in actual_change:
            actual_file = list(actual_change.keys())[0]

        # 計算預測準確度
        file_correct = "1" if predicted_file == actual_file else "0"

        # 寫入記錄
        with open(self.prediction_log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{predicted_file},{actual_file},{file_correct}\n")

    def predict_next_change(self, current_state: dict) -> Dict[str, any]:
        """預測下一次的檔案變更"""
        if not self.is_trained:
            return {"message": "尚未載入訓練好的模型"}

        try:
            # 如果有當前狀態，使用第一個檔案
            if current_state and len(current_state) > 0:
                current_file = list(current_state.keys())[0]
            else:
                # 如果沒有當前狀態，使用最後一次記錄的檔案
                if self.records.changes:
                    current_file = self.records.changes[-1]["file_path"]
                else:
                    return {"message": "沒有足夠的歷史資料來進行預測"}

            # 準備特徵資料
            try:
                # 檢查檔案是否在訓練資料中
                if current_file not in self.records.label_encoder.classes_:
                    return {"message": f"無法預測未知的檔案路徑：{current_file}"}

                # 轉換檔案路徑
                transform_result = self.records.label_encoder.transform([current_file])
                encoded_file = np.array(transform_result, dtype=np.float32)[0]

                # 處理資料夾路徑
                folder_name = os.path.dirname(current_file)
                if folder_name not in self.records.label_encoder.classes_:
                    return {"message": f"無法預測未知的資料夾路徑：{folder_name}"}

                transform_result = self.records.label_encoder.transform([folder_name])
                folder_encoded = np.array(transform_result, dtype=np.float32)[0]
                folder_depth = len(current_file.split(os.sep)) - 1

                now = datetime.now()
                current_hour = now.hour
                current_minute = now.minute
                current_second = now.second
                current_weekday = now.weekday()

                # 建立特徵陣列
                features = np.array(
                    [
                        [
                            current_hour,
                            current_minute,
                            current_second,
                            current_weekday,
                            encoded_file,
                            folder_encoded,
                            folder_depth,
                        ]
                    ],
                    dtype=np.float32,
                )

                # 進行預測
                predicted_file_idx = self.file_predictor.predict(features)[0]

                # 將預測的索引轉換回檔案路徑
                predicted_file = self.records.label_encoder.inverse_transform(
                    [predicted_file_idx]
                )[0]

                return {
                    "predicted_file": predicted_file,
                }
            except Exception as e:
                return {"message": f"預測過程發生錯誤: {str(e)}"}

        except Exception as e:
            return {"message": f"預測過程發生錯誤: {str(e)}"}


class MerkleNode:
    def __init__(
        self, hash_value: str, left=None, right=None, is_file=False, file_path=None
    ):
        self.hash_value = hash_value
        self.left = left
        self.right = right
        self.is_file = is_file
        self.file_path: Optional[str] = file_path

    def get_short_hash(self) -> str:
        """獲取短哈希值用於顯示"""
        return self.hash_value[:6]  # 只顯示前6位


class FolderMonitor:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.previous_tree: Optional[MerkleNode] = None
        self.ml_predictor = MLPredictor()
        self.last_prediction = None

    def print_tree(
        self,
        node: Optional[MerkleNode] = None,
        prev_str: str = "",
        is_left: bool = True,
    ) -> None:
        """以更美觀的樹狀結構打印默克爾樹
        Args:
            node: 當前節點
            prev_str: 前導字符串
            is_left: 是否為左子節點
        """
        if node is None:
            node = self.previous_tree
            if node is None:
                print("樹為空！")
                return

        # 先處理右子樹
        if node.right:
            self.print_tree(
                node.right, prev_str + ("    " if is_left else "│   "), False
            )

        # 打印當前節點
        connector = "└───" if is_left else "├───"
        if node.is_file and node.file_path:
            # 如果是文件節點，顯示文件名和短哈希
            file_name = os.path.basename(str(node.file_path))
            print(f"{prev_str}{connector}📄 {file_name} [{node.get_short_hash()}]")
        else:
            # 如果是中間節點，只顯示短哈希
            print(f"{prev_str}{connector}📁 [{node.get_short_hash()}]")

        # 處理左子樹
        if node.left:
            self.print_tree(node.left, prev_str + ("    " if is_left else "│   "), True)

    def calculate_file_hash(self, file_path: str) -> str:
        """計算單個文件的哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_all_files(self) -> List[str]:
        """獲取資料夾中所有文件的路徑"""
        file_paths = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return sorted(file_paths)  # 排序以確保順序一致

    def build_merkle_tree(self, file_paths: List[str]) -> Optional[MerkleNode]:
        """構建默克爾樹"""
        if not file_paths:
            return None

        # 創建葉子節點
        nodes = []
        for file_path in file_paths:
            hash_value = self.calculate_file_hash(file_path)
            nodes.append(MerkleNode(hash_value, is_file=True, file_path=file_path))

        # 構建樹
        while len(nodes) > 1:
            new_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else None

                if right:
                    combined_hash = hashlib.sha256(
                        (left.hash_value + right.hash_value).encode()
                    ).hexdigest()
                    new_level.append(MerkleNode(combined_hash, left, right))
                else:
                    new_level.append(left)
            nodes = new_level

        return nodes[0]

    def detect_changes(self) -> Dict[str, str]:
        """檢測文件變更"""
        current_files = self.get_all_files()
        current_tree = self.build_merkle_tree(current_files)

        changes: Dict[str, str] = {}

        # 如果是首次運行
        if self.previous_tree is None:
            self.previous_tree = current_tree
            return {"message": "初始化完成，首次建立默克爾樹"}

        # 比較根節點哈希值
        if (
            current_tree
            and self.previous_tree
            and current_tree.hash_value != self.previous_tree.hash_value
        ):
            # 遍歷所有文件比較哈希值
            current_hashes = {
                str(node.file_path): node.hash_value
                for node in self.get_leaf_nodes(current_tree)
                if node.file_path is not None
            }
            previous_hashes = {
                str(node.file_path): node.hash_value
                for node in self.get_leaf_nodes(self.previous_tree)
                if node.file_path is not None
            }

            # 檢測變更並記錄
            for file_path in current_hashes:
                if file_path not in previous_hashes:
                    changes[file_path] = "新增"
                    self.ml_predictor.records.add_record(str(file_path), "新增")

            for file_path in previous_hashes:
                if file_path not in current_hashes:
                    changes[file_path] = "刪除"
                    self.ml_predictor.records.add_record(str(file_path), "刪除")

            for file_path in current_hashes:
                if (
                    file_path in previous_hashes
                    and current_hashes[file_path] != previous_hashes[file_path]
                ):
                    changes[file_path] = "修改"
                    self.ml_predictor.records.add_record(str(file_path), "修改")

            # 如果有上一次的預測，記錄預測結果
            if self.last_prediction:
                self.ml_predictor.log_prediction_result(self.last_prediction, changes)

        self.previous_tree = current_tree
        return changes if changes else {"message": "沒有檢測到變更"}

    def predict_next_change(self) -> Dict[str, any]:
        """預測下一次可能的檔案變更"""
        current_state = self.detect_changes()
        if current_state.get("message"):
            current_state = {}
        prediction = self.ml_predictor.predict_next_change(current_state)
        self.last_prediction = prediction
        return prediction

    def get_leaf_nodes(self, node: Optional[MerkleNode]) -> List[MerkleNode]:
        """獲取所有葉子節點"""
        if not node:
            return []
        if node.is_file:
            return [node]

        leaves = []
        if node.left:
            leaves.extend(self.get_leaf_nodes(node.left))
        if node.right:
            leaves.extend(self.get_leaf_nodes(node.right))
        return leaves


def main():
    import time

    monitor = FolderMonitor("./test_folder")
    print("開始監控資料夾...")
    initial_state = monitor.detect_changes()
    print("初始狀態:", initial_state)
    print("\n當前默克爾樹結構：")
    # monitor.print_tree()

    try:
        while True:
            changes = monitor.detect_changes()
            if changes.get("message") != "沒有檢測到變更":
                print("\n檢測到以下變更：")
                for file_path, change_type in changes.items():
                    print(f"{change_type}: {file_path}")
                print("\n更新後的默克爾樹結構：")
                # monitor.print_tree()

                # 預測下一次變更
                prediction = monitor.predict_next_change()
                if "message" not in prediction:
                    print("\n預測下一次變更：")
                    print(f"預測檔案：{prediction['predicted_file']}")
                else:
                    print(f"\n預測狀態：{prediction['message']}")

            time.sleep(0.1)  # 每100毫秒檢查一次
    except KeyboardInterrupt:
        print("\n停止監控")


if __name__ == "__main__":
    main()
