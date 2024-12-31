import os
import random
import time
from datetime import datetime


class RandomWriter:
    def __init__(self, folder_path: str, max_files: int = 5):
        self.folder_path = folder_path
        self.max_files = max_files
        self.ensure_folder_exists()

    def ensure_folder_exists(self):
        """確保目標資料夾存在"""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def get_existing_files(self):
        """獲取現有檔案列表"""
        files = []
        for root, _, filenames in os.walk(self.folder_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    def create_new_file(self):
        """建立新檔案"""
        existing_files = self.get_existing_files()
        if len(existing_files) >= self.max_files:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.folder_path, f"file_{timestamp}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            content = f"初始內容 - {timestamp}\n"
            f.write(content)

        return file_path

    def modify_random_file(self):
        """隨機修改現有檔案"""
        files = self.get_existing_files()
        if not files:
            return self.create_new_file()

        target_file = random.choice(files)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            with open(target_file, "a", encoding="utf-8") as f:
                content = f"修改於 {timestamp}\n"
                f.write(content)
            return target_file
        except Exception as e:
            print(f"修改檔案時發生錯誤: {e}")
            return None

    def random_action(self):
        """執行隨機操作"""
        # 80% 機率修改現有檔案，20% 機率建立新檔案
        if random.random() < 0.8 and self.get_existing_files():
            return self.modify_random_file(), "修改"
        else:
            new_file = self.create_new_file()
            return new_file, "新增" if new_file else None


def main():
    writer = RandomWriter("./test_folder", max_files=5)
    print("開始隨機寫入操作...")

    try:
        while True:
            result = writer.random_action()
            if result[0]:
                print(f"{result[1]}檔案: {result[0]}")
            # time.sleep(random.uniform(1, 3))  # 隨機等待1-3秒
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n停止寫入操作")


if __name__ == "__main__":
    main()
