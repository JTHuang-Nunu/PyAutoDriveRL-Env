import os
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO, SAC
from CarRLEnvironment import CarRLEnvironment
from CarDataService import CarSocketService
from util.logger import logger

class ModelManager:
    def __init__(self):
        self.model_dir = None
        self.model_curr_dir = None

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    def create_model_directory(self, model_name):
        """
        每次訓練新模型時建立新資料夾，使用當前時間來命名。
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_curr_dir = os.path.join(self.model_dir, f"{model_name}_{current_time}")
        os.makedirs(self.model_curr_dir, exist_ok=True)
        
        return self.model_curr_dir

    def get_latest_model(self, model_name, type='best'):
        """
        latest
        """
        # Set the model type
        if type == 'best':
            weight_name = 'best_model.pth'
        elif type == 'latest':
            weight_name = 'latest_model.pth'

        subdirs = sorted(
            [d for d in os.listdir(self.model_dir) if d.startswith(model_name)],
            reverse=True
        )

        for subdir in subdirs:
            subdir_path = os.path.join(self.model_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, weight_name)
                if os.path.exists(model_path):
                    logger.info(f"Loading model from: {model_path}")
                    return model_path
        return None


    def save_model(self, model, type, file_name=None):
        """
        保存模型至指定的資料夾。
        """
        # Update model.pth
        if type == 'latest':
            latest_path = os.path.join(self.model_curr_dir, "latest_model.pth")
            model.save(latest_path)
            logger.info(f"Latest model saved to: {latest_path}")
            # # bak the latest model
            # latest_path_bak = os.path.join(self.model_dir, "latest_model.pth")
            # model.save(latest_path_bak)
        elif type == 'best':
            best_path = os.path.join(self.model_curr_dir, "best_model.pth")
            model.save(best_path)
            logger.info(f"Best model saved to: {best_path}")
            # # bak the best model
            # best_path_bak = os.path.join(self.model_dir, "best_model.pth")
            # model.save(best_path_bak)
        elif type == 'manual':
            if file_name is None:
                path = os.path.join(self.model_curr_dir, "manual_model.pth")
            else: 
                path = os.path.join(self.model_curr_dir, f"{file_name}.pth")
            model.save(path)
            logger.info(f"Manual model saved to: {path}")


        else:
            raise ValueError("Invalid mode. Choose from 'latest' or 'best'.")


    def load_model(self, model, model_path):
        """
        載入指定路徑的模型。
        """
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            model.load(model_path)
            # model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        

if __name__ == "__main__":
    share_dict = dict()
    car_service = CarSocketService(system_delay=0.1, )  # Modify system delay to match the environment
    env = CarRLEnvironment(car_service, share_dict)  # Adjust frame_stack_num based on how many frames to stack
    model = PPO("MultiInputPolicy",env)
    # 初始化 Loader 实例
    loader = ModelManager()

    # 设置模型主目录
    base_dir = "runs/"
    loader.set_model_dir(base_dir)

    # 创建新的模型目录
    model_name = "PPO"
    model_dir = loader.create_model_directory(model_name)
    print(f"Model directory created at: {model_dir}")

    # 获取最新的模型路径并加载模型
    latest_model_path = loader.get_latest_model(model_name, type='latest')
    if latest_model_path:
        model = loader.load_model(model, latest_model_path)

    # 获取最佳的模型路径并加载模型
    best_model_path = loader.get_latest_model(model_name, type='best')
    if best_model_path:
        model = loader.load_model(model, best_model_path)