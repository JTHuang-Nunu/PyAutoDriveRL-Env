import os
import torch
from datetime import datetime
from utils.logger import logger
import numpy as np
from utils.logger import logger
class Loader:
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
        model_name = None
        if type == 'best':
            model_name = 'best_model.pth'
        elif type == 'latest':
            model_name = 'latest_model.pth'

        # Search for the specific model 
        for i in sorted(os.listdir(self.model_dir),reverse=True):
            path = os.path.join(self.model_dir, i)
            if os.path.isdir(path):
                if os.path.exists(os.path.join(path,model_name)):
                    model_path = os.path.join(path, model_name)
                    logger.info(f"Loading model from: {model_path}")
                    return model_path
        return None


    def save_model(self, model, type):
        """
        保存模型至指定的資料夾。
        """
        # Update model.pth
        if type == 'latest':
            latest_path = os.path.join(self.model_curr_dir, "latest_model.pth")
            model.save(latest_path)
            logger.info(f"Latest model saved to: {latest_path}")
            # bak the latest model
            latest_path_bak = os.path.join(self.model_dir, "latest_model.pth")
            model.save(latest_path_bak)
        elif type == 'best':
            best_path = os.path.join(self.model_curr_dir, "best_model.pth")
            model.save(best_path)
            logger.info(f"Best model saved to: {best_path}")
            # bak the best model
            best_path_bak = os.path.join(self.model_dir, "best_model.pth")
            model.save(best_path_bak)

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
    def load_model(self, model, model_path):
        """
        載入指定路徑的模型。
        """
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model.load(model_path)
            # model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")