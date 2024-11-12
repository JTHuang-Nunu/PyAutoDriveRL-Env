import os
import torch
from datetime import datetime
from logger import logger
import numpy as np

def set_model_dir(model_dir):
    global dir
    dir = model_dir
def create_model_directory(model_name):
    """
    每次訓練新模型時建立新資料夾，使用當前時間來命名。
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(dir, f"{model_name}_{current_time}")
    # model_dir = f"{model_name}_{current_time}"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def get_latest_model(model_name):
    """
    找到最新的模型檔案 (依照時間排序)。
    """
    model_files = sorted([f for f in os.listdir() if f.startswith(model_name) and f.endswith(".pth")], reverse=True)
    return model_files[0] if model_files else None

def save_model(mode, model):
    """
    保存模型至指定的資料夾。
    """
    # model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to: {model_path}")

    # Update model.pth
    if mode == 'latest':
        latest_path = os.path.join(dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)
        logger.info(f"Latest model saved to: {latest_path}")
    elif mode == 'best':
        best_path = os.path.join(dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        logger.info(f"Best model saved to: {best_path}")
    else:
        raise ValueError("Invalid mode. Choose from 'latest' or 'best'.")


def load_model(model, model_path):
    """
    載入指定路徑的模型。
    """
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
        # model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")