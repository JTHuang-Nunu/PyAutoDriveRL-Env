import cv2
from stable_baselines3 import PPO, SAC

from util.image_process import ImageProcessing

from datetime import datetime
from CarDataService import CarSocketService, CarData
import numpy as np
import random

# 假設這是你的影像處理函式
def _preprocess_observation(image):
    return cv2.resize(image, (128, 128))  # 這裡簡單地將影像縮放

# 儲存影像的函式
def save_image(image, filename="image.png"):
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

def RL_Process(car_data):
    # 假設 processed_image 是你從車輛獲得的處理過的影像

    processed_image = _preprocess_observation(car_data.image)
    lane_processed_imag = ImageProcessing.lane_detection_pipeline_forCNN(processed_image)
    cv2.imshow("lane_processed_imag", lane_processed_imag)

    key = cv2.waitKey(10)  # 等待1ms並取得按鍵的ASCII值

    # print(f"Key pressed: {key  & 0xFF}")
    if (key == (ord('z') & 0xFF)) or ((key == ord('Z') & 0xFF)):  # 當按下 'xX' 鍵時儲存影像
        print("You pressed 's'!")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"car_image_{timestamp}.png"
        save_image(processed_image, filename)

    return 0, 0, np.random.rand()

# 主程式
if __name__ == '__main__':
    car_service = CarSocketService(system_delay=0.1)
    rl_model = SAC.load(r".\runs\SAC_20241122_175744\best_model.pth")
    car_service.start_with_RLProcess(RL_Process)
