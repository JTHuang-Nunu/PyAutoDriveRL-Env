import sys
import signal
from train.train_stable_baseline_my import *
from multiprocessing import Process, Manager
from CarDataWindow import CarDataWindow
from utils.logger import logger
# train_car_rl(strategy='PPO', model_mode='new', timesteps=100000, save_timesteps=10)
# train_car_rl(strategy='PPO', model_mode='load_best', timesteps=100000, save_timesteps=10, n_steps=1000)

# def signal_handler(sig, frame):
#     print("Shutdown signal received. Terminating all processes...")
#     train_process.terminate()
#     # sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)

unit = 64
if __name__ == '__main__':
    with Manager() as manager:
        share_dict = manager.dict()
        # share_dict['car_data'] = str("")
        car_data_window = CarDataWindow(share_dict)
        
        train_process = Process(
            target=train_car_rl,
            kwargs={
                "strategy": "PPO",
                "model_mode": "load_best",
                "timesteps": 1000000,
                "save_timesteps": unit*100,
                "n_steps": unit*5,
                "batch_size": unit,
                "share_dict": share_dict
            }
        )
        train_process.start()        
        car_data_window.start()

        train_process.join()
