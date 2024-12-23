import sys
import signal
from train.train_stable_baseline_my import *
from multiprocessing import Process, Manager
from CarDataWindow import CarDataWindow
from util.logger import logger
# train_car_rl(strategy='PPO', model_mode='new', timesteps=100000, save_timesteps=10)
# train_car_rl(strategy='PPO', model_mode='load_best', timesteps=100000, save_timesteps=10, n_steps=1000)

def signal_handler(sig, frame):
    logger.info("Shutdown signal received. Terminating all processes...")
    if train_process.is_alive():
        train_process.terminate()
        train_process.join()
    if car_data_window:
        car_data_window.stop()
    sys.exit(0)


unit = 64 # 1202 64
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with Manager() as manager:
        share_dict = manager.dict()

        car_data_window = CarDataWindow(share_dict)

        # train_process = Process(  #Main
        #     target=train_car_rl,
        #     kwargs={
        #         "strategy": "SAC",
        #         "model_mode": "load_best",
        #         "timesteps": 500000, # 100000
        #         "save_timesteps": unit * 30, # 100
        #         # "n_steps": unit * 10,
        #         "batch_size": unit,
        #         "share_dict": share_dict,
        #         "image_wh_size": 128
        #     }
        # )
        train_process = Process( # New
            target=train_car_rl,
            kwargs={
                "strategy": "SAC",
                "model_mode": "new",
                "timesteps": 100000, # 100000
                "save_timesteps": unit * 50, # 100
                # "n_steps": unit * 10,
                "batch_size": unit,
                "share_dict": share_dict,
                "image_wh_size": 128
            }
        )
        try:
            
            train_process.start()
            # car_data_window.start() # show car data tkinter window
            train_process.join()
            # pass
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Stopping processes...")
            signal_handler(None, None)
        finally:
            if train_process.is_alive():
                train_process.terminate()
                train_process.join()
            if car_data_window:
                car_data_window.stop()
