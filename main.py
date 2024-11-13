from train.train_stable_baseline_my import *
from multiprocessing import Process, Manager
from CarDataWindow import CarDataWindow
# train_car_rl(strategy='PPO', model_mode='new', timesteps=100000, save_timesteps=10)
# train_car_rl(strategy='PPO', model_mode='load_best', timesteps=100000, save_timesteps=10, n_steps=1000)

if __name__ == '__main__':
    with Manager() as manager:
        share_dict = manager.dict()
        share_dict['car_data'] = str("")
        car_data_window = CarDataWindow(share_dict)
        
        train_process = Process(
            target=train_car_rl,
            kwargs={
                "strategy": "PPO",
                "model_mode": "load_best",
                "timesteps": 100000,
                "save_timesteps": 10,
                "n_steps": 64*10,
                "batch_size": 64,
                "share_dict": share_dict
            }
        )
        train_process.start()
        car_data_window.run()
        
        train_process.join()
