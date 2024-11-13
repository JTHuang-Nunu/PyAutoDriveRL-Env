<<<<<<< HEAD
import sys
import signal
=======
>>>>>>> 37fc28b (ADD: GUI for Car_data and rewrite entry file to "main.py")
from train.train_stable_baseline_my import *
from multiprocessing import Process, Manager
from CarDataWindow import CarDataWindow
# train_car_rl(strategy='PPO', model_mode='new', timesteps=100000, save_timesteps=10)
# train_car_rl(strategy='PPO', model_mode='load_best', timesteps=100000, save_timesteps=10, n_steps=1000)

<<<<<<< HEAD
def signal_handler(sig, frame):
    print("Shutdown signal received. Terminating all processes...")
    train_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

=======
>>>>>>> 37fc28b (ADD: GUI for Car_data and rewrite entry file to "main.py")
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
<<<<<<< HEAD
                "timesteps": 1000000,
                "save_timesteps": 64*100,
=======
                "timesteps": 100000,
                "save_timesteps": 10,
>>>>>>> 37fc28b (ADD: GUI for Car_data and rewrite entry file to "main.py")
                "n_steps": 64*10,
                "batch_size": 64,
                "share_dict": share_dict
            }
        )
        train_process.start()
<<<<<<< HEAD
        
        try:
            car_data_window.run()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Shutting down...")
            train_process.terminate()

=======
        car_data_window.run()
        
>>>>>>> 37fc28b (ADD: GUI for Car_data and rewrite entry file to "main.py")
        train_process.join()
