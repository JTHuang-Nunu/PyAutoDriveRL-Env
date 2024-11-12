from train.train_stable_baseline_my import *

train_car_rl(strategy='PPO', model_mode='new', timesteps=100000, save_interval=10)
