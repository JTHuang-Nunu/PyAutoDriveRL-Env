import os
from utils.logger import logger
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env

import torch as th
import torch.nn as nn
from gymnasium import spaces

from CarRLEnvironment import CarRLEnvironment
from CarDataService import CarSocketService
from utils.cnn_task import *
from utils.model_manager import *

loader = ModelManager()
# logger.basicConfig(level=logger.INFO, format="%(levelname)s - %(message)s")



def train_car_rl(strategy='PPO', model_mode='load',manual_path=None, timesteps=1000000, save_timesteps=None, n_steps=None,batch_size=None, share_dict=None, log_path='log/', image_wh_size=128):
    """
    Parameters:
        strategy (str): The RL strategy to use ('PPO' or 'SAC').
        train_type (str): The training type ('load' or 'new').

    Main training loop for the car RL environment using SAC (or you can change it to PPO).
    Modifiable sections are marked with comments to help first-time users easily adjust the code.
    """
    # -------------------------------------------------
    # ------------------- Initialize ------------------
    # Initialize the CarSocketService
    car_service = CarSocketService(system_delay=0.1, )  # Modify system delay to match the environment
    # Initialize the custom RL environment
    env = CarRLEnvironment(car_service, share_dict, image_wh_size)  # Adjust frame_stack_num based on how many frames to stack

    # Check if the environment follows the Gym standards
    check_env(env)

    # Define policy arguments with the custom CNN feature extractor
    # policy_kwargs = {
    #     "features_extractor_class": CustomCNN,
    #     "features_extractor_kwargs": {"features_dim": 256},  # Change feature dimensions if needed
    # }
    policy_kwargs = {
        "features_extractor_class": MultiTaskCNN,
        "features_extractor_kwargs": {"features_dim": 256},  # Change feature dimensions if needed
    }
    

    # Choose between SAC or PPO model (PPO used here for example)
    try: 
        if strategy == 'PPO':
            model = PPO("MultiInputPolicy", 
                        env, 
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        tensorboard_log=log_path)
        elif strategy == 'SAC':
            model = SAC("MultiInputPolicy",
                        env, 
                        policy_kwargs=policy_kwargs, 
                        buffer_size=1_000_000, 
                        verbose=0,
                        tensorboard_log=log_path)
        else:
            raise ValueError("Invalid strategy.")
    except ValueError as e:
        logger.error(e)
        return
    
    model_dir = 'runs/'
    loader.set_model_dir(model_dir)
    # Load or create a new model
    try:
        if model_mode != "new":
            model_path = None
            # Get the latest model path
            if model_mode == "load_latest":
                model_path = loader.get_latest_model(strategy, type='latest')
            # Get the best model path
            elif model_mode == "load_best":
                model_path = loader.get_latest_model(strategy, type='best')
            # Get the manual model path
            elif model_mode == "manual":
                if manual_path is None:
                    raise ValueError("Please specify the manual_path for the model.")
                model_path = manual_path
            else:
                raise ValueError("Invalid model_mode. Choose from 'load', 'new', or 'manual'.")
            
            # Load the model if found
            if model_path:
                    model = loader.load_model(model, model_path)
            else:
                logger.info("No existing model found. Starting new training.")

        # Create a runs weight folder
        model_dir = loader.create_model_directory(strategy)
        logger.info(f"Created new model directory: {model_dir}")

    except FileNotFoundError as e:
        logger.error(e)
        return


    # Set training parameters
    total_timesteps = timesteps # Number of timesteps to train in each loop
    save_timesteps = save_timesteps  # How often to save the model (in timesteps)
    best_reward = -np.inf  # Initial best reward
    current_timesteps = 0  # Record the current training timesteps
    early_stopping_counter = 0
    patience = 100  # Patience for early stopping From 10 to 100

    # -------------------------------------------------
    # ----------------- Training Loop -----------------
    # Initialize observation and info

    # Training loop
    if model is not None:
        # logger.info(f"Training {model.__class__.__name__} model...")
        obs, info = env.reset()
        while current_timesteps < total_timesteps:
            timesteps_to_train = min(save_timesteps, total_timesteps - current_timesteps)
            model.learn(total_timesteps=timesteps_to_train, reset_num_timesteps=False)
            current_timesteps += timesteps_to_train

            # Save latest model
            loader.save_model(model, 'latest')

            # Evaluate the model by running
            mean_reward = evaluate(model, env)
            if mean_reward > best_reward:
                best_reward = mean_reward
                loader.save_model(model, 'best')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check for early stopping
            if early_stopping_counter >= patience:
                logger.info("Early stopping triggered. Ending training.")
                break

            # Display training progress
            logger.info(f"Training step complete. Mean reward: {mean_reward:.2f}, Best reward: {best_reward:.2f}")
    else: 
        logger.error("Model not found. Training failed.")
        return

    # -------------------------------------------------
    # -------------------------------------------------

def evaluate(model, env, n_episodes=5):
    total_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, truncated, info = env.step(action)
            total_reward += rewards
            if done:
                break
        total_rewards.append(total_reward)
    mean_reward = np.mean(total_rewards)
    logger.info(f"Evaluation over {n_episodes} episodes: Mean reward = {mean_reward}")
    return mean_reward
