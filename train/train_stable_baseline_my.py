import os
from logger import logger
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gymnasium import spaces

from CarRLEnvironment import CarRLEnvironment
from CarDataService import CarSocketService
from load_model import *

# logger.basicConfig(level=logger.INFO, format="%(levelname)s - %(message)s")

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for handling image input and extracting features.

    Args:
        observation_space (spaces.Dict): The observation space which includes the image input.
        features_dim (int): The dimension of the output feature vector after CNN layers.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Extract the 'image' shape from observation space, assuming image is (64, 64, 3)
        super(CustomCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space['image'].shape[0]  # Get the number of input channels (stacked frames)

        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=9, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Get the output dimension of the CNN layers
        with th.no_grad():
            sample_input = th.zeros(1, *observation_space['image'].shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]

        # Define a fully connected layer to combine CNN output with other inputs (steering/speed)
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim + 2, features_dim),  # Add steering and speed (2,)
            nn.ReLU(),
        )

    def forward(self, observations):
        """
        Forward pass for feature extraction.

        Args:
            observations (dict): A dictionary containing 'image' and 'steering_speed' inputs.

        Returns:
            Tensor: A tensor representing extracted features from image and steering/speed.
        """
        image = observations['image']  # Extract image input
        image_features = self.cnn(image)  # Extract features using CNN

        # Process non-image input (steering and speed)
        steering_speed = observations['steering_speed']

        # Concatenate image features and steering/speed, and pass through the linear layer
        return self.linear(th.cat([image_features, steering_speed], dim=1))

def train_car_rl(strategy='PPO', model_mode='load',manual_path=None, timesteps=1000000, save_interval=10000):
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
    env = CarRLEnvironment(car_service)  # Adjust frame_stack_num based on how many frames to stack

    # Check if the environment follows the Gym standards
    check_env(env)

    # Define policy arguments with the custom CNN feature extractor
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},  # Change feature dimensions if needed
    }

    # Choose between SAC or PPO model (PPO used here for example)
    try: 
        if strategy == 'PPO':
            model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        elif strategy == 'SAC':
            model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, buffer_size=1_000_000, verbose=0)
        else:
            raise ValueError("Invalid strategy.")
    except ValueError as e:
        logger.error(e)
        return
    
    model_dir = 'runs/'
    set_model_dir(model_dir)
    # Load or create a new model
    try: 
        if model_mode == "new":
            model_dir = create_model_directory(strategy)
            logger.info(f"Created new model directory: {model_dir}")

        elif model_mode == "load":
            latest_model_path = get_latest_model(strategy)
            if latest_model_path:
                model = load_model(model, latest_model_path)
            else:
                logger.info("No existing model found. Starting new training.")
                model_dir = create_model_directory(strategy)

        elif model_mode == "manual":
            if manual_path is None:
                raise ValueError("Please specify the manual_path for the model.")
            model = load_model(model, manual_path)
        else:
            raise ValueError("Invalid model_mode. Choose from 'load', 'new', or 'manual'.")
    except FileNotFoundError as e:
        logger.error(e)
        return


    # Set training parameters
    total_timesteps = timesteps # Number of timesteps to train in each loop
    save_interval = save_interval  # How often to save the model (in timesteps)
    best_reward = -np.inf  # Initial best reward
    current_timesteps = 0  # Record the current training timesteps
    early_stopping_counter = 0
    patience = 10  # Patience for early stopping
    epochs = total_timesteps // save_interval

    # -------------------------------------------------
    # ----------------- Training Loop -----------------
    # Initialize observation and info


    # Training loop
    if model is not None:
        # logger.info(f"Training {model.__class__.__name__} model...")
        for e in range(epochs):
            obs, info = env.reset()
            while current_timesteps < total_timesteps:
                timesteps_to_train = min(save_interval, total_timesteps - current_timesteps)
                model.learn(total_timesteps=timesteps_to_train, reset_num_timesteps=False)
                current_timesteps += timesteps_to_train

                # Save latest model
                logger.info(f"Saving latest model: {latest_model_path}")
                save_model('latest', model)

                # Evaluate the model by running
                mean_reward = evaluate(model, env)
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    save_model('best', model)
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

def print_debug_info(cardata):
    print(cardata)
    
    

