import os
import numpy as np
import eventlet
import socketio
from flask import Flask
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from PIL import Image
from io import BytesIO
import base64

from load_model import *
from logger import logger

class CarData:
    def __init__(self):
        self.image = None
        self.steering_angle = np.nan
        self.throttle = np.nan
        self.speed = np.nan

    def update(self, data):
        self.image = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))))[..., ::-1]
        self.steering_angle = float(data["steering_angle"]) if data["steering_angle"] != "N/A" else np.nan
        self.throttle = float(data["throttle"]) if data["throttle"] != "N/A" else np.nan
        self.speed = float(data["speed"]) if data["speed"] != "N/A" else np.nan

class CarSocketService:
    def __init__(self, system_delay=0.1):
        self.sio = socketio.Server()
        self.app = Flask(__name__)
        self.carData = CarData()
        self.system_delay = system_delay
        self.client_connected = False
        self.initial_data_received = False

        self.register_sio_events()

    def register_sio_events(self):
        @self.sio.on('telemetry')
        def telemetry(sid, data):
            if data:
                self.carData.update(data)
                self.initial_data_received = True

        @self.sio.on('connect')
        def connect(sid, environ):
            print("Client connected:", sid)
            self.client_connected = True

        @self.sio.on('disconnect')
        def disconnect(sid):
            print("Client disconnected:", sid)
            self.client_connected = False

    def wait_for_new_data(self):
        while self.carData.image is None:
            eventlet.sleep(self.system_delay)

    def send_control(self, steering_angle, throttle, reset_trigger=0):
        self.sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle),
                'reset_trigger': str(reset_trigger),
            },
            skip_sid=True
        )

def train_car_rl(strategy='PPO', model_mode='load', manual_path=None, timesteps=100000):
    # Initialize CarSocketService
    car_service = CarSocketService(system_delay=0.1)

    # Define the model
    if strategy == 'PPO':
        model = PPO("CnnPolicy", car_service, verbose=1)
    elif strategy == 'SAC':
        model = SAC("CnnPolicy", car_service, verbose=1)
    else:
        logger.error("Invalid strategy.")
        return

    # Load or create the model
    if model_mode == "new":
        logger.info("Starting new model training.")
    elif model_mode == "load":
        model_path = get_latest_model(strategy)
        if model_path:
            model = load_model(model, model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("No existing model found, starting new training.")
    elif model_mode == "manual" and manual_path:
        model = load_model(model, manual_path)
        logger.info(f"Loaded model from {manual_path}")
    else:
        logger.error("Invalid model_mode.")
        return

    total_timesteps = timesteps
    current_timesteps = 0

    try:
        while current_timesteps < total_timesteps:
            # Wait for new telemetry data
            car_service.wait_for_new_data()

            # Prepare observation from CarData
            obs = {
                'image': car_service.carData.image,
                'steering_speed': np.array([car_service.carData.steering_angle, car_service.carData.speed])
            }

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            steering_angle, throttle = action

            # Send control signals
            car_service.send_control(steering_angle, throttle)

            # Update training step
            model.learn(total_timesteps=1000, reset_num_timesteps=False)
            current_timesteps += 1000

            # Save model periodically
            if current_timesteps % 10000 == 0:
                save_model("latest", model)
                logger.info(f"Model saved at timestep {current_timesteps}")

    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving model...")
        save_model("interrupted", model)

    logger.info("Training complete.")

