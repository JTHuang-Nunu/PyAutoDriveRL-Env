import os
import time
import numpy as np
import cv2
import eventlet
import gymnasium as gym
import torch as th 

from collections import deque
from gymnasium import spaces

from CarDataService import CarSocketService, CarData
from util.image_process import ImageProcessing
from util.reward_task import MixTask

class CarRLEnvironment(gym.Env):
    def __init__(self, car_service: CarSocketService, share_dict, image_wh_size=64):
        """
        Initialize the CarRL environment with a given car service and number of frames to stack.

        Args:
            car_service (CarSocketService): The service that communicates with the car's simulator.
            frame_stack_num (int): Number of frames to stack for observation space.
        """
        super(CarRLEnvironment, self).__init__()

        self.car_service = car_service
        self.car_service.start_with_nothing()
        self.image_size = image_wh_size

        # Observation space includes stacked frames and steering/speed information.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8),
            "steering_speed": spaces.Box(low=np.array([-25.0, 0.0]), high=np.array([25.0, 100.0]), dtype=np.float32)
        })

        # Action space: steering angle (-1 to 1) and throttle (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize observation and other variables
        self.current_observation = {
            "image": np.zeros(self.observation_space['image'].shape, dtype=np.uint8),
            "steering_speed": np.zeros(self.observation_space['steering_speed'].shape, dtype=np.float32)
        }
        self.done = False
        self._last_timestamp = 0
        self.start_time = None
        self.system_delay = car_service.system_delay
        self.progress_queue = deque(maxlen=5)
        self.__check_done_use_last_timestamp = 0
        self.__check_done_use_progress = 0

        self.shared_dict = share_dict
        self.reward_task = MixTask(task_weights={
            'progress': 0.7,
            'tracking': 1.0,
            'collision': 1.0,
            'anomaly': 1.0
        },)
        
        self.seq_len = 4
        self.frame_buffer = deque(maxlen=self.seq_len)

        # Wait for connection and data
        while not (self.car_service.client_connected and self.car_service.initial_data_received):
            eventlet.sleep(self.system_delay)

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed for environment reset.
            options: Optional additional parameters.

        Returns:
            observation (dict): The initial observation containing the stacked images and steering/speed.
            info (dict): Additional information (empty in this case).
        """
        self.done = False
        self.car_service.send_control(0, 0, 1)  # Send stop command for a clean reset
        self.car_service.wait_for_new_data()

        car_data = self.car_service.carData

        # Preprocess the image and initialize the frame history
        image = car_data.image if car_data.image is not None else np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        processed_image = self._preprocess_observation(image)

        # Initialize observation with steering and speed
        self.current_observation = {
            "image": processed_image,
            "steering_speed": np.array([0.0, 0.0], dtype=np.float32)
        }

        self.start_time = time.time()
        self._last_timestamp = car_data.timestamp
        self.progress_queue.clear()
        self.__check_done_use_last_timestamp = car_data.timestamp
        self.__check_done_use_progress = 0

        # Reward Task reset
        self.reward_task.reset()
        return self.current_observation, {}

    def step(self, action):
        """
        Execute one step in the environment with the given action.

        Args:
            action (array): The action containing [steering_angle, throttle].

        Returns:
            observation (dict): Updated observation with stacked images and steering/speed data.
            reward (float): The reward for the step.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode is truncated (always False here).
            info (dict): Additional info (empty in this case).
        """
        # DO NOT CHANGE THE FOLLOWING CODE
        steering_angle, throttle = action
        self.car_service.send_control(steering_angle, throttle)
        self.car_service.wait_for_new_data()
        # DO NOT CHANGE THE PREVIOUS CODE

        car_data = self.car_service.carData
        self.progress_queue.append(float(car_data.progress))

        # Process and stack images
        image = car_data.image if car_data.image is not None else np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        processed_image = self._preprocess_observation(image)

        current_steering = float(car_data.steering_angle)
        current_speed = min(float(car_data.speed), 100.0)

        self.current_observation = {
            "image": processed_image,
            "steering_speed": np.array([current_steering, current_speed], dtype=np.float32)
        }

        reward = self._compute_reward_3(car_data)
        self.done = self._check_done(car_data)

        # ===== debug message =====
        # Show the image
        # cv2.imshow('Image', car_data.image)
        # cv2.waitKey(0)
        # cv2.imwrite('output_image.png', car_data.image)
        # self.car_data_window.update_data(car_data)
        self.shared_dict['car_data'] = car_data
        # self._clear_console()
        # self._print_debug_info(car_data)  # Debugging info for telemetry and time intervals
        # =========================

        #  =====Update timestamp and calculate FPS=====
        time_diff = self.car_service.carData.timestamp - self._last_timestamp
        fps = int(1000 / time_diff) if time_diff > 0 else 0
        print(f"\r{fps: 05.1f} fps -> unity world {fps/car_data.time_speed_up_scale: 05.1f} fps, reward: {reward: 05.2f}", end="\t")
        self._last_timestamp = car_data.timestamp
        #  ============================================
        
        return self.current_observation, reward, self.done, False, {}
    

    def _clear_console(self):
        """
        Clear the console output for easier debugging.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_debug_info(self, car_data):
        """
        Print car telemetry and frame time interval for debugging purposes.

        Args:
            car_data (CarData): Contains telemetry information like speed, progress, etc.
        """
        print(car_data)
    def _compute_reward(self, car_data: CarData):
        """
        Compute the reward for the current step based on the car's progress and position.

        Args:
            car_data (CarData): The current car data received from the car service.

        Returns:
            reward (float): The calculated reward based on progress and track position.
        """
        reward = (self.progress_queue[-1] - self.progress_queue[0]) * 100 + car_data.velocity_z * 0.005
        if car_data.y < 0:
            reward -= 10  # Penalize if off track
        # if car_data.obstacle_car == 1:
        #     reward -= 0.01  # Penalize if there is an obstacle
        return reward
    def _compute_reward_2(self, car_data: CarData):
        """
        Compute the reward for the current step based on the car's performance and behavior.

        Args:
            car_data (CarData): The current car data received from the car service.

        Returns:
            reward (float): The calculated reward based on multiple factors.
        """
        # Progress Reward
        progress_reward = (self.progress_queue[-1] - self.progress_queue[0]) * 100

        # Speed Reward: Encourage speed between 10 and 30 m/s
        speed = car_data.speed
        if 10 <= speed <= 30:
            speed_reward = speed * 0.1
        else:
            speed_reward = -abs(speed - 20) * 0.1  # Penalize if speed is outside the ideal range

        # Steering Penalty: Penalize excessive steering
        steering_angle = abs(car_data.steering_angle)
        steering_penalty = -steering_angle * 0.5 if steering_angle > 0.5 else 0

        # Off-Track Penalty: Apply a heavy penalty if the car is off-track
        off_track_penalty = -20 if car_data.y < 0 else 0

        # Collision Penalty: Penalize if the car collides with an obstacle
        collision_penalty = -10 if car_data.obstacle_car == 1 else 0

        # Stability Penalty: Penalize high angular velocity to encourage stable driving
        stability_penalty = -abs(car_data.angular_velocity_z) * 0.1

        # Total Reward Calculation
        reward = (
            progress_reward +
            speed_reward +
            steering_penalty +
            off_track_penalty +
            collision_penalty +
            stability_penalty
        )

        return reward
    def _compute_reward_3(self, car_data: CarData):
        reward = self.reward_task.reward(car_data)
        return reward




    def _check_done(self, car_data: CarData):
        """
        Check if the episode is done based on the car's position or progress.

        Args:
            car_data (CarData): The current car data received from the car service.

        Returns:
            done (bool): Whether the episode is finished.
        """
        if self.reward_task.done(car_data):
            return True
        
        if car_data.y < 0 or car_data.progress >= 100.0:
            return True

        if car_data.timestamp - self.__check_done_use_last_timestamp > 10000 / car_data.time_speed_up_scale: # MODIFY: 30000－＞ 10000
            if car_data.progress - self.__check_done_use_progress < 0.001:
                return True
            self.__check_done_use_last_timestamp = car_data.timestamp
            self.__check_done_use_progress = car_data.progress

        return False

    def _preprocess_observation(self, image):
        """
        Preprocess the image for observation by resizing and converting it to grayscale.

        Args:
            image (numpy.ndarray): The original image from the car's camera.

        Returns:
            processed_image (numpy.ndarray): The processed grayscale image.
        """
        resized_image = cv2.resize(image, (self.image_size, self.image_size))
        lane_processed_imag = ImageProcessing.lane_detection_pipeline(resized_image)
        enhanced_image = ImageProcessing.enhance_red_objects(lane_processed_imag)

        return enhanced_image.astype(np.uint8)
        

    def render(self, mode="human"):
        """
        Render the current car camera view (used for debugging).

        Args:
            mode (str): The render mode (default is 'human').
        """
        if self.car_service.carData.image is not None:
            cv2.imshow("Car Camera", self.car_service.carData.image)
            cv2.waitKey(1)

    def close(self):
        """
        Clean up any resources (e.g., close OpenCV windows).
        """
        cv2.destroyAllWindows()
