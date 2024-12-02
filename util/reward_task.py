import numpy as np
from util.logger import logger
from collections import deque
from CarDataService import CarData
import numpy as np
from enum import Enum
from util.logger import logger
from collections import deque
from CarDataService import CarData

# Define UnitTestMode Enum
class UnitTestMode(Enum):
    DISABLE = "Disable"
    PROGRESS = "Progress"
    TRACKING = "Tracking"
    COLLISION = "Collision"
    ANOMALY = "Anomaly"

class MixTask:
    def __init__(self, task_weights: dict, unit_test: UnitTestMode = UnitTestMode.DISABLE, 
                 progress_reward: float = 100.0, collision_penalty: float = -10.0, anomaly_penalty: float = -10.0, 
                 speed_limits: tuple[float, float] = (3.0, 30.0)):
        '''
        Initialize the tasks with the given weights

        Unit test modes:
        - Disable: Run all tasks
        - Progress: Test only ProgressTask
        - Tracking: Test only TrackingTask
        - Collision: Test only CollisionTask
        - Anomaly: Test only AnomalyHandlingTask
        '''
        self.task_weights = task_weights
        self.ProgressTask = MaximizeProgressTask(progress_reward=progress_reward)
        self.TrackingTask = TrackingTask(min_speed=speed_limits[0], max_speed=speed_limits[1], speed_stability_threshold=1)
        self.CollisionTask = CollisionTask(collision_penalty=-collision_penalty)
        self.AnomalyHandlingTask = AnomalyHandlingTask(anomaly_penalty=anomaly_penalty)
        
        self.unit_test = unit_test
        if unit_test == UnitTestMode.DISABLE:
            logger.debug(f"***********************************")
            logger.debug(f"   Unit test mode: {unit_test.value}    ")
            logger.debug(f"***********************************")
        # Initialize rewards
        self.progress_reward = -np.inf
        self.tracking_reward = -np.inf
        self.collision_reward = -np.inf
        self.anomaly_reward = -np.inf

        logger.debug('Task Initialized')

    def reward(self, car_data) -> tuple[float, bool]:
        '''
        Calculate the reward for each task and return the total reward
        '''
        # Define the tasks
        tasks = (
            (UnitTestMode.PROGRESS, 'progress', self.ProgressTask),
            (UnitTestMode.TRACKING, 'tracking', self.TrackingTask),
            (UnitTestMode.COLLISION, 'collision', self.CollisionTask),
            (UnitTestMode.ANOMALY, 'anomaly', self.AnomalyHandlingTask),
        )

        # Calculate individual task rewards
        if self.unit_test == UnitTestMode.DISABLE:
            rewards = []
            for task_mode, weight_key, task_instance in tasks:
                reward = self.task_weights[weight_key] * task_instance.reward(car_data)
                setattr(self, f"{weight_key}_reward", reward)  # Dynamically set the corresponding attribute
                rewards.append(reward)
            total_reward = sum(rewards)
        else:
            # Find the task instance for the unit test
            task = next((t for t in tasks if t[0] == self.unit_test), None)
            if task:
                _, weight_key, task_instance = task
                reward = self.task_weights[weight_key] * task_instance.reward(car_data)
                setattr(self, f"{weight_key}_reward", reward)  # Dynamically set the corresponding attribute
                total_reward = reward
            else:
                total_reward = 0  # Default reward if no matching task is found

        # Log the rewards
        self.log_rewards()

        return total_reward

    def done(self, car_data) -> bool:
        '''
        Check termination conditions from all tasks
        '''
        # Task mapping for done checks
        task_done_mapping = {
            UnitTestMode.PROGRESS: self.ProgressTask.done,
            UnitTestMode.TRACKING: self.TrackingTask.done,
            UnitTestMode.COLLISION: self.CollisionTask.done,
            UnitTestMode.ANOMALY: self.AnomalyHandlingTask.done,
        }

        # Check if done based on the unit_test
        if self.unit_test in task_done_mapping:
            done = task_done_mapping[self.unit_test](car_data)
        else:
            done = (
                self.ProgressTask.done(car_data) or
                self.CollisionTask.done(car_data) or
                self.TrackingTask.done(car_data) or
                self.AnomalyHandlingTask.done(car_data)
            )

        return done


    def reset(self):
        '''
        Reset all tasks
        '''
        self.ProgressTask.reset()
        self.TrackingTask.reset()
        self.CollisionTask.reset()
        self.AnomalyHandlingTask.reset()
    
    def log_rewards(self):
        '''
        Log the rewards for each task
        '''
        # Reward mapping
        reward_mapping = {
            UnitTestMode.DISABLE: f"Progress: {self.progress_reward:.2f} "
                                f"Tracking: {self.tracking_reward:.2f} "
                                f"Collision: {self.collision_reward:.2f} "
                                f"Anomaly: {self.anomaly_reward:.2f}",
            UnitTestMode.PROGRESS: f"Progress: {self.progress_reward:.2f}",
            UnitTestMode.TRACKING: f"Tracking: {self.tracking_reward:.2f}",
            UnitTestMode.COLLISION: f"Collision: {self.collision_reward:.2f}",
            UnitTestMode.ANOMALY: f"Anomaly: {self.anomaly_reward:.2f}",
        }

        # Log the reward based on the unit_test
        logger.debug(reward_mapping.get(self.unit_test, "Unknown Unit Test Mode"))


class MaximizeProgressTask:
    def __init__(self, progress_reward: float = 100.0):
        self._progress_reward = progress_reward
        self.progress_queue = deque(maxlen=5)

    def reward(self, car_data) -> float:
        self.progress_queue.append(float(car_data.progress))
        delta_progress = self.progress_queue[-1] - self.progress_queue[0]
        progress_reward = delta_progress * self._progress_reward

        # Avoiding negative progress 
        if progress_reward < 0.0 and 0 not in self.progress_queue: # 0 is the initial progress value, avoiding to calculate the first
            self.negative_progress_count += 1
            progress_reward *= (1 + 0.5 * self.negative_progress_count)
        else:
            self.negative_progress_count = 0  # Reset if progress is positive
        curr_reward = progress_reward
        return curr_reward
    def done(self, car_data)->bool:
        return False
    def reset(self):
        self.progress_queue = deque(maxlen=5)

class TrackingTask:
    def __init__(self, min_speed: float = 5.0, max_speed: float = 30.0, speed_stability_threshold: float = 0.5, warm_up_steps: int = 10):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_queue = deque(maxlen=10)
        self.speed_stability_threshold = speed_stability_threshold

        # Counter for low-speed duration
        self.low_speed_count = 0
        self.high_speed_count = 0
        self.low_speed_count_max = 30

        # Delay thresholds for low-speed and high-speed penalties
        self.low_speed_delay_threshold = 5
        self.high_speed_delay_threshold = 5
        
        # Warm-up steps to stabilize the speed
        self.warm_up_steps = warm_up_steps
        self.protection_counter = 0

        # Y-position and record
        self.y_position_queue = deque(maxlen=5)
        self.acceleration_z_queue = deque(maxlen=5)

    def reward(self, car_data: CarData) -> float:
        # Get current speed
        speed = np.linalg.norm(car_data.speed)
        self.speed_queue.append(speed)
        self.y_position_queue.append(car_data.y)
        self.acceleration_z_queue.append(car_data.acceleration_z)

        # Initialize reward
        total_reward = 0.0
        
        # (Early Protection)Protection counter to avoid early penalization
        if self.protection_counter < self.warm_up_steps:
            self.protection_counter += 1
            return 0.0

        # \Reward based on speed range===================
        speed_reward = 0.0
        if self.min_speed <= speed <= self.max_speed:
            # Reward for maintaining speed within the range
            speed_reward += 0.1 * (1 - abs(speed - (self.min_speed + self.max_speed) / 2) / ((self.max_speed - self.min_speed) / 2))
            self.low_speed_count = 0
        elif speed < self.min_speed:
            # Low-speed delay counter
            self.low_speed_count += 1
            if self.low_speed_count > self.low_speed_delay_threshold:
                deviation = (self.min_speed - speed) / self.min_speed
                speed_reward -= 0.3 * deviation
        if speed > self.max_speed:
            # High-speed delay counter
            self.high_speed_count += 1
            if self.high_speed_count > self.high_speed_delay_threshold:
                deviation = (speed - self.max_speed) / self.max_speed
                speed_reward -= 0.05 * deviation
        
        total_reward += speed_reward


        # \Calculate speed stability reward==============
        if len(self.speed_queue) == self.speed_queue.maxlen:
            speed_std = np.std(self.speed_queue)
            if speed_std < self.speed_stability_threshold:
                speed_reward += 0.1


        # \Yaw acceleration reward=======================
        if self.acceleration_z_queue ==5 and self.y_position_queue == 5:
            reward_weight = 0.2 # 0.1
            yaw_threshold = 5  # 10
            yaw_reward = 0
            avg_acceleration_z = np.mean(self.acceleration_z_queue)
            if car_data.pitch > 270 and 360 - car_data.pitch > yaw_threshold: # Go up, pitch [350-335]
                if avg_acceleration_z > 0.5: # Encoraging the car to go up
                    yaw_reward += reward_weight * avg_acceleration_z
                else:
                    yaw_reward -= reward_weight * abs(avg_acceleration_z)

            if car_data.pitch < 90 and car_data.pitch > yaw_threshold: # Go down, pitch [10-25]
                if avg_acceleration_z < 0: # Encoraging the car to go down
                    yaw_reward += reward_weight * abs(avg_acceleration_z)
                else:
                    yaw_reward -= reward_weight * abs(avg_acceleration_z)

            total_reward  += yaw_reward

        # \Y Position acceleration reward================
        if len(self.y_position_queue) == 5:
            reward_weight = 0.2
            y_position_reward = 0
            y_position_change = self.y_position_queue[-1] - self.y_position_queue[0]
            avg_acceleration_z = np.mean(self.acceleration_z_queue)
            
            if y_position_change > 1:
                if avg_acceleration_z > 0.5:  # 
                    y_position_reward += reward_weight * avg_acceleration_z
                else: 
                    y_position_reward -= reward_weight * abs(avg_acceleration_z)
            elif y_position_change < -1: 
                if avg_acceleration_z < -0.5: 
                    y_position_reward += reward_weight * abs(avg_acceleration_z)
                else: 
                    y_position_reward -= reward_weight * abs(avg_acceleration_z)

        total_reward  += y_position_reward
        # ==============================================


        return total_reward

    def done(self, car_dat) -> bool:
        # End task if the low-speed count exceeds the limit
        if self.low_speed_count > self.low_speed_count_max:
            return True
        return False

    def reset(self):
        self.speed_queue = deque(maxlen=10)
        self.protection_counter = 0
        self.low_speed_count = 0
        self.high_speed_count = 0
    
class CollisionTask:
    def __init__(self, collision_penalty: float = -10.0, max_obstacle: int = 3):
        self.collision_penalty = collision_penalty
        self.max_obstacle = max_obstacle
        self._last_obstacle = 0

    def reward(self, car_data) -> float:
        '''
        This reward is "penalty" for collision.
        '''
        # 检查是否发生碰撞
        if car_data.obstacle_car > self._last_obstacle:
            self._last_obstacle += 1
            logger.debug("Obstacle collision detected")
            return self.collision_penalty
        return 0.0

    def done(self, car_data) -> bool:
        if car_data.obstacle_car > self.max_obstacle:
            return True    
        return False

    def reset(self): 
        self._last_obstacle = 0

class AnomalyHandlingTask:
    def __init__(self, anomaly_penalty: float = -10.0):
        self.anomaly_penalty = anomaly_penalty
        self.finish = False
        self.anomaly_t_count = 0
        self.anamaly_t_max = 20
    def reward(self, car_data: CarData) -> float:
        '''
        This reward is "penalty" for anomaly.
        '''

        return 0.0
    
    def done(self, car_data: CarData) -> bool:
        # Roll penalty
        if abs(car_data.roll-180) < 178:
            if self.anomaly_t_count < self.anamaly_t_max:
                self.anomaly_t_count += 1
            else:
                self.finish = True
                logger.debug("Roll anomaly detected")
                self.anamaly_t_count = 0
        
        # Yaw penalty
        if car_data.y < 0.4:
            self.finish = True
            logger.debug("Y Position anomaly detected")

        if car_data.speed < 0.5:
            if self.anomaly_t_count < self.anamaly_t_max:
                self.anomaly_t_count += 1
            else:
                self.finish = True
                logger.debug("Speed anomaly detected")
                self.anamaly_t_count = 0
        return self.finish
        

    def reset(self):
        self.finish = False
        self.anomaly_t_count = 0