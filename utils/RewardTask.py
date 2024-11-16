import numpy as np
from utils.logger import logger
from collections import deque

class MixTask:
    def __init__(self, task_weights: dict):
        '''
        Initialize the tasks with the given weights
        '''
        self.task_weights = task_weights or {
            'progress': 1.0,
            'tracking': 1.0,
            'collision': 1.0,
        }
        self.ProgressTask = MaximizeProgressTask(progress_reward=100.0)
        self.TrackingTask = TrackingTask(min_speed=5.0, max_speed=30.0, speed_stability_threshold=1)
        self.CollisionTask = CollisionTask(collision_penalty=-10.0, max_obstacle=5)
        
        # Initialize rewards
        self.progress_reward = -np.inf
        self.tracking_reward = -np.inf
        self.collision_reward = -np.inf

        logger.debug('Task Initialized')

    def reward(self, car_data) -> tuple[float, bool]:
        '''
        Calculate the reward for each task and return the total reward
        - Calcuates the reward for each task
        - Sum the rewards
        - Check if the episode is done
        - Return the total reward and done flag
        '''
        # Calculate individual task rewards
        self.progress_reward = self.task_weights['progress'] * self.ProgressTask.reward(car_data)
        self.tracking_reward = self.task_weights['tracking'] * self.TrackingTask.reward(car_data)
        self.collision_reward = self.task_weights['collision'] * self.CollisionTask.reward(car_data)

        # Total reward calculation
        rewards = (
            self.progress_reward,
            self.tracking_reward,
            self.collision_reward,
        )
        total_reward = sum(rewards)

        # Log the rewards
        self.log_rewards()

        # Check if the episode is done
        done = self.done(car_data)
        if done:
            total_reward = -10  # Penalty if the episode is done due to failure conditions

        return total_reward, done

    def done(self, car_data) -> bool:
        '''
        Check termination conditions from all tasks
        '''
        done = (
            self.ProgressTask.done(car_data) or
            self.CollisionTask.done(car_data) or
            self.TrackingTask.done(car_data)
        )
        return done

    def reset(self):
        '''
        Reset all tasks
        '''
        self.ProgressTask.reset()
        self.TrackingTask.reset()
        self.CollisionTask.reset()
    
    def log_rewards(self):
        '''
        Log the rewards for each task
        '''
        logger.debug(f"Progress: {self.progress_reward:.2f} \
                       Tracking: {self.tracking_reward:.2f}, \
                       Collision: {self.collision_reward:.2f}")
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
        self.low_speed_limit = 30

        # Delay thresholds for low-speed and high-speed penalties
        self.low_speed_delay_threshold = 5
        self.high_speed_delay_threshold = 5
        
        # Warm-up steps to stabilize the speed
        self.warm_up_steps = warm_up_steps
        self.protection_counter = 0



    def reward(self, car_data) -> float:
        # Get current speed
        speed = np.linalg.norm(car_data.speed)
        self.speed_queue.append(speed)

        # Initialize reward
        speed_reward = 0.0
        
        # (Early Protection)Protection counter to avoid early penalization
        if self.protection_counter < self.warm_up_steps:
            self.protection_counter += 1
            return 0.0

        # ========Reward based on speed range============
        if speed <= self.min_speed <= self.max_speed:
            # Reward for maintaining speed within the range
            speed_reward += 0.1 * (1 - abs(speed - (self.min_speed + self.max_speed) / 2) / ((self.max_speed - self.min_speed) / 2))
        elif speed < self.min_speed:
            # Low-speed delay counter
            self.low_speed_count += 1
            if self.low_speed_count > self.low_speed_delay_threshold:
                deviation = (self.min_speed - speed) / self.min_speed
                speed_reward -= 0.1 * deviation
        else:
            self.low_speed_count = 0

        if speed > self.max_speed:
            # High-speed delay counter
            self.high_speed_count += 1
            if self.high_speed_count > self.high_speed_delay_threshold:
                deviation = (speed - self.max_speed) / self.max_speed
                speed_reward -= 0.05 * deviation
        # ==============================================


        # ======Calculate speed stability reward========
        if len(self.speed_queue) == self.speed_queue.maxlen:
            speed_std = np.std(self.speed_queue)
            if speed_std < self.speed_stability_threshold:
                speed_reward += 0.1
        # ==============================================

        return speed_reward

    def done(self, car_dat) -> bool:
        # End task if the low-speed count exceeds the limit
        if self.low_speed_count > self.low_speed_limit:
            return True
        return False

    def reset(self):
        self.speed_queue = deque(maxlen=10)
        self.protection_counter = 0
        self.low_speed_count = 0
        self.high_speed_count = 0
    
class CollisionTask:
    def __init__(self, collision_penalty: float = -10.0, max_obstacle: int = 5):
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
            return self.collision_penalty
        return 0.0

    def done(self, car_data) -> bool:
        if car_data.obstacle_car > self.max_obstacle:
            return True    
        return False

    def reset(self): 
        self._last_obstacle = 0