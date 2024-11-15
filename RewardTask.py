import numpy as np
from logger import logger
from collections import deque

class MixTask:
    def __init__(self, task_weights: dict):
        # Define the weights for different components
        self.task_weights = task_weights or {
            'progress': 1.0,
            'tracking': 1.0,
            'collision': 1.0,
        }
        self.ProgressTask = MaximizeProgressTask()
        self.TrackingTask = TrackingTask(optimal_speed=0,
                                        max_speed=0,
                                        min_speed=0,
                                        speed_tolerance=0,
                                        )
        self.CollisionTask = MaximizeProgressTask()
        # self.TrackingTask = TrackingTask(obs)
        # self.CollisionTask = CollisionDetectionTask(obs)
        
        # Initialize rewards
        self.progress_reward = -np.inf
        self.tracking_reward = -np.inf
        self.collision_reward = -np.inf
        self.speed_reward = -np.inf
        self.steering_reward = -np.inf
        self.stability_reward = -np.inf

        logger.debug('Task Initialized')

    def reward(self, car_data) -> tuple[float, bool]:
        # Calculate individual task rewards
        self.progress_reward = self.task_weights['progress'] * self.ProgressTask.reward(car_data)
        # self.tracking_reward = self.task_weights['tracking'] * self.TrackingTask.reward(states, action)
        # self.collision_reward = self.task_weights['collision'] * self.CollisionTask.reward(states, action)

        # # Additional reward components
        # self.speed_reward = self._compute_speed_reward(states)
        # self.steering_reward = self._compute_steering_penalty(states)
        # self.stability_reward = self._compute_stability_penalty(states)

        # Total reward calculation
        total_reward = (
            self.progress_reward
        )
        # total_reward = (
        #     self.progress_reward +
        #     self.tracking_reward +
        #     self.collision_reward
        # )

        done = self.done(car_data)
        if done:
            total_reward = -10  # Penalty if the episode is done due to failure conditions

        return total_reward, done

    def done(self, states) -> bool:
        # Check termination conditions from all tasks
        done = (
            self.ProgressTask.done(states) or
            self.CollisionTask.done(states) or
            self.TrackingTask.done(states)
        )
        return done

    def reset(self):
        # Reset all tasks
        self.ProgressTask.reset()
        self.TrackingTask.reset()
        self.CollisionTask.reset()

    # def _compute_speed_reward(self, states) -> float:
    #     speed = states['speed']
    #     # Encourage speed between 10 and 30 m/s
    #     if 10 <= speed <= 30:
    #         speed_reward = speed * 0.1
    #     else:
    #         speed_reward = -abs(speed - 20) * 0.1  # Penalize if speed is outside the ideal range
    #     return self.task_weights['speed'] * speed_reward

    # def _compute_steering_penalty(self, states) -> float:
    #     steering_angle = abs(states['steering_angle'])
    #     # Penalize excessive steering angles
    #     steering_penalty = -steering_angle * 0.5 if steering_angle > 0.5 else 0
    #     return self.task_weights['steering'] * steering_penalty

    # def _compute_stability_penalty(self, states) -> float:
    #     angular_velocity_z = abs(states['angular_velocity_z'])
    #     # Penalize high angular velocity for stability
    #     stability_penalty = -angular_velocity_z * 0.1
    #     return self.task_weights['stability'] * stability_penalty

    # def console(self, action, mode: list = ['all'], num=10):
    #     # Console output for debugging and monitoring
    #     console_list = [
    #         f'Progress Reward: {self.progress_reward:.3f}, Tracking Reward: {self.tracking_reward:.3f}, Collision Reward: {self.collision_reward:.3f}',
    #         f'Speed Reward: {self.speed_reward:.3f}, Steering Reward: {self.steering_reward:.3f}, Stability Reward: {self.stability_reward:.3f}',
    #         f'Progress Cumulative: {sum(self.progress_cum_reward):.3f}, Tracking Cumulative: {sum(self.tracking_cum_reward):.3f}, Collision Cumulative: {sum(self.collision_cum_reward):.3f}',
    #         f'Speed Cumulative: {sum(self.speed_cum_reward):.3f}, Steering Cumulative: {sum(self.steering_cum_reward):.3f}, Stability Cumulative: {sum(self.stability_cum_reward):.3f}',
    #         f'motor: {action[0]}, steer: {action[1]}'
    #     ]
    #     if 'all' in mode:
    #         print(console_list)
    #         return
    #     if 'reward' in mode:
    #         print(console_list[0])
    #     if 'cumulative' in mode:
    #         print(console_list[2])
    #     if 'cumulative_range' in mode:
    #         print(f'Progress Range: {sum(self.progress_cum_reward[-num:]):.3f}, Tracking Range: {sum(self.tracking_cum_reward[-num:]):.3f}, Collision Range: {sum(self.collision_cum_reward[-num:]):.3f}')
    #     if 'action' in mode:
    #         print(console_list[4])

class MaximizeProgressTask:
    def __init__(self, progress_reward: float = 100.0):
        self._progress_reward = progress_reward
        self.progress_queue = deque(maxlen=5)

        # --Speed Control-
        self._t_low_motor = int(0)
        self._opitmal_speed = 2
        self._max_speed = 4
        self._min_speed = 0.3
        
        self._t_low_upper = 1000
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

        # Control Speed
        # velocity = np.linalg.norm(states['velocity'])
        # speed_reward = 0.0
        # if self._min_speed < velocity < self._opitmal_speed:
        #     speed_reward = 0.005 * velocity
        # elif velocity >= self._opitmal_speed:
        #     speed_reward = 0.003 - 0.003 * (velocity - self._opitmal_speed)
        # else:#DEL
        #     speed_reward = -0.02 * abs(self._min_speed - velocity)
        # if velocity < 1:
        #     self._t_low_motor += 1
        #     if self._t_low_motor > 100:
        #         curr_reward += -0.003*self._t_low_motor
        # elif velocity > 1:
        #     self._t_low_motor = 0
        #     curr_reward += 0.001 * velocity

        return curr_reward
    def done(self, states)->bool:
        if self._t_low_motor > self._t_low_upper:
            return True
        return False
    def reset(self):
        self.progress_queue = deque(maxlen=5)

class TrackingTask:
    def __init__(self, optimal_speed: float = 2.0, speed_tolerance: float = 0.5, max_speed: float = 4.0, min_speed: float = 0.3):
        self.optimal_speed = optimal_speed
        self.speed_tolerance = speed_tolerance
        self.max_speed = max_speed
        self.min_speed = min_speed

        # Speed queue to calculate speed change rate
        self.speed_queue = deque(maxlen=10)
        self.speed_stability_threshold = 0.2

        # Counter for low-speed duration
        self.low_speed_count = 0
        self.low_speed_limit = 100

    def reward(self, car_data) -> float:
        # Get current speed
        velocity = np.linalg.norm(car_data.velocity)
        self.speed_queue.append(velocity)

        # Initialize reward
        speed_reward = 0.0

        # Reward based on speed stability
        if self.min_speed <= velocity <= self.max_speed:
            # Reward if the speed is within the optimal range
            if abs(velocity - self.optimal_speed) <= self.speed_tolerance:
                speed_reward += 0.05 * (1 - abs(velocity - self.optimal_speed) / self.speed_tolerance)
            # Penalize if the speed deviates from the optimal range
            else:
                speed_reward -= 0.03 * abs(velocity - self.optimal_speed)

        # Evaluate speed stability
        if len(self.speed_queue) == self.speed_queue.maxlen:
            # Calculate the standard deviation of speed
            speed_std = np.std(self.speed_queue)
            # Reward if the standard deviation is low (more stable speed)
            if speed_std < self.speed_stability_threshold:
                speed_reward += 0.05
            else:
                speed_reward -= 0.05 * speed_std

        # Penalize low speed
        if velocity < self.min_speed:
            self.low_speed_count += 1
            speed_reward -= 0.01 * self.low_speed_count
        else:
            self.low_speed_count = 0

        return speed_reward

    def done(self, car_data) -> bool:
        # End task if the low-speed count exceeds the limit
        if self.low_speed_count > self.low_speed_limit:
            return True
        return False

    def reset(self):
        self.speed_queue = deque(maxlen=10)
        self.low_speed_count = 0
