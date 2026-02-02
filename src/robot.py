"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

from utils import NEAR_ZERO
from environment import Environment
from sensors import SensorInterface

import random
import math
import pandas as pd

class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        # TODO: set the environment property to the parameter value
        self.env = env
        # TODO: initialize the sensors property as an empty list
        self.sensors = []

        self.current_lin_vel = 0.0 # m/s
        self.current_ang_vel = 0.0 # rad/s
        
        self.noise_x = 0.05
        self.noise_y = 0.05
        self.noise_theta = 0.03
        self.execution_noise_linear = 0.05
        self.execution_noise_angular = 0.03

    def robot_step_differential(self, lin_vel: float, ang_vel: float):
        """
        Differential-drive mode. Given forward linear and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        # TODO: fill in the function

        # noisify the execution proportionally
        lin_vel = lin_vel * (1 + random.gauss(0, self.execution_noise_linear))
        ang_vel = ang_vel * (1 + random.gauss(0, self.execution_noise_angular))

        # current recording
        self.current_lin_vel = lin_vel
        self.current_ang_vel = ang_vel

        # linear only; drive in a straight line
        if abs(ang_vel) < NEAR_ZERO:
            dx = lin_vel * self.env.DT * math.cos(self.env.robot_pose.theta)
            dy = lin_vel * self.env.DT * math.sin(self.env.robot_pose.theta)
            dtheta = 0.0
        # linear and angular; drive in an arc
        else:
            r = lin_vel / ang_vel
            dtheta = ang_vel * self.env.DT
            dx = r * (
                math.sin(self.env.robot_pose.theta + dtheta)
                - math.sin(self.env.robot_pose.theta)
            )
            dy = -r * (
                math.cos(self.env.robot_pose.theta + dtheta)
                - math.cos(self.env.robot_pose.theta)
            )

        return dx, dy, dtheta

    def robot_step_translational(self, x_vel: float, y_vel: float, ang_vel: float):
        """
        Swerve-drive mode. Given x, y, and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        # TODO: fill in the function
        dx = x_vel * self.env.DT * (1 + random.gauss(0, self.noise_x))
        dy = y_vel * self.env.DT * (1 + random.gauss(0, self.noise_y))
        dtheta = ang_vel * self.env.DT * (1 + random.gauss(0, self.noise_theta))
        

        return dx, dy, dtheta

    def take_sensor_measurements(self) -> pd.DataFrame:
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        # TODO: fill in the function
        measurements = pd.DataFrame({"time": [self.env.time]})
        for sensor in self.sensors:
            sensor_data = sensor.sample()
            if 'time' in sensor_data.columns:
                sensor_data = sensor_data.drop(columns=['time'])
            measurements = pd.concat([measurements, sensor_data], axis=1)
        return measurements
