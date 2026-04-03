"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

from utils import NEAR_ZERO, floating_mod_zero
from environment import Environment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sensors import SensorInterface, LandmarkPinger, GPS, Odometry, InsituInstrument

import random
import math
import pandas as pd
import numpy as np

class Robot:
    """
    The Robot class represents a mobile robotic agent. It can execute linear and
    angular velocity commands to move in the world. It can also noisily sense its
    environment with a variety of sensors.

    Attributes:
        env (Environment): the environment this robot is operating in
        cmd_lin_vel (float): most recent linear velocity command
        cmd_ang_vel (float): most recent angular velocity command
        actual_lin_vel (float): most recent executed linear velocity
        actual_ang_vel (float): most recent executed angular velocity
        sensors: list or dict of all robot sensors
    """

    def __init__(
        self,
        env: Environment,
        robot_info: dict = None,
        sensor_info: dict = None,
    ):
        """
        Initialize an instance of the Robot class.
        """
        # TODO: set the environment property to the parameter value
        self.env = env
        self.cmd_lin_vel = 0.0
        self.cmd_ang_vel = 0.0
        self.actual_lin_vel = 0.0
        self.actual_ang_vel = 0.0

        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.current_x_vel = 0.0
        self.current_y_vel = 0.0
        self.drive_mode = "differential"

        if robot_info is not None and sensor_info is not None:
            # simulator.py workflow: config-driven setup
            mtr_info = robot_info["MotorCommands"]
            self.EXECUTION_NOISE_LINEAR = mtr_info["linear_noise"]
            self.EXECUTION_NOISE_ANGULAR = mtr_info["angular_noise"]
            self.execution_noise_linear = self.EXECUTION_NOISE_LINEAR
            self.execution_noise_angular = self.EXECUTION_NOISE_ANGULAR

            gps_info = sensor_info["GPS"]
            lmp_info = sensor_info["LandmarkPinger"]
            odom_info = sensor_info["Odometry"]
            inst_info = sensor_info["InsituInstrument"]
            self.sensors: dict[str, SensorInterface] = {
                "GPS": GPS(
                    robot=self,
                    name="GPS",
                    interval=gps_info["interval"],
                    x_noise=gps_info["x_noise"],
                    y_noise=gps_info["y_noise"],
                ),
                "LandmarkPinger": LandmarkPinger(
                    robot=self,
                    name="LandmarkPinger",
                    interval=lmp_info["interval"],
                    max_range=env.lm_range,
                    range_noise=lmp_info["range_noise_const"],
                    range_prop_noise=lmp_info["range_noise_prop"],
                    bearing_noise=lmp_info["bearing_noise_const"],
                    bearing_prop_noise=lmp_info["bearing_noise_prop"],
                ),
                "Odometry": Odometry(
                    robot=self,
                    name="Odometry",
                    interval=odom_info["interval"],
                    lin_noise=odom_info["linear_noise_const"],
                    linear_noise_ratio=odom_info["linear_noise_prop"],
                    ang_noise=odom_info["angular_noise_const"],
                    angular_noise_ratio=odom_info["angular_noise_prop"],
                ),
                "InsituInstrument": InsituInstrument(
                    robot=self,
                    name="InsituInstrument",
                    interval=inst_info["interval"],
                    noise=inst_info["noise"],
                ),
            }
        else:
            # main.py workflow: manual sensor setup
            self.sensors = []
            self.noise_x = 0.05
            self.noise_y = 0.05
            self.noise_theta = 0.03
            self.execution_noise_linear = 0.05
            self.execution_noise_angular = 0.03
            self.EXECUTION_NOISE_LINEAR = self.execution_noise_linear
            self.EXECUTION_NOISE_ANGULAR = self.execution_noise_angular

        # initialize the robot's belief (for informative sampling)
        if env.continuous_field is not None:
            self.kernel = env.continuous_field.kernel
            self.belief = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=15,
                random_state=env.continuous_field.random_seed,
            )
        else:
            self.belief = None
        self.pose_history = []
        self.observation_history = []

    # --- Robot Belief Methods ---
    def update_belief(self, measurement, pose):
        """
        Updates the robot's belief based on a located-observation.

        Inputs:
            measurement (float): value of the field being measured
            pose (Position): location from where the measurement was taken
        """
        if self.belief is None:
            return
        self.pose_history.append((pose.pos.x, pose.pos.y))
        self.observation_history.append(measurement)
        self.belief.fit(np.asarray(self.pose_history), np.asarray(self.observation_history))

    # --- Controller Methods ---
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
        self.cmd_lin_vel = lin_vel
        self.cmd_ang_vel = ang_vel

        # noisify the execution proportionally
        lin_vel = lin_vel * (1 + random.gauss(0, self.execution_noise_linear))
        ang_vel = ang_vel * (1 + random.gauss(0, self.execution_noise_angular))

        # current recording
        self.current_lin_vel = lin_vel
        self.current_ang_vel = ang_vel
        self.actual_lin_vel = lin_vel
        self.actual_ang_vel = ang_vel

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
        # noisify the execution proportionally
        x_vel = x_vel * (1 + random.gauss(0, self.noise_x))
        y_vel = y_vel * (1 + random.gauss(0, self.noise_y))
        ang_vel = ang_vel * (1 + random.gauss(0, self.noise_theta))

        # current recording (translational mode)
        self.drive_mode = "translational"
        self.current_x_vel = x_vel
        self.current_y_vel = y_vel
        self.current_ang_vel = ang_vel
        self.actual_lin_vel = math.sqrt(x_vel**2 + y_vel**2)
        self.actual_ang_vel = ang_vel

        dx = x_vel * self.env.DT
        dy = y_vel * self.env.DT
        dtheta = ang_vel * self.env.DT

        return dx, dy, dtheta

    # --- Sensing Methods ---
    def take_sensor_measurements(self) -> pd.DataFrame:
        """
        Return noisy sensor readings of the environment at this timestep.
        """
        if isinstance(self.sensors, dict):
            measurements = pd.DataFrame({"Time": [self.env.time]})
            for s in self.sensors.values():
                if floating_mod_zero(self.env.time, s.interval):
                    measurements = pd.merge(
                        measurements,
                        s.sample(),
                        left_index=True,
                        right_index=True,
                    )
            measurements["CMD_LinearVelocity"] = [self.cmd_lin_vel]
            measurements["CMD_AngularVelocity"] = [self.cmd_ang_vel]
        else:
            measurements = pd.DataFrame({"time": [self.env.time]})
            for sensor in self.sensors:
                sensor_data = sensor.sample()
                if 'time' in sensor_data.columns:
                    sensor_data = sensor_data.drop(columns=['time'])
                measurements = pd.concat([measurements, sensor_data], axis=1)
        return measurements

    def take_state_snapshot(self) -> pd.DataFrame:
        """
        Return timestep-specific GT data for CSV logging.
        """
        env_data = self.env.take_state_snapshot()
        env_data["Actual_LinearVelocity"] = self.actual_lin_vel
        env_data["Actual_AngularVelocity"] = self.actual_ang_vel
        return env_data

    def info(self) -> pd.DataFrame:
        """
        Return a dictionary of frozen environment information.
        """
        # set up the table
        columns = ["Sensor Name", "Constant Noise", "Proportional Noise", "Model"]
        data = []

        lin_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
        lin_row["Sensor Name"] = "MotorControllerLinear"
        lin_row["Constant Noise"] = self.EXECUTION_NOISE_LINEAR
        data.append(lin_row)

        ang_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
        ang_row["Sensor Name"] = "MotorControllerAngular"
        ang_row["Constant Noise"] = self.EXECUTION_NOISE_ANGULAR
        data.append(ang_row)

        belief_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
        belief_row["Sensor Name"] = "belief"
        belief_row["Model"] = self.belief
        data.append(belief_row)

        if not isinstance(self.sensors, dict):
            return pd.concat(data)

        for name, sensor in self.sensors.items():
            if isinstance(sensor, GPS):
                row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                row["Sensor Name"] = name + "X"
                row["Constant Noise"] = sensor.X_NOISE
                data.append(row)
                row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                row["Sensor Name"] = name + "Y"
                row["Constant Noise"] = sensor.Y_NOISE
                data.append(row)
            elif isinstance(sensor, Odometry):
                lin_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                lin_row["Sensor Name"] = name + "Linear"
                lin_row["Constant Noise"] = sensor.LIN_NOISE
                lin_row["Proportional Noise"] = sensor.LINEAR_NOISE_RATIO
                data.append(lin_row)
                ang_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                ang_row["Sensor Name"] = name + "Angular"
                ang_row["Constant Noise"] = sensor.ANG_NOISE
                ang_row["Proportional Noise"] = sensor.ANGULAR_NOISE_RATIO
                data.append(ang_row)
            elif isinstance(sensor, LandmarkPinger):
                range_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                range_row["Sensor Name"] = name + "Range"
                range_row["Constant Noise"] = sensor.RANGE_PROP_NOISE
                range_row["Proportional Noise"] = sensor.RANGE_PROP_NOISE
                data.append(range_row)
                bearing_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                bearing_row["Sensor Name"] = name + "Angular"
                bearing_row["Constant Noise"] = sensor.BEARING_NOISE
                data.append(bearing_row)
            elif isinstance(sensor, InsituInstrument):
                inst_row = pd.DataFrame(0, index=pd.RangeIndex(1), columns=columns)
                inst_row["Sensor Name"] = name
                inst_row["Constant Noise"] = sensor.noise
                data.append(inst_row)

        return pd.concat(data)
