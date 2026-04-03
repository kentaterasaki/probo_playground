"""
An abstract base class that all sensor classes must inherit from. This structure guarantees that all sensors have certain traits, including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants. Different sensors may use different distributions to model noise, and may take in different parameters to shape that noise. For example, one sensor might have a constant noise mean, while another might have noise that grows proportionally with distance or time.

Exteroceptive sensors measure the robot's relationship to the world. This includes GPS, cameras, LiDAR, and anything else that takes a measurement that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states. This includes IMUs, wheel encoders, and anything else that measures how the robot's state is relatively changing, without relating the robot to the world.
"""

# from robot import Robot
from utils import BearingRange, Position, SEED
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from math import pi
import random
from sympy.abc import x, y, theta, j, k
from sympy import Matrix, Symbol, sqrt, atan2
import math
import pandas as pd
import numpy as np
from utils import BearingRange


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float):
        """
        Initialize a sensor class instace.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.robot = robot
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """
        Getter for the name property.
        """
        return self._name

    @property
    def interval(self) -> float:
        """
        Getter for the interval property.
        """
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """
        Getter for the time of last measurement property.
        """
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """
        Setter for the time of last measurement property.
        """
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """
        Sample the environment and return the noisy measurement(s).
        """
        pass


class WheelEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.
    - Differential drive: linear and angular velocities
    - Translational drive: x, y, and angular velocities

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        LIN_NOISE: absolute noise for linear velocity stdev
        ANG_NOISE: absolute noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        name="wheel_encoder",
        interval=0.1,
        lin_noise=0.05,
        ang_noise=0.03,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        # TODO: save all noise constants as properties
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s

    def sample(self):
        """
        Sample the robot's linear and angular velocity.
        """
        # TODO: fill in the function
        ang_vel = self.robot.current_ang_vel * (1 + random.gauss(0, self.ANG_NOISE))

        # Translational (swerve) mode: report x_vel and y_vel
        if self.robot.drive_mode == "translational":
            x_vel = self.robot.current_x_vel * (1 + random.gauss(0, self.LIN_NOISE))
            y_vel = self.robot.current_y_vel * (1 + random.gauss(0, self.LIN_NOISE))
            return pd.DataFrame({
                "time": [self.robot.env.time],
                "encoder_x_vel": [x_vel],
                "encoder_y_vel": [y_vel],
                "encoder_ang_vel": [ang_vel],
            })
        # Differential drive mode: report lin_vel
        else:
            lin_vel = self.robot.current_lin_vel * (1 + random.gauss(0, self.LIN_NOISE))
            return pd.DataFrame({
                "time": [self.robot.env.time],
                "encoder_lin_vel": [lin_vel],
                "encoder_ang_vel": [ang_vel],
            })


class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing between the robot and the floating-point landmarks on the map. In practice, this sensor could be a ToF sensor, a node in a network of beacons, or even a camera.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        MAX_RANGE (int): maximum distance from a beacon for it to be visible
        RANGE_NOISE (float): absolute noise for range stdev
        RANGE_NOISE_RATIO (float): porportional noise for range stdev
        BEARING_NOISE (float): absolute noise for bearing stdev
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        range_noise=0.5,
        range_prop_noise=0.05,
        bearing_noise=pi / 6,
        bearing_prop_noise=0.0,
        max_range=5.0,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
        """
        super().__init__(name, robot, interval)
        # TODO: save max range and all noise constants as properties
        self.MAX_RANGE = max_range  # meters
        self.RANGE_NOISE = range_noise  # meters
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise  # radians
        self.BEARING_PROP_NOISE = bearing_prop_noise  # radians

        # TODO: define the nonlinear measurement model symbolically
        self.h_x: Matrix = Matrix(
            [
                [sqrt((j-x)**2 + (k-y)**2)],  # calculation of r (range)
                [atan2(k-y, j-x) - theta],  # calculation of phi (bearing)
            ]
        )

        # TODO: define the Jacobian of h(x) symbolically
        self.H: Matrix = self.h_x.jacobian([x, y, theta])

        self.subs: dict[Symbol, float] = {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            k: 0.0,
            j: 0.0,
        }

    def sample(self):
        """
        Reports noisy measurements of the bearing and range between the robot and all nearby landmarks.
        """
        landmarks_data_noisy = pd.DataFrame()
        gt_prox_to_landmarks = self.robot.env.get_proximity_to_landmarks()
        for landmark in gt_prox_to_landmarks.columns:
            gt : BearingRange = gt_prox_to_landmarks[landmark].values[0]
            if gt.range <= self.MAX_RANGE:
                noisy_bearing = gt.bearing + random.gauss(0, self.BEARING_NOISE)
                noisy_range = gt.range + random.gauss(0, self.RANGE_NOISE + self.RANGE_PROP_NOISE * gt.range)
                br_noisy = BearingRange(gt.landmark_id, noisy_bearing, noisy_range)
            else:
                # Out of range
                br_noisy = BearingRange(gt.landmark_id, float('inf'), float('inf'))
            landmarks_data_noisy[f"{self.name}_{landmark}"] = [br_noisy]
        return landmarks_data_noisy

    def R(self, z):
        """
        Estimate variance of a given pinger measurement.

        Args:
            z (ndarray): pinger observation [[range 0], [0 bearing]]

        Returns:
            Sensor noise model for pinger measurement
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + z[0] * self.RANGE_PROP_NOISE
        return np.diag([range_stdev, bearing_stdev]) ** 2

    def H_eval(self, state_vec, lm_id):
        """
        Evaluate the Jacobian of h(x) at x, which reshapes a state vector to be in the observation space. This matrix is used to turn a state prediction into an observation prediction for a specific landmark.

        Args:
            x: the current state vector, to linearize with respect to
            lm_id: the ID of the landmark that we are predicting an observation of
        """
        # TODO: find the x and y position of the given landmark
        landmark = None
        for lm in self.robot.env.LANDMARKS:
            if lm.id == lm_id:
                landmark = lm
                break
        
        if landmark is None:
            raise ValueError(f"Landmark with ID {lm_id} not found")
        
        lm_x = landmark.pos.x
        lm_y = landmark.pos.y

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        from sympy.abc import x, y, theta, j, k
        self.subs[x] = state_vec[0]
        self.subs[y] = state_vec[1]
        self.subs[theta] = state_vec[2]
        self.subs[j] = lm_x  # note: we use j for landmark x position
        self.subs[k] = lm_y  # note: we use k for landmark y position

        # TODO: evaluate the Jacobian at the subs values and convert it to a numpy array
        H_eval = np.array(self.H.subs(self.subs)).astype(np.float64)

        # return
        return H_eval

    def y(self, z, state_vec, lm_id):
        """
        Calculate the residual between an observation x and a predicted observation derived from a predicted state. The predicted observation is in reference to a specified landmark.
        """
        # TODO: find the x and y position of the given landmark
        landmark = None
        for lm in self.robot.env.LANDMARKS:
            if lm.id == lm_id:
                landmark = lm
                break
        
        if landmark is None:
            raise ValueError(f"Landmark with ID {lm_id} not found")
        
        lm_x = landmark.pos.x
        lm_y = landmark.pos.y

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        from sympy.abc import x, y, theta, j, k
        self.subs[x] = state_vec[0]
        self.subs[y] = state_vec[1]
        self.subs[theta] = state_vec[2]
        self.subs[j] = lm_x  # note: we use j for landmark x position
        self.subs[k] = lm_y  # note: we use k for landmark y position

        # TODO: evaluate the measurement model at the subs values and convert it to a numpy array
        hx_eval = np.array(self.h_x.subs(self.subs)).astype(np.float64).flatten()

        # TODO: calculate the residual
        residual = z - hx_eval
        
        # Wrap the bearing component (index 1) to [-pi, pi]
        from utils import wrap_angle
        residual[1] = wrap_angle(residual[1])

        # return
        return residual

class GPS(SensorInterface):
    """
    This class represents a GPS sensor that measures the position of the robot in 2D space.

    Attributes:
        name (str): string identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        last_meas_t (float): time of last measurement
        X_NOISE (float): absolute noise for x stdev
        Y_NOISE (float): absolute noise for y stdev
    """

    def __init__(
        self,
        robot,
        name,
        interval,
        x_noise,
        y_noise,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            x_noise (float): absolute noise for x stdev
            y_noise (float): absolute noise for y stdev
        """
        super().__init__(name, robot, interval)
        self.X_NOISE = x_noise
        self.Y_NOISE = y_noise

        # TODO: fill in the measurement model
        # m (num of measurements) x (num of states)
        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        # TODO: fill in the noise model
        self.R = np.array([
            [self.X_NOISE**2, 0.0],
            [0.0, self.Y_NOISE**2],
        ])

    def sample(self):
        """
        Take a noisy GPS measurement of robot position.
        """
        pose = self.robot.env.robot_pose
        noisy_x = pose.pos.x + random.gauss(0, self.X_NOISE)
        noisy_y = pose.pos.y + random.gauss(0, self.Y_NOISE)
        return pd.DataFrame({
            "time": [self.robot.env.time],
            "gps_x": [noisy_x],
            "gps_y": [noisy_y],
            "H": [self.H],
            "R": [self.R],
        })


class InsituInstrument(SensorInterface):
    """
    This class represents a sensor that measures from a continuous field in an environment.

    Attributes:
        name (str): reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        noise (float): noise of a scalar measurement
    """

    def __init__(
        self,
        robot,
        name,
        interval,
        noise,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            noise (float): noise of a scalar measurement
        """
        super().__init__(name, robot, interval)
        self.noise = noise  # noise character of the sensor

    def sample(self) -> pd.DataFrame:
        """
        Noisily measure the in situ status of the continuous field.

        Returns:
            A dictionary containing the noisy field measurement.
        """
        field_measurement = self.robot.env.get_gt_field_value()
        noisy_measurement = random.gauss(field_measurement, self.noise)
        return pd.DataFrame({self.name: noisy_measurement})

# --- Proprioceptive Sensors ---
# these measure the robot's physical relationship to its prior states
class Odometry(SensorInterface):
    """
    This class represents an odometry sensor that measures the robot's velocity based on wheel encoders.
    Reports noisy estimates of linear and angular velocities.

    Attributes:
        name (str): string identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        last_meas_t (float): time of last measurement
        LINEAR_NOISE_RATIO (float): proportional noise for linear velocity stdev
        ANGULAR_NOISE_RATIO (float): proportional noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        name,
        interval,
        lin_noise,
        ang_noise,
        linear_noise_ratio,
        angular_noise_ratio,
    ):
        """
        Initialize an instance of the Odometry class.

        Args:
            robot (Robot): reference robot
            name (str): reference identifier
            interval (float): period between measurements
            linear_noise_ratio (float): proportional noise for linear velocity
            angular_noise_ratio (float): proportional noise for angular
        """
        super().__init__(name, robot, interval)
        self.LIN_NOISE = lin_noise
        self.ANG_NOISE = ang_noise
        self.LINEAR_NOISE_RATIO = linear_noise_ratio
        self.ANGULAR_NOISE_RATIO = angular_noise_ratio

    def sample(self) -> pd.DataFrame:
        """
        Take a noisy odometry measurement of the robot's velocities.

        Returns:
            A dictionary containing noisy linear and angular velocity estimates.
        """
        # Get the commanded velocities from the robot
        true_lin_vel = self.robot.actual_lin_vel
        true_ang_vel = self.robot.actual_ang_vel

        # Add proportional noise
        noisy_lin_vel = random.gauss(
            true_lin_vel, self.LIN_NOISE + abs(true_lin_vel) * self.LINEAR_NOISE_RATIO
        )
        noisy_ang_vel = random.gauss(
            true_ang_vel, self.ANG_NOISE + abs(true_ang_vel) * self.ANGULAR_NOISE_RATIO
        )

        return pd.DataFrame(
            {
                f"{self.name}_LinearVelocity": [noisy_lin_vel],
                f"{self.name}_AngularVelocity": [noisy_ang_vel],
            }
        )


# NOTE: the IMU and VIO classes were both generated by Claude, I haven't checked them or rewritten it myself yet
# class IMU(SensorInterface):
#     """
#     This class represents an IMU (Inertial Measurement Unit) that measures linear and angular acceleration.
#     Note: IMUs suffer from bias drift over time, making long-term integration challenging.

#     Attributes:
#         name (str): string identifier
#         robot (Robot): reference robot
#         interval (float): period between measurements
#         last_meas_t (float): time of last measurement
#         LINEAR_ACCEL_NOISE (float): absolute noise for linear acceleration stdev (m/s²)
#         ANGULAR_ACCEL_NOISE (float): absolute noise for angular acceleration stdev (rad/s²)
#         LINEAR_BIAS_DRIFT (float): bias drift rate for linear acceleration (m/s² per second)
#         ANGULAR_BIAS_DRIFT (float): bias drift rate for angular acceleration (rad/s² per second)
#     """

#     def __init__(
#         self,
#         robot: Robot,
#         name: str = "IMU",
#         interval: float = 0.01,  # IMUs are very fast, 100 Hz
#         linear_accel_noise: float = 0.05,
#         angular_accel_noise: float = 0.05,
#         linear_bias_drift: float = 0.001,
#         angular_bias_drift: float = 0.001,
#     ):
#         """
#         Initialize an instance of the IMU class.

#         Args:
#             robot (Robot): reference robot
#             name (str): reference identifier
#             interval (float): period between measurements
#             linear_accel_noise (float): absolute noise for linear acceleration (m/s²)
#             angular_accel_noise (float): absolute noise for angular acceleration (rad/s²)
#             linear_bias_drift (float): bias drift rate for linear acceleration (m/s² per second)
#             angular_bias_drift (float): bias drift rate for angular acceleration (rad/s² per second)
#         """
#         super().__init__(name, robot, interval)
#         self.LINEAR_ACCEL_NOISE = linear_accel_noise
#         self.ANGULAR_ACCEL_NOISE = angular_accel_noise
#         self.LINEAR_BIAS_DRIFT = linear_bias_drift
#         self.ANGULAR_BIAS_DRIFT = angular_bias_drift

#         # IMU bias (slowly drifting offset)
#         self.linear_bias = 0.0
#         self.angular_bias = 0.0

#         # Track previous velocities to compute acceleration
#         self.prev_lin_vel = 0.0
#         self.prev_ang_vel = 0.0
#         self.prev_time = 0.0

#     def sample(self) -> dict[str, float]:
#         """
#         Take a noisy IMU measurement of linear and angular acceleration.

#         Returns:
#             A dictionary containing noisy linear and angular acceleration estimates.
#         """
#         current_time = self.robot.env.time

#         # Compute true accelerations from velocity changes
#         if self.prev_time > 0:
#             dt = current_time - self.prev_time
#             if dt > 0:
#                 true_linear_accel = (self.robot.last_lin_vel - self.prev_lin_vel) / dt
#                 true_angular_accel = (self.robot.last_ang_vel - self.prev_ang_vel) / dt
#             else:
#                 true_linear_accel = 0.0
#                 true_angular_accel = 0.0
#         else:
#             true_linear_accel = 0.0
#             true_angular_accel = 0.0

#         # Update bias (random walk)
#         time_delta = current_time - self.prev_time if self.prev_time > 0 else 0.0
#         self.linear_bias += random.gauss(0, self.LINEAR_BIAS_DRIFT * time_delta)
#         self.angular_bias += random.gauss(0, self.ANGULAR_BIAS_DRIFT * time_delta)

#         # Add noise and bias
#         noisy_linear_accel = (
#             true_linear_accel
#             + self.linear_bias
#             + random.gauss(0, self.LINEAR_ACCEL_NOISE)
#         )
#         noisy_angular_accel = (
#             true_angular_accel
#             + self.angular_bias
#             + random.gauss(0, self.ANGULAR_ACCEL_NOISE)
#         )

#         # Update previous values for next sample
#         self.prev_lin_vel = self.robot.last_lin_vel
#         self.prev_ang_vel = self.robot.last_ang_vel
#         self.prev_time = current_time

#         return {
#             "linear_acceleration": noisy_linear_accel,
#             "angular_acceleration": noisy_angular_accel
#         }

# class VisualOdometry(SensorInterface):
#     """
#     Estimates motion from visual feature tracking.
#     """
#     def __init__(
#         self,
#         robot: Robot,
#         name: str = "VisualOdometry",
#         interval: float = 0.1,
#         translation_noise: float = 0.03,
#         rotation_noise: float = 0.08,
#         failure_rate: float = 0.05,  # 5% chance of tracking loss
#     ):
#         super().__init__(name, robot, interval)
#         self.TRANSLATION_NOISE = translation_noise
#         self.ROTATION_NOISE = rotation_noise
#         self.FAILURE_RATE = failure_rate
#         self.prev_pose = None

#     def sample(self):
#         """
#         Estimate motion since last measurement.
#         """
#         current_pose = self.robot.env.get_gt_robot_pose()

#         # First measurement - no motion estimate yet
#         if self.prev_pose is None:
#             self.prev_pose = current_pose
#             return None

#         # Simulate tracking failure
#         if random.random() < self.FAILURE_RATE:
#             self.prev_pose = current_pose
#             return None

#         # Compute true displacement
#         dx = current_pose.pos.x - self.prev_pose.pos.x
#         dy = current_pose.pos.y - self.prev_pose.pos.y
#         dtheta = current_pose.theta - self.prev_pose.theta
#         dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

#         # Add noise
#         noisy_dx = random.gauss(dx, abs(dx) * self.TRANSLATION_NOISE + 0.01)
#         noisy_dy = random.gauss(dy, abs(dy) * self.TRANSLATION_NOISE + 0.01)
#         noisy_dtheta = random.gauss(dtheta, abs(dtheta) * self.ROTATION_NOISE + 0.01)

#         self.prev_pose = current_pose

#         return {
#             "dx": noisy_dx,
#             "dy": noisy_dy,
#             "dtheta": noisy_dtheta
#         }
