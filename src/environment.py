"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

from utils import Position, Pose, BearingRange, Bounds, Landmark
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from itertools import product
import pandas as pd
import numpy as np
import math


class Field:
    """Creates a continuous function that can be sampled.
    
    Attributes:
        DIMS (Bounds): the four corners of the environment
        variance (float): the variance of the GP kernel
        lengthscale (float): the lengthscale of the GP kernel
        random_seed (int): random seed for setting world draw
    """
    def __init__(
        self,
        dimensions: Bounds,
        variance: float = 0.1,
        lengthscale: float = 1.0,
        random_seed: int = 10,
    ):
        """
        Initialize the continuous field in an environment.

        Args:
            dimensions (Bounds): the four corners of the environment
            variance (float): the variance of the GP kernel
            lengthscale (float): the lengthscale of the GP kernel
            random_seed (int): random seed for setting consistent world draw
        """
        self.DIMS = dimensions
        self.variance = variance
        self.lengthscale = lengthscale
        self.random_seed = random_seed
        self._initialize_field()

    def _initialize_field(self):
        """Initializes the continuous field in an environment."""
        self.kernel = ConstantKernel(1.0, (1e-3, 1e-3)) * RBF([self.lengthscale, self.lengthscale], (self.variance, 100*self.variance))
        field = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=15, random_state=self.random_seed)
        x, y = np.linspace(self.DIMS.x_min, self.DIMS.x_max, 20), np.linspace(self.DIMS.y_min, self.DIMS.y_max, 20)
        M = np.array(list(product(x, y)))
        init_sample = field.sample_y(M, 1, random_state=self.random_seed)
        field.fit(M, init_sample)
        self.field = field

    def info(self) -> dict:
        return {"Variance": self.variance,
                "Lengthscale": self.lengthscale,
                "Random Seed": self.random_seed,
                "Model": self.field}


class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float = None,
        obstacles: list[Bounds] = None,
        landmarks: list[Landmark] = None,
        robot_starting_pose: Pose = None,
        field: Field = None,
        timestep: float = 0.1,
        lm_range: float = 10.0,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions (Bounds): the four corners of the environment
            dt (float): timestep size (used by main.py positional call)
            obstacles (list[Bounds]): a list of all intraversible areas
            landmarks (list[Landmark]): a list of all identifiable landmarks
            robot_starting_pose (Pose): the initial position and heading of the robot
            field (Field): continuous field for informative sampling
            timestep (float): timestep size (used by simulator.py keyword call)
            lm_range (float): detectable range of landmarks
        """
        # TODO: set the dimensions property to the parameter value
        self.DIMENSIONS = dimensions
        self.DT = dt if dt is not None else timestep
        self.time = 0
        self.OBSTACLES = obstacles if obstacles is not None else []
        self.LANDMARKS = landmarks if landmarks is not None else []
        self.robot_pose = robot_starting_pose if robot_starting_pose is not None else Pose(Position(0, 0), 0)
        self.lm_range = lm_range
        self.continuous_field = field

    # --- Motion Execution ---

    def robot_step(self, dx: float, dy: float, dtheta: float):
        """
        Update the robot's position and heading in the world. The robot should not be able to pass through obstacles or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        # TODO: fill in the function
        new_pos = self.is_valid_motion(dx, dy)

        updated_theta = self.robot_pose.theta + dtheta
        updated_theta = (updated_theta + math.pi) % (2 * math.pi) - math.pi

        self.time = round(self.time + self.DT, 3)
        self.robot_pose = Pose(new_pos, updated_theta)


    def is_valid_motion(self, dx: float, dy: float):
        """
        Given attempted x and y motion by the robot, determine what motion is physically possible (i.e. doesn't go through any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """
        # TODO: fill in the function
        x_new = self.robot_pose.pos.x
        if self.is_valid_position(Position(self.robot_pose.pos.x + dx, self.robot_pose.pos.y)):
            x_new = self.robot_pose.pos.x + dx

        y_new = self.robot_pose.pos.y
        if self.is_valid_position(Position(x_new, self.robot_pose.pos.y + dy)):
            y_new = self.robot_pose.pos.y + dy

        return Position(x_new, y_new)

    
    def is_valid_position(self, position: Position):
        """
        Check if a given robot position is valid; i.e. not out-of-bounds or within an obstacle. Return a boolean representing whether or not this condition is true.

        Args:
            position: the robot position

        Returns:
            true if the position is valid and false otherwise
        """
        # TODO: fill in the function
        if not self.DIMENSIONS.within_bounds(position):
            return False
        
        for obstacle in self.OBSTACLES:
            if obstacle.within_bounds(position):
                return False
        
        return True

    # --- Ground Truth Sensing ---

    def get_robot_pose(self) -> Pose:
        """
        Return the true robot pose.
        """
        # TODO: fill in the function
        return self.robot_pose


    def get_landmark_by_id(self, id: int):
        """
        Retrieve a landmark object using its id number.
        """
        for l in self.LANDMARKS:
            if l.id == id:
                return l
        return None

    def get_landmarks_by_pos(self, pos: Position):
        """
        Retrieve any landmarks at a given position.
        """
        lms = []
        for l in self.LANDMARKS:
            if l.pos == pos:
                lms.append(l)
        return lms

    def get_proximity_to_landmarks(self) -> pd.DataFrame:
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        # TODO: fill in the function
        measurements = pd.DataFrame()
        for i in self.LANDMARKS:
            delta_x = i.pos.x - self.robot_pose.pos.x
            delta_y = i.pos.y - self.robot_pose.pos.y
            range = math.sqrt(delta_x**2 + delta_y**2)
            bearing = math.atan2(delta_y, delta_x) - self.robot_pose.theta
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            measurements[f"Landmark{i.id}"] = [BearingRange(i.id, bearing, range)]
        return measurements

    def get_gt_field_value(self) -> float:
        """
        Returns the ground truth field measurement of the robot at the current ground truth pose.
        """
        # if no field, return 0
        if self.continuous_field is None:
            return 0.0
        # predict the field value at the robot's coordinates using the field model for ground truth 
        return self.continuous_field.field.predict(np.asarray((self.robot_pose.pos.x, self.robot_pose.pos.y)).reshape(1,-1))

    # --- Logging ---

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        # TODO: fill in the function
        df0 = pd.DataFrame({
            "time": [self.time],
            "robot_pose": [self.robot_pose.to_dict()],
        })

        prox_to_landmarks = self.get_proximity_to_landmarks()

        return pd.concat([df0, prox_to_landmarks], axis=1)

    def info(self) -> dict:
        """
        Return a dictionary of frozen environment information.
        """
        result = {
            "Dimensions": self.DIMENSIONS.to_dict(),
            "Obstacles": [obs.to_dict() for obs in self.OBSTACLES],
            "Landmarks": [l.to_dict() for l in self.LANDMARKS],
            "Timestep": self.DT,
            "Pinger Range": self.lm_range,
        }
        if self.continuous_field is not None:
            result["Field"] = self.continuous_field.info()
        return result

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        # TODO: fill in the function
        return {
            "dimensions": self.DIMENSIONS,
            "timestep": self.DT,
            "obstacles": [obstacle.to_dict() for obstacle in self.OBSTACLES],
            "landmarks": [landmark.to_dict() for landmark in self.LANDMARKS],
        }
