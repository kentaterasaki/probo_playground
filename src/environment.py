"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

from utils import Position, Pose, Bounds, Landmark, BearingRange
import pandas as pd
import math


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
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
        """
        # TODO: set the dimensions property to the parameter value
        self.DIMENSIONS = dimensions

        # TODO: set the timestep size property to the parameter value
        self.DT = dt

        # TODO: set the current time to zero
        self.time = 0

        # TODO: set the obstacles and landmarks properties to the parameter lists
        self.OBSTACLES = obstacles
        self.LANDMARKS = landmarks

        # TODO: set the robot pose property to the parameter value
        self.robot_pose = robot_starting_pose

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

        new_theta = self.robot_pose.theta + dtheta

        self.time = round(self.time + self.DT, 3)
        self.robot_pose = Pose(new_pos, new_theta)


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
        y_new = self.robot_pose.pos.y
        if self.is_valid_position(Position(x_new + dx, y_new + dy)):
            x_new = self.robot_pose.pos.x + dx
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

    def get_robot_pose(self) -> Pose:
        """
        Return the true robot pose.
        """
        # TODO: fill in the function
        return self.robot_pose


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
