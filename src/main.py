"""
Main file for running the simulator.
"""

from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from sensors import WheelEncoder, LandmarkPinger
from viz import save_environment_config
import pandas as pd

if __name__ == "__main__":
    # set up the environment
    # TODO: choose values for each input parameter, using the expected datatype
    dimensions = Bounds(x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0)
    dt = 0.1
    obstacles = [Bounds(x_min=5.0, x_max=7.0, y_min=6.0, y_max=8.0), Bounds(x_min=3.0, x_max=5.0, y_min=4.0, y_max=6.0)]
    landmarks = [Landmark(pos=Position(x=2.0, y=2.0), id=1), Landmark(pos=Position(x=8.0, y=8.0), id=2)]
    initial_robot_pose = Pose(pos=Position(x=0.0, y=0.0), theta=0.0)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env)
    
    # add sensors to the robot
    wheel_encoder = WheelEncoder(robot, name="wheel_encoder")
    landmark_pinger = LandmarkPinger(robot, name="landmark_pinger")
    robot.sensors.append(wheel_encoder)
    robot.sensors.append(landmark_pinger)

    # set up timekeeping
    # TODO: set the total_seconds variable to however long you want the simulator to run (not real-time!)
    total_seconds = 15.0
    total_timesteps = total_seconds / env.DT

    # set up logging
    ground_truth_history = []
    sensor_data_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = "input/vel_cmd_example.csv"
    output_ground_truth_filepath = "output/ground_truth_history.csv"
    output_sensor_data_filepath = "output/sensor_data_history.csv"

    # Load all velocity commands from the CSV file
    commands_df = pd.read_csv(input_commands_filepath)
    
    # iterate through each timestep
    for step in range(int(total_timesteps) + 1):
        # TODO: take a ground truth snapshot and add it to the history
        ground_truth_history.append(env.take_state_snapshot())
        # TODO: take sensor measurements and add it to the history
        sensor_data_history.append(robot.take_sensor_measurements())
        
        # Determine which command is active at the current time
        current_time = step * env.DT
        # Find the most recent command that has a timestamp <= current_time
        active_commands = commands_df[commands_df['timestamp'] <= current_time]
        # Get the most recent command
        active_command = active_commands.iloc[-1]
        lin_vel = active_command['linear_vel']
        ang_vel = active_command['angular_vel']
        
        # TODO: execute the motor command
        dx, dy, dtheta = robot.robot_step_differential(float(lin_vel), float(ang_vel))
        env.robot_step(dx, dy, dtheta)


    # at the end, write the histories into output files
    with open(output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file
        pd.concat(ground_truth_history, ignore_index=True).to_csv(gt_data, index=False)
    with open(output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file
        pd.concat(sensor_data_history, ignore_index=True).to_csv(sensor_data, index=False)
    
    save_environment_config("output", dimensions, obstacles, landmarks, dt, landmark_pinger.MAX_RANGE)
