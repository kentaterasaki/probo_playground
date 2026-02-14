"""
Main file for running the simulator.
"""

from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from sensors import WheelEncoder, LandmarkPinger, GPS
from kalman_filter import KalmanFilter as KF
from viz import save_environment_config
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # set up the environment
    # TODO: choose values for each input parameter, using the expected datatype
    dimensions = Bounds(x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0)
    dt = 0.1
    obstacles = [Bounds(x_min=5.0, x_max=7.0, y_min=8.0, y_max=9.0), Bounds(x_min=3.0, x_max=5.0, y_min=5.0, y_max=6.0)]
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

    # Initialize Kalman Filter with Initial Robot Pose
    prior = np.array([initial_robot_pose.pos.x, initial_robot_pose.pos.y, initial_robot_pose.theta])
    kf = KF(dt, prior)


    # add sensors to the robot
    wheel_encoder = WheelEncoder(robot, name="wheel_encoder")
    landmark_pinger = LandmarkPinger(robot, name="landmark_pinger")
    gps = GPS(robot, name="gps", interval=1.0, x_noise=0.5, y_noise=0.5)  # actual sensor noise (simulation truth)
    robot.sensors.append(wheel_encoder)
    robot.sensors.append(landmark_pinger)
    robot.sensors.append(gps)

    # Assumed or tuned measurement noise for GPS
    R_assumed = np.array([
        [0.55, 0.0],  # assumed variance in x. (R = guess_x_noise^2) in meters. Ex: R = .01 means sigma = 0.1 m and very accurate gps
        [0.0,  0.55],  # assumed variance in y
    ])

    # set up timekeeping
    # TODO: set the total_seconds variable to however long you want the simulator to run (not real-time!)
    total_seconds = 15.0
    total_timesteps = total_seconds / env.DT

    # set up logging
    ground_truth_history = []
    sensor_data_history = []
    kalman_filter_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = "input/vel_cmd_example_translational.csv"
    output_ground_truth_filepath = "output/ground_truth_history.csv"
    output_sensor_data_filepath = "output/sensor_data_history.csv"
    output_kalman_filter_filepath = "output/kalman_filter_history.csv"

    # Load all velocity commands from the CSV file
    commands_df = pd.read_csv(input_commands_filepath)

    # Set drive mode before simulation so encoder reports consistent columns
    if input_commands_filepath == "input/vel_cmd_example_translational.csv":
        robot.drive_mode = "translational"
    
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

        # Kalman Filter Prediction Step
        u = np.array([active_command['x_vel'], active_command['y_vel'], active_command['angular_vel']])
        kf_prediction = kf.predict(u)

        # print(f"prediction: {kf_prediction}")
        # print(f"sensor data: {sensor_data_history[-1]['H']}")
        # print(f"sensor data: {sensor_data_history[-1]['R']}")

        #Kalman Filter Update Step
        if sensor_data_history:
            latest = sensor_data_history[-1]
            z = np.array([latest['gps_x'].iloc[0], latest['gps_y'].iloc[0]])
            H = latest['H'].iloc[0]
            kf_update = kf.update(z, H, R_assumed)

        # Log KF state after predict+update for this timestep
        kalman_filter_history.append({
            "time": current_time,
            "x": kf.x[0],
            "y": kf.x[1],
            "theta": kf.x[2],
            "P_xx": kf.P[0, 0],
            "P_yy": kf.P[1, 1],
            "P_tt": kf.P[2, 2],
            "P_xy": kf.P[0, 1],
        })
        

        # TODO: Retrieve input from csv and execute the motor command
        if input_commands_filepath == "input/vel_cmd_example_translational.csv":
            x_vel = active_command['x_vel']
            y_vel = active_command['y_vel']
            ang_vel = active_command['angular_vel']
            dx, dy, dtheta = robot.robot_step_translational(float(x_vel), float(y_vel), float(ang_vel))
        else:
            lin_vel = active_command['linear_vel']
            ang_vel = active_command['angular_vel']
            dx, dy, dtheta = robot.robot_step_differential(float(lin_vel), float(ang_vel))

        env.robot_step(dx, dy, dtheta)

    # at the end, write the histories into output files
    with open(output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file
        pd.concat(ground_truth_history, ignore_index=True).to_csv(gt_data, index=False)
    with open(output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file
        pd.concat(sensor_data_history, ignore_index=True).to_csv(sensor_data, index=False)
    with open(output_kalman_filter_filepath, "w") as kf_data:
        pd.DataFrame(kalman_filter_history).to_csv(kf_data, index=False)
    
    save_environment_config("output", dimensions, obstacles, landmarks, dt, landmark_pinger.MAX_RANGE)

from pathlib import Path
from viz import Visualizer

# Create visualizer
viz = Visualizer(output_path=Path("output"))

# Option 1: Generate only static plot (fast)
# viz.draw_all(animate=False)

# Option 2: Generate static plot + animation (slower)
viz.draw_all(animate=True, fps=30, speedup=1.0, linger_seconds=2.0)

print("\n✓ Done! Check the output/ folder for visualizations!")
