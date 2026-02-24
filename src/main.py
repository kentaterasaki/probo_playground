"""
Main file for running the simulator.
"""

from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from sensors import WheelEncoder, LandmarkPinger, GPS
from kalman_filter import KalmanFilter as KF
from extended_kalman_filter import ExtendedKalmanFilter as EKF
from viz import save_environment_config
import pandas as pd
import numpy as np
from pathlib import Path


def run_simulation(filter_type, input_filepath, drive_mode, ekf_use_gps=True, ekf_use_landmarks=True):
    """
    Run a simulation for a specific filter type.
    
    Args:
        filter_type: "lkf" or "ekf"
        input_filepath: Path to velocity commands CSV
        drive_mode: "translational" or "differential"
        ekf_use_gps: Whether EKF should use GPS measurements (only applies to EKF)
        ekf_use_landmarks: Whether EKF should use LandmarkPinger measurements (only applies to EKF)
        
    Returns:
        dict: Parameters used in simulation (Q, R, sensor noise, etc.)
    """
    # set up the environment
    dimensions = Bounds(x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0)
    dt = 0.1
    obstacles = []
    # obstacles = [Bounds(x_min=5.0, x_max=7.0, y_min=8.0, y_max=9.0), Bounds(x_min=3.0, x_max=5.0, y_min=5.0, y_max=6.0)]
    landmarks = [
        Landmark(pos=Position(x=5.0, y=5.0), id=1),
        Landmark(pos=Position(x=5.0, y=2.5), id=2),
        Landmark(pos=Position(x=2.5, y=0.0), id=3),
        Landmark(pos=Position(x=0.0, y=-5.0), id=4),
        Landmark(pos=Position(x=6.0, y=-3.5), id=5),

    ]
    
    initial_robot_pose = Pose(pos=Position(x=-8.0, y=-8.0), theta=0.0)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env)
    robot.drive_mode = drive_mode

    # Initialize Kalman Filter with Initial Robot Pose
    prior = np.array([initial_robot_pose.pos.x, initial_robot_pose.pos.y, initial_robot_pose.theta])
    if filter_type == "lkf":
        kf = KF(dt, prior)
    else:  # ekf
        kf = EKF(dt, prior)

    # add sensors to the robot
    wheel_encoder = WheelEncoder(robot, name="wheel_encoder")
    landmark_pinger = LandmarkPinger(robot, name="landmark_pinger")
    robot.sensors.append(wheel_encoder)
    robot.sensors.append(landmark_pinger)
    
    # Only add GPS sensor if it will be used
    if filter_type == "lkf" or (filter_type == "ekf" and ekf_use_gps):
        gps = GPS(robot, name="gps", interval=1.0, x_noise=0.1, y_noise=0.1)  # actual gps sensor noise (simulation truth)
        robot.sensors.append(gps)

    # Assumed or tuned measurement noise for GPS
    R = np.array([
        [0.01, 0.0],  # assumed variance in x. (R = guess_x_noise^2) in meters. Ex: R = .01 means sigma = 0.1 m and very accurate gps. Lower noise = more accurate gps
        [0.0,  0.01],  # assumed variance in y
    ])

    # set up timekeeping
    # TODO: set the total_seconds variable to however long you want the simulator to run (not real-time!)
    total_seconds = 50.0
    total_timesteps = total_seconds / env.DT

    # set up logging
    ground_truth_history = []
    sensor_data_history = []
    filter_history = []

    # Load velocity commands
    commands_df = pd.read_csv(input_filepath)
    
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
        if filter_type == "lkf":
            # LKF uses translational commands: [x_vel, y_vel, angular_vel]
            u = np.array([active_command['x_vel'], active_command['y_vel'], active_command['angular_vel']])
        else:  # ekf
            # EKF uses differential commands: [linear_vel, angular_vel]
            u = np.array([active_command['linear_vel'], active_command['angular_vel']])
        
        kf_prediction = kf.predict(u)

        # Kalman Filter Update Step
        if sensor_data_history:
            latest = sensor_data_history[-1]
            
            if filter_type == "lkf":
                # LKF uses GPS measurements
                z = np.array([latest['gps_x'].iloc[0], latest['gps_y'].iloc[0]])
                H = latest['H'].iloc[0]
                kf_update = kf.update(z, H, R)
            else:  # ekf
                # EKF can use both GPS and LandmarkPinger
                # First, update with GPS if enabled and available
                if ekf_use_gps and pd.notna(latest['gps_x'].iloc[0]):
                    z_gps = np.array([latest['gps_x'].iloc[0], latest['gps_y'].iloc[0]])
                    H_gps = latest['H'].iloc[0]
                    kf_update = kf.update(H_gps, R, z_gps, None)
                
                # Then, update with each landmark observation if enabled
                if ekf_use_landmarks:
                    for col in latest.columns:
                        if col.startswith('landmark_pinger_'):
                            br = latest[col].iloc[0]
                            # Check if landmark is in range (not infinite)
                            if br.range != float('inf') and br.bearing != float('inf'):
                                lm_id = br.landmark_id
                                z_lm = np.array([br.range, br.bearing])
                                H_lm = landmark_pinger.H_eval(kf.x, lm_id)
                                R_lm = landmark_pinger.R(z_lm)
                                y_lm = landmark_pinger.y(z_lm, kf.x, lm_id)
                                kf_update = kf.update(H_lm, R_lm, None, y_lm)

        # Log filter state after predict+update for this timestep
        filter_history.append({
            "time": current_time,
            "x": kf.x[0],
            "y": kf.x[1],
            "theta": kf.x[2],
            "P_xx": kf.P[0, 0],
            "P_yy": kf.P[1, 1],
            "P_tt": kf.P[2, 2],
            "P_xy": kf.P[0, 1],
        })

        # Execute the motor command
        if drive_mode == "translational":
            x_vel = active_command['x_vel']
            y_vel = active_command['y_vel']
            ang_vel = active_command['angular_vel']
            dx, dy, dtheta = robot.robot_step_translational(float(x_vel), float(y_vel), float(ang_vel))
        else:  # differential
            lin_vel = active_command['linear_vel']
            ang_vel = active_command['angular_vel']
            dx, dy, dtheta = robot.robot_step_differential(float(lin_vel), float(ang_vel))
        
        env.robot_step(dx, dy, dtheta)

    # Write the histories into output files
    output_dir = Path(f"output/{filter_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ground_truth_history.csv", "w") as gt_data:
        pd.concat(ground_truth_history, ignore_index=True).to_csv(gt_data, index=False)
    # TODO: write sensor_data_history to a file
    with open(output_dir / "sensor_data_history.csv", "w") as sensor_data:
        pd.concat(sensor_data_history, ignore_index=True).to_csv(sensor_data, index=False)
    
    filter_output_name = "kalman_filter_history.csv" if filter_type == "lkf" else "extended_kalman_filter_history.csv"
    with open(output_dir / filter_output_name, "w") as kf_data:
        pd.DataFrame(filter_history).to_csv(kf_data, index=False)
    
    save_environment_config(str(output_dir), dimensions, obstacles, landmarks, dt, landmark_pinger.MAX_RANGE)
    
    # Return parameters for visualization
    params = {
        'Q_stdev': kf.Q_stdev if hasattr(kf, 'Q_stdev') else np.sqrt(np.mean(np.diag(kf.Q))),
    }
    
    # Add GPS parameters if used
    if filter_type == "lkf" or (filter_type == "ekf" and ekf_use_gps):
        params['R_gps'] = R
        params['gps_x_noise'] = gps.X_NOISE
        params['gps_y_noise'] = gps.Y_NOISE
    
    # Add pinger parameters if using landmarks
    if filter_type == "ekf" and ekf_use_landmarks:
        params['pinger_range_noise'] = landmark_pinger.RANGE_NOISE
        params['pinger_bearing_noise'] = landmark_pinger.BEARING_NOISE
    
    return params


if __name__ == "__main__":
    # VISUALIZATION
    GENERATE_LKF_VIZ = True   # Set to True to generate LKF visualizations
    GENERATE_EKF_VIZ = False   # Set to True to generate EKF visualizations
    GENERATE_ANIMATION = False # Set to True to generate animated GIFs
    
    #EKF SENSOR SELECTION
    EKF_USE_GPS = False              # Set to True to use GPS measurements with EKF
    EKF_USE_LANDMARK_PINGER = True  # Set to True to use LandmarkPinger measurements with EKF
    
    from viz import Visualizer
    
    if GENERATE_LKF_VIZ:
        lkf_params = run_simulation("lkf", "input/vel_cmd_example_translational.csv", "translational")
        print("\nGenerating Linear Kalman Filter visualizations...")
        viz_lkf = Visualizer(output_path=Path("output/lkf"), filter_type="kalman", 
                           show_landmarks=False, params=lkf_params)
        viz_lkf.draw_all(animate=GENERATE_ANIMATION, fps=30, speedup=1.0, linger_seconds=2.0, 
                         output_png="lkf_dataset_viz.png", output_gif="lkf_trajectory_animation.gif")
        print("LKF visualizations complete!")

    if GENERATE_EKF_VIZ:
        print("\nRunning EKF Simulation")
        sensor_info = []
        if EKF_USE_GPS:
            sensor_info.append("GPS")
        if EKF_USE_LANDMARK_PINGER:
            sensor_info.append("LandmarkPinger")
        print(f"EKF Sensors: {' + '.join(sensor_info)}")
        
        ekf_params = run_simulation("ekf", "input/vel_cmd_example.csv", "differential", 
                                   ekf_use_gps=EKF_USE_GPS, ekf_use_landmarks=EKF_USE_LANDMARK_PINGER)
        print("\nGenerating Extended Kalman Filter visualizations...")
        viz_ekf = Visualizer(output_path=Path("output/ekf"), filter_type="extended_kalman", 
                           show_landmarks=EKF_USE_LANDMARK_PINGER, params=ekf_params)
        viz_ekf.draw_all(animate=GENERATE_ANIMATION, fps=30, speedup=1.0, linger_seconds=2.0,
                         output_png="ekf_dataset_viz.png", output_gif="ekf_trajectory_animation.gif")
        print("EKF visualizations complete!")

    print("\nDone! Check the output/lkf/ and output/ekf/ folders for visualizations!")
