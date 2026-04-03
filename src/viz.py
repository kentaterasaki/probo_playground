import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from utils import Pose, Landmark
from itertools import product
import copy
import ast


def save_environment_config(output_path, dimensions, obstacles, landmarks, timestep, pinger_range):
    """
    Save environment configuration to CSV for visualization. Had Claude Code help generate this function as I was struggling with getting all the objects on the visualizations.
    
    Args:
        output_path: Path to output directory
        dimensions: Bounds object with x_min, x_max, y_min, y_max
        obstacles: List of Bounds objects
        landmarks: List of Landmark objects with pos.x, pos.y, and id
        timestep: Simulation timestep (dt)
        pinger_range: Maximum range for landmark pinger sensor
    """
    output_path = Path(output_path)
    config_file = output_path / "environment_config.csv"
    
    config_data = []
    
    # Save dimensions
    config_data.append({
        'type': 'dimension',
        'x_min': dimensions.x_min,
        'x_max': dimensions.x_max,
        'y_min': dimensions.y_min,
        'y_max': dimensions.y_max,
        'id': 0
    })
    
    # Save obstacles
    for i, obs in enumerate(obstacles):
        config_data.append({
            'type': 'obstacle',
            'x_min': obs.x_min,
            'x_max': obs.x_max,
            'y_min': obs.y_min,
            'y_max': obs.y_max,
            'id': i
        })
    
    # Save landmarks
    for lm in landmarks:
        config_data.append({
            'type': 'landmark',
            'x_min': lm.pos.x,
            'x_max': lm.pos.y,
            'y_min': 0,
            'y_max': 0,
            'id': lm.id
        })
    
    # Save timestep
    config_data.append({
        'type': 'timestep',
        'x_min': timestep,
        'x_max': 0,
        'y_min': 0,
        'y_max': 0,
        'id': 0
    })
    
    # Save pinger range
    config_data.append({
        'type': 'pinger_range',
        'x_min': pinger_range,
        'x_max': 0,
        'y_min': 0,
        'y_max': 0,
        'id': 0
    })
    
    pd.DataFrame(config_data).to_csv(config_file, index=False)


class Visualizer:
    """
    Visualizer for Kalman Filter trajectory estimation results.
    """

    def __init__(self, output_path: Path, filter_type: str = "kalman", show_landmarks: bool = True, 
                 params: dict = None):
        """
        Initialize the visualizer class.
        Supports two data formats:
          - CSV files (from main.py / KF workflow)
          - Pickle files (from simulator.py / informative sampling workflow)
        """
        self.output_path = output_path
        self.filter_type = filter_type
        self.show_landmarks = show_landmarks
        self.params = params or {}
        self.kf_log = None
        self.env_info = None
        self.sensor_info = None

        gt_pkl_path = output_path / "groundtruth_log.pkl"
        if gt_pkl_path.exists():
            self._load_from_pickle()
        else:
            self._load_from_csv()

    def _load_from_pickle(self):
        """Load data from pickle files (simulator.py workflow)."""
        import pickle
        with open(self.output_path / "groundtruth_log.pkl", "rb") as f:
            self.gt_log = pickle.load(f)
        with open(self.output_path / "sensor_log.pkl", "rb") as f:
            self.sensor_log = pickle.load(f)
        with open(self.output_path / "env_info.pkl", "rb") as f:
            self.env_info = pickle.load(f)
        with open(self.output_path / "sensor_info.pkl", "rb") as f:
            self.sensor_info = pickle.load(f)

    def _load_from_csv(self):
        """Load data from CSV files (main.py / KF workflow)."""
        gt_csv_path = self.output_path / "ground_truth_history.csv"
        sensor_csv_path = self.output_path / "sensor_data_history.csv"
        env_config_path = self.output_path / "environment_config.csv"

        if self.filter_type == "kalman":
            kf_csv_path = self.output_path / "kalman_filter_history.csv"
            self.filter_label = "Linear Kalman Filter"
        elif self.filter_type == "extended_kalman":
            kf_csv_path = self.output_path / "extended_kalman_filter_history.csv"
            self.filter_label = "Extended Kalman Filter"
        else:
            kf_csv_path = None

        self.gt_log = pd.read_csv(gt_csv_path)
        self.sensor_log = pd.read_csv(sensor_csv_path)

        if kf_csv_path and kf_csv_path.exists():
            self.kf_log = pd.read_csv(kf_csv_path)

        self._parse_csv_data()
        self._load_env_config_from_csv(env_config_path)
 
    def _parse_csv_data(self):
        """Parse string representations in CSV data."""
        def parse_pose_string(pose_str):
            """Parse robot pose string into a dict."""
            try:
                return ast.literal_eval(pose_str)
            except:
                return None
        
        self.gt_log['robot_pose_parsed'] = self.gt_log['robot_pose'].apply(parse_pose_string)
    
    def _load_env_config_from_csv(self, config_path: Path):
        """Load environment configuration from CSV file."""
        config_df = pd.read_csv(config_path)
        
        dim_row = config_df[config_df['type'] == 'dimension'].iloc[0]
        dimensions = {
            'x_min': dim_row['x_min'],
            'x_max': dim_row['x_max'],
            'y_min': dim_row['y_min'],
            'y_max': dim_row['y_max']
        }
        
        obstacles = []
        obs_rows = config_df[config_df['type'] == 'obstacle']
        for _, obs in obs_rows.iterrows():
            obstacles.append({
                'x_min': obs['x_min'],
                'x_max': obs['x_max'],
                'y_min': obs['y_min'],
                'y_max': obs['y_max']
            })
        
        landmarks = []
        lm_rows = config_df[config_df['type'] == 'landmark']
        for _, lm in lm_rows.iterrows():
            landmarks.append({
                'id': int(lm['id']),
                'pos': {
                    'x': lm['x_min'],
                    'y': lm['x_max']
                }
            })
        
        ts_row = config_df[config_df['type'] == 'timestep']
        timestep = ts_row.iloc[0]['x_min']
        
        pr_row = config_df[config_df['type'] == 'pinger_range']
        pinger_range = pr_row.iloc[0]['x_min']
        
        self.env_info = {
            'Dimensions': dimensions,
            'Obstacles': obstacles,
            'Landmarks': landmarks,
            'Pinger Range': pinger_range,
            'Timestep': timestep
        }
    

    def plot_env(self):
        """
        Plot the environment features with no trajectories.
        """
        # set up axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Environment Map")

        # Set up the plot boundaries
        dims = self.env_info["Dimensions"]
        ax.set_xlim(dims["x_min"] - 1, dims["x_max"] + 1)
        ax.set_ylim(dims["y_min"] - 1, dims["y_max"] + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # # Plot ground truth field and field measuremnts
        # belief_info = self.sensor_info.loc[self.sensor_info["Sensor Name"] == "belief"]
        # obs_model = belief_info["Model"][0]
        # measurements = obs_model.y_train_
        # measurement_positions = obs_model.X_train_

        # x = np.linspace(dims["x_min"], dims["x_max"], 10)
        # y = np.linspace(dims["y_min"], dims["y_max"], 10)
        # X, Y = np.meshgrid(x, y)
        # M = np.array(list(product(x,y)))
        
        # model = self.env_info["Field"]["Model"]
        # c_sample = model.sample_y(M, 1, random_state=self.env_info["Field"]["Random Seed"])
        # ax.contourf(X, Y, c_sample.reshape(10,10).T, 10,
        #             vmin=np.nanmin(measurements), vmax=np.nanmax(measurements))

        # ax.scatter(measurement_positions[:,0], measurement_positions[:,1], c=measurements, cmap="viridis",
        #            s=200, lw=1.5, edgecolors='k', vmin=np.nanmin(measurements), vmax=np.nanmax(measurements))


        # set up env boundaries
        width = dims["x_max"] - dims["x_min"]
        height = dims["y_max"] - dims["y_min"]
        walls = patches.Rectangle(
            (dims["x_min"], dims["y_min"]),
            width,
            height,
            linewidth=5,
            edgecolor="black",
            facecolor="none",
            alpha=1.0,
        )
        ax.add_patch(walls)

        # Plot obstacles
        for obs in self.env_info["Obstacles"]:
            width = obs["x_max"] - obs["x_min"]
            height = obs["y_max"] - obs["y_min"]
            rect = patches.Rectangle(
                (obs["x_min"], obs["y_min"]),
                width,
                height,
                linewidth=2,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
                label="Obstacle" if obs == self.env_info["Obstacles"][0] else "",
            )
            ax.add_patch(rect)

        # Plot landmarks
        if self.show_landmarks:
            for lm in self.env_info["Landmarks"]:
                # Plot pinging range circle
                circle = patches.Circle(
                    (lm["pos"]["x"], lm["pos"]["y"]),
                    self.env_info["Pinger Range"],
                    linewidth=1,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.1,
                    label="Pinging Range" if lm == self.env_info["Landmarks"][0] else "",
                )
                ax.add_patch(circle)

                # plot floating point landmarks
                ax.plot(
                    lm["pos"]["x"],
                    lm["pos"]["y"],
                    "r*",
                    markersize=15,
                    label="Landmark" if lm == self.env_info["Landmarks"][0] else "",
                )
                ax.annotate(
                    f"LM{lm['id']}",
                    (lm["pos"]["x"], lm["pos"]["y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10,
                    color="red",
                )

        # Add parameter info to legend if provided
        if self.params:
            param_lines = []
            if 'Q' in self.params:
                Q_val = self.params['Q']
                if isinstance(Q_val, np.ndarray):
                    if Q_val.ndim == 2:
                        Q_display = np.mean(np.diag(Q_val))
                    else:
                        Q_display = Q_val.flat[0]
                else:
                    Q_display = Q_val
                param_lines.append(f"Q (process): {Q_display:.4f}")
            if 'R' in self.params:
                R_val = self.params['R']
                param_lines.append(f"R (meas): [{R_val[0,0]:.2f}, {R_val[1,1]:.2f}]")
            if 'gps_noise' in self.params:
                param_lines.append(f"GPS σ: {self.params['gps_noise']:.2f}m")
            if 'pinger_range_noise' in self.params:
                param_lines.append(f"Pinger range σ: {self.params['pinger_range_noise']:.2f}m")
            if 'pinger_bearing_noise' in self.params:
                bearing_deg = np.degrees(self.params['pinger_bearing_noise'])
                param_lines.append(f"Pinger bearing σ: {bearing_deg:.1f}°")
            
        return fig, ax

    def plot_belief_env(self):
        """
        Plot the environment features with no trajectories from belief.
        """
        # set up axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Belief Map")

        # Set up the plot boundaries
        dims = self.env_info["Dimensions"]
        ax.set_xlim(dims["x_min"] - 1, dims["x_max"] + 1)
        ax.set_ylim(dims["y_min"] - 1, dims["y_max"] + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Plot belief truth field
        belief_info = self.sensor_info.loc[self.sensor_info["Sensor Name"] == "belief"]
        obs_model = belief_info["Model"][0]
        measurements = obs_model.y_train_
        measurement_positions = obs_model.X_train_

        x = np.linspace(dims["x_min"], dims["x_max"], 10)
        y = np.linspace(dims["y_min"], dims["y_max"], 10)
        X, Y = np.meshgrid(x, y)
        M = np.array(list(product(x,y)))
        
        c_sample, std_dev = obs_model.predict(M, return_std=True)
        ax.contourf(X, Y, c_sample.reshape(10,10).T, 10,
                    vmin=np.nanmin(measurements), vmax=np.nanmax(measurements))
        ax.scatter(measurement_positions[:,0], measurement_positions[:,1], c=measurements, cmap="viridis",
                   s=200, lw=1.5, edgecolors='k', vmin=np.nanmin(measurements), vmax=np.nanmax(measurements))
        
        ax_inset = ax.inset_axes([0.8, 0.05, 0.3, 0.3])
        ax_inset.contourf(X, Y, std_dev.reshape(10,10).T, 10)
        ax_inset.scatter(measurement_positions[:,0], measurement_positions[:,1], s=1, lw=1.5, edgecolors='k')
        ax_inset.set_title("Belief Uncertainty")

        # set up env boundaries
        width = dims["x_max"] - dims["x_min"]
        height = dims["y_max"] - dims["y_min"]
        walls = patches.Rectangle(
            (dims["x_min"], dims["y_min"]),
            width,
            height,
            linewidth=5,
            edgecolor="black",
            facecolor="none",
            alpha=1.0,
        )
        ax.add_patch(walls)

        # Plot obstacles
        for obs in self.env_info["Obstacles"]:
            width = obs["x_max"] - obs["x_min"]
            height = obs["y_max"] - obs["y_min"]
            rect = patches.Rectangle(
                (obs["x_min"], obs["y_min"]),
                width,
                height,
                linewidth=2,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
                label="Obstacle" if obs == self.env_info["Obstacles"][0] else "",
            )
            ax.add_patch(rect)

        # Plot landmarks
        if self.show_landmarks:
            for lm in self.env_info["Landmarks"]:
                # Plot pinging range circle
                circle = patches.Circle(
                    (lm["pos"]["x"], lm["pos"]["y"]),
                    self.env_info["Pinger Range"],
                    linewidth=1,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.1,
                    label="Pinging Range" if lm == self.env_info["Landmarks"][0] else "",
                )
                ax.add_patch(circle)

                # plot floating point landmarks
                ax.plot(
                    lm["pos"]["x"],
                    lm["pos"]["y"],
                    "r*",
                    markersize=15,
                    label="Landmark" if lm == self.env_info["Landmarks"][0] else "",
                )
                ax.annotate(
                    f"LM{lm['id']}",
                    (lm["pos"]["x"], lm["pos"]["y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10,
                    color="red",
                )

        # Add parameter info to legend if provided generated by Claude Opus 4.6
        if self.params:
            param_lines = []
            if 'Q' in self.params:
                Q_val = self.params['Q']
                if isinstance(Q_val, np.ndarray):
                    if Q_val.ndim == 2:
                        Q_display = np.mean(np.diag(Q_val))
                    else:
                        Q_display = Q_val.flat[0]
                else:
                    Q_display = Q_val
                param_lines.append(f"Q (process): {Q_display:.4f}")
            if 'R' in self.params:
                R_val = self.params['R']
                param_lines.append(f"R (meas): [{R_val[0,0]:.2f}, {R_val[1,1]:.2f}]")
            if 'gps_noise' in self.params:
                param_lines.append(f"GPS σ: {self.params['gps_noise']:.2f}m")
            if 'pinger_range_noise' in self.params:
                param_lines.append(f"Pinger range σ: {self.params['pinger_range_noise']:.2f}m")
            if 'pinger_bearing_noise' in self.params:
                bearing_deg = np.degrees(self.params['pinger_bearing_noise'])
                param_lines.append(f"Pinger bearing σ: {bearing_deg:.1f}°")
            
        return fig, ax

    def poses_from_odom(self):
        """
        Reconstruct trajectory from odometry sensor data.
        Supports both pickle (Pose objects) and CSV (parsed strings) formats.
        """
        if self.sensor_info is not None:
            first_pose = self.gt_log['robot_pose'].iloc[0]
            if isinstance(first_pose, dict):
                x = first_pose['pos']['x']
                y = first_pose['pos']['y']
                theta = first_pose['theta']
            else:
                x = first_pose.pos.x
                y = first_pose.pos.y
                theta = first_pose.theta
            dt = self.env_info["Timestep"]
            time_col = "time"
        else:
            first_pose = self.gt_log['robot_pose_parsed'].iloc[0]
            x = first_pose['pos']['x']
            y = first_pose['pos']['y']
            theta = first_pose['theta']
            dt = self.env_info["Timestep"] if self.env_info else 0.1
            time_col = "time"

        poses = []
        NEAR_ZERO = 1e-6

        is_translational = 'encoder_x_vel' in self.sensor_log.columns
        has_odom = 'Odometry_LinearVelocity' in self.sensor_log.columns

        for idx, row in self.sensor_log.iterrows():
            if is_translational:
                w = row['encoder_ang_vel']
                vx = row['encoder_x_vel']
                vy = row['encoder_y_vel']
                dx = vx * dt
                dy = vy * dt
                dtheta = w * dt
            elif has_odom:
                v = row.get('Odometry_LinearVelocity', 0)
                w = row.get('Odometry_AngularVelocity', 0)
                if pd.isna(v) or pd.isna(w):
                    poses.append({"Time": row[time_col], "x": x, "y": y, "theta": theta})
                    continue
                if abs(w) < NEAR_ZERO:
                    dx = v * dt * np.cos(theta)
                    dy = v * dt * np.sin(theta)
                    dtheta = 0.0
                else:
                    r = v / w
                    dtheta = w * dt
                    dx = r * (np.sin(theta + dtheta) - np.sin(theta))
                    dy = -r * (np.cos(theta + dtheta) - np.cos(theta))
            elif 'encoder_lin_vel' in self.sensor_log.columns:
                v = row['encoder_lin_vel']
                w = row['encoder_ang_vel']
                if abs(w) < NEAR_ZERO:
                    dx = v * dt * np.cos(theta)
                    dy = v * dt * np.sin(theta)
                    dtheta = 0.0
                else:
                    r = v / w
                    dtheta = w * dt
                    dx = r * (np.sin(theta + dtheta) - np.sin(theta))
                    dy = -r * (np.cos(theta + dtheta) - np.cos(theta))
            else:
                dx, dy, dtheta = 0, 0, 0

            x += dx
            y += dy
            theta += dtheta
            theta = theta % (2 * np.pi)
            if theta > np.pi:
                theta -= 2 * np.pi

            poses.append({"Time": row[time_col], "x": x, "y": y, "theta": theta})

        return pd.DataFrame(poses)

    def poses_from_gt(self):
        """
        Extract ground truth poses from the GT log.
        Supports both pickle and CSV formats.
        """
        poses = []

        if self.sensor_info is not None:
            for idx, row in self.gt_log.iterrows():
                pose = row['robot_pose']
                if isinstance(pose, dict):
                    poses.append({"Time": row['time'], "x": pose['pos']['x'], "y": pose['pos']['y'], "theta": pose['theta']})
                else:
                    poses.append({"Time": row['time'], "x": pose.pos.x, "y": pose.pos.y, "theta": pose.theta})
        else:
            for idx, row in self.gt_log.iterrows():
                pose_dict = row['robot_pose_parsed']
                if pose_dict:
                    poses.append({
                        "Time": row['time'],
                        "x": pose_dict['pos']['x'],
                        "y": pose_dict['pos']['y'],
                        "theta": pose_dict['theta']
                    })

        return pd.DataFrame(poses)

    def poses_from_gps(self):
        """
        Extract GPS position measurements from sensor log.
        Supports both pickle (Position objects) and CSV (float columns) formats.
        """
        poses = []

        if self.sensor_info is not None and 'GPS' in self.sensor_log.columns:
            for row in self.sensor_log.itertuples():
                if hasattr(row, "GPS") and row.GPS is not None:
                    try:
                        poses.append({"Time": row.Time, "x": row.GPS.x, "y": row.GPS.y, "theta": np.nan})
                    except (AttributeError, TypeError):
                        continue
        elif 'gps_x' in self.sensor_log.columns and 'gps_y' in self.sensor_log.columns:
            for idx, row in self.sensor_log.iterrows():
                if pd.notna(row['gps_x']) and pd.notna(row['gps_y']):
                    poses.append({"Time": row['time'], "x": row['gps_x'], "y": row['gps_y'], "theta": np.nan})

        return pd.DataFrame(poses)

    def poses_from_kf(self):
        """
        Extract Kalman Filter estimated poses from the KF log.
        Output a DataFrame with columns: Time | x | y | theta
        Also stores covariance data for uncertainty ellipses.
        Claude Opus 4.6 helped me write this function.
        """
        if self.kf_log is None or self.kf_log.empty:
            return pd.DataFrame()
        
        poses = pd.DataFrame({
            "Time": self.kf_log["time"],
            "x": self.kf_log["x"],
            "y": self.kf_log["y"],
            "theta": self.kf_log["theta"],
        })
        return poses

    def plot_uncertainty_ellipses(self, ax, color="blue", alpha=0.15, skip=None):
        """
        Plot covariance ellipses from the Kalman Filter at regular intervals.
        Each ellipse represents the 95% confidence region (2-sigma) of the KF estimate.
        Claude Opus 4.6 helped me write this function.
        """
        if self.kf_log is None or self.kf_log.empty:
            return
        
        from matplotlib.patches import Ellipse
        
        if skip is None:
            skip = max(1, len(self.kf_log) // 15)
        
        for idx in range(0, len(self.kf_log), skip):
            row = self.kf_log.iloc[idx]
            
            # Extract 2x2 position covariance submatrix
            P_xx = row["P_xx"]
            P_yy = row["P_yy"]
            P_xy = row["P_xy"]
            cov = np.array([[P_xx, P_xy], [P_xy, P_yy]])
            
            # Eigen decomposition to get ellipse axes and rotation
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Clamp negative eigenvalues (numerical issues) to small positive
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # 95% confidence: chi-squared with 2 DOF -> scale factor ~2.4477
            scale = 2.0 * np.sqrt(eigenvalues) * 2.4477
            
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            ellipse = Ellipse(
                xy=(row["x"], row["y"]),
                width=scale[0],
                height=scale[1],
                angle=angle,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=1,
            )
            ax.add_patch(ellipse)

    def plot_single_trajectory(
        self,
        label,
        pose_table: pd.DataFrame,
        color="blue",
        alpha=1.0,
        scatter=False,
        zorder=None,
    ):
        """
        Given a DataFrame of robot pose data with the following columns:
        Time | x | y | theta
        Plot the trajectory as a quiver on an xy grid with theta as arrow angle.
        """
        # Get current axes or create new ones
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig, ax = self.plot_env()

        # Plot trajectory path (skip if scatter-only like GPS)
        if not scatter:
            ax.plot(
                pose_table["x"],
                pose_table["y"],
                "-",
                color=color,
                linewidth=2,
                label=label,
                alpha=alpha,
                zorder=zorder,
            )

        # Plot arrows showing heading at intervals
        # Show arrows every N points to avoid clutter
        skip = max(1, len(pose_table) // 20)

        for idx in range(0, len(pose_table), skip):
            row = pose_table.iloc[idx]
            dx = 0.5 * np.cos(row["theta"])
            dy = 0.5 * np.sin(row["theta"])

            ax.arrow(
                row["x"],
                row["y"],
                dx,
                dy,
                head_width=0.3,
                head_length=0.2,
                fc=color,
                ec=color,
                alpha=alpha * 0.25,
            )
        if scatter:
            ax.scatter(
                pose_table["x"],
                pose_table["y"],
                marker="*",
                s=30,
                color=color,
                linewidth=1.5,
                alpha=alpha,
                label=label,
                zorder=zorder,
            )

        # Mark start and end positions
        start = pose_table.iloc[0]
        end = pose_table.iloc[-1]

        ax.plot(
            start["x"],
            start["y"],
            "o",
            color=color,
            markersize=10,
            alpha=alpha,
        )
        ax.plot(
            end["x"],
            end["y"],
            "s",
            color=color,
            markersize=10,
            alpha=alpha,
        )

        return ax

    def draw_all(self, animate=False, fps=30, speedup=3.0, linger_seconds=2.0, 
                 output_png="dataset_viz.png", output_gif="trajectory_animation.gif"):
        """
        Docstring for draw_all

        :param self: Description
        """
        self.plot_env()
        self.plot_single_trajectory(
            "Ground Truth",
            self.poses_from_gt(),
            "green",
            zorder=5,
        )
        self.plot_single_trajectory(
            "Dead Reckoning",
            self.poses_from_odom(),
            "red",
            zorder=5,
        )
        
        gps_poses = self.poses_from_gps()
        if not gps_poses.empty:
            self.plot_single_trajectory(
                "GPS Measurements",
                gps_poses,
                "orange",
                scatter=True,
                zorder=0,
            )
        
        kf_poses = self.poses_from_kf()
        if not kf_poses.empty:
            self.plot_single_trajectory(
                getattr(self, 'filter_label', 'Filter Estimate'),
                kf_poses,
                "blue",
                zorder=5,
            )
            # Draw uncertainty ellipses on the current axes
            self.plot_uncertainty_ellipses(plt.gca(), color="blue", alpha=0.1)
        
        ax = plt.gca()
        if self.params:
            param_lines = []
            if 'Q_stdev' in self.params:
                param_lines.append(f"Q (σ): {self.params['Q_stdev']:.4f}")
            if 'R_gps' in self.params:
                R_val = self.params['R_gps']
                param_lines.append(f"R X: {R_val[0,0]:.2f}")
                param_lines.append(f"R Y: {R_val[1,1]:.2f}")
            if 'gps_x_noise' in self.params and 'gps_y_noise' in self.params:
                param_lines.append(f"GPS X_NOISE: {self.params['gps_x_noise']:.2f}m")
                param_lines.append(f"GPS Y_NOISE: {self.params['gps_y_noise']:.2f}m")
            if 'pinger_range_noise' in self.params:
                param_lines.append(f"RANGE_NOISE: {self.params['pinger_range_noise']:.2f}m")
            if 'pinger_bearing_noise' in self.params:
                bearing_deg = np.degrees(self.params['pinger_bearing_noise'])
                param_lines.append(f"BEARING_NOISE: {bearing_deg:.1f}°")
            
            if param_lines:
                param_text = '\n'.join(param_lines)
                legend = ax.legend(loc="upper left", title=param_text, 
                                 frameon=True, fancybox=False, shadow=False,
                                 framealpha=1.0, edgecolor='black')
                legend.get_title().set_fontsize(9)
                legend.get_title().set_ha('left')
        else:
            ax.legend(loc="upper left")
        
        plt.savefig(self.output_path / output_png)
        print(f"Static plot saved: {self.output_path / output_png}")
        
        if animate:
            print("Generating animation...")
            self.animate_trajectories(fps=fps, speedup=speedup, linger_seconds=linger_seconds, 
                                    output_gif=output_gif)

        if hasattr(self, 'sensor_info') and self.sensor_info is not None:
            self.plot_belief_env()
            self.plot_single_trajectory(
                "Ground Truth",
                self.poses_from_gt(),
                "green",
            )
            plt.savefig(self.output_path / "dataset_belief_viz.png")
            print(f"Belief plot saved: {self.output_path / 'dataset_belief_viz.png'}")

    def animate_trajectories(
        self,
        fps=30,
        speedup=3.0,
        linger_seconds=5.0,
        output_gif="trajectory_animation.gif",
    ):
        """
        Create an animated GIF showing trajectories being drawn over time.

        Args:
            fps: Frames per second for the animation
            save_path: Where to save the GIF
            speedup: Speed multiplier (2.0 = 2x faster, 0.5 = half speed)
            linger_seconds: How long to hold on final frame with end markers
        """
        # Get all trajectory data
        gt_poses = self.poses_from_gt()
        odom_poses = self.poses_from_odom()
        gps_poses = self.poses_from_gps()
        kf_poses = self.poses_from_kf()

        # Find the maximum number of frames needed
        max_frames = max(len(gt_poses), len(odom_poses))

        # Apply speedup by sampling fewer frames
        frame_skip = int(speedup)
        frame_indices = list(range(0, max_frames, max(1, frame_skip)))

        # Add linger frames at the end (repeat last frame)
        linger_frames = int(linger_seconds * fps)
        frame_indices.extend([frame_indices[-1]] * linger_frames)

        # Initialize the plot
        fig, ax = self.plot_env()

        # Initialize line objects for each trajectory
        (gt_line,) = ax.plot(
            [], [], "-", color="green", linewidth=2, label="Ground Truth", alpha=0.8, zorder=5
        )
        (odom_line,) = ax.plot(
            [], [], "-", color="red", linewidth=2, label="Dead Reckoning", alpha=0.8, zorder=5
        )
        gps_scatter = ax.scatter(
            [],
            [],
            c="orange",
            s=30,
            marker="x",
            label="GPS Measurements",
            alpha=0.8,
            zorder=0,
        )
        (kf_line,) = ax.plot(
            [], [], "-", color="blue", linewidth=2, label=self.filter_label, alpha=0.8, zorder=5
        )

        # Initialize end marker objects (hidden initially)
        gt_end = ax.plot([], [], "s", color="green", markersize=10, alpha=0)[0]
        odom_end = ax.plot([], [], "s", color="red", markersize=10, alpha=0)[0]
        gps_end = ax.plot([], [], "s", color="orange", markersize=10, alpha=0)[0]
        kf_end = ax.plot([], [], "s", color="blue", markersize=10, alpha=0)[0]

        # List to track uncertainty ellipse patches for animation
        kf_ellipses = []

        # Add time display
        time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.legend(loc="upper right")

        def init():
            """Initialize animation"""
            gt_line.set_data([], [])
            odom_line.set_data([], [])
            gps_scatter.set_offsets(np.empty((0, 2)))
            kf_line.set_data([], [])
            gt_end.set_data([], [])
            odom_end.set_data([], [])
            gps_end.set_data([], [])
            kf_end.set_data([], [])
            time_text.set_text("")
            return (
                gt_line,
                odom_line,
                gps_scatter,
                kf_line,
                gt_end,
                odom_end,
                kf_end,
                time_text,
            )

        def animate(frame_idx):
            """Update function for each frame"""
            actual_frame = (
                frame_indices[frame_idx]
                if frame_idx < len(frame_indices)
                else frame_indices[-1]
            )
            is_final_frame = frame_idx >= len(frame_indices) - linger_frames

            # Update ground truth
            if actual_frame < len(gt_poses):
                gt_data = gt_poses.iloc[: actual_frame + 1]
                gt_line.set_data(gt_data["x"], gt_data["y"])
                current_time = gt_data.iloc[-1]["Time"]
                time_text.set_text(f"Time: {current_time:.1f}s")

                # Show end marker on final frames
                if is_final_frame:
                    gt_end.set_data([gt_data.iloc[-1]["x"]], [gt_data.iloc[-1]["y"]])
                    gt_end.set_alpha(0.8)

            # Update dead reckoning
            if actual_frame < len(odom_poses):
                odom_data = odom_poses.iloc[: actual_frame + 1]
                odom_line.set_data(odom_data["x"], odom_data["y"])

                # Show end marker on final frames
                if is_final_frame:
                    odom_end.set_data(
                        [odom_data.iloc[-1]["x"]], [odom_data.iloc[-1]["y"]]
                    )
                    odom_end.set_alpha(0.8)

            # Update GPS measurements synchronized by time
            # Find GPS measurements up to the current time
            if actual_frame < len(gt_poses) and not gps_poses.empty:
                current_time = gt_poses.iloc[actual_frame]["Time"]
                gps_up_to_now = gps_poses[gps_poses["Time"] <= current_time]
                if len(gps_up_to_now) > 0:
                    gps_scatter.set_offsets(gps_up_to_now[["x", "y"]].values)

            # Update Kalman Filter trajectory. Claude Opus 4.6 helped me write this if statement.
            if not kf_poses.empty and actual_frame < len(kf_poses):
                kf_data = kf_poses.iloc[: actual_frame + 1]
                kf_line.set_data(kf_data["x"], kf_data["y"])

                # Draw uncertainty ellipse at current position (remove previous)
                from matplotlib.patches import Ellipse
                for e in kf_ellipses:
                    e.remove()
                kf_ellipses.clear()

                if self.kf_log is not None and actual_frame < len(self.kf_log):
                    row = self.kf_log.iloc[actual_frame]
                    P_xx, P_yy, P_xy = row["P_xx"], row["P_yy"], row["P_xy"]
                    cov = np.array([[P_xx, P_xy], [P_xy, P_yy]])
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    eigenvalues = np.maximum(eigenvalues, 1e-10)
                    scale = 2.0 * np.sqrt(eigenvalues) * 2.4477
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    ellipse = Ellipse(
                        xy=(row["x"], row["y"]),
                        width=scale[0], height=scale[1], angle=angle,
                        facecolor="blue", edgecolor="blue", alpha=0.15, linewidth=1,
                    )
                    ax.add_patch(ellipse)
                    kf_ellipses.append(ellipse)

                # Show end marker on final frames
                if is_final_frame:
                    kf_end.set_data([kf_data.iloc[-1]["x"]], [kf_data.iloc[-1]["y"]])
                    kf_end.set_alpha(0.8)

            return gt_line, odom_line, gps_scatter, kf_line, gt_end, odom_end, kf_end, time_text

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frame_indices),
            interval=1000 / fps,  # milliseconds between frames
            blit=False,
            repeat=True,
        )

        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(self.output_path / output_gif, writer=writer)
        plt.close(fig)
        print(f"Animation saved: {self.output_path / output_gif}")
        return anim
