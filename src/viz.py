import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
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

    def __init__(self, output_path: Path):
        """
        Initialize the visualizer class.
        """
        self.output_path = output_path
        
        gt_csv_path = output_path / "ground_truth_history.csv"
        sensor_csv_path = output_path / "sensor_data_history.csv"
        env_config_path = output_path / "environment_config.csv"
        
        self.gt_log = pd.read_csv(gt_csv_path)
        self.sensor_log = pd.read_csv(sensor_csv_path)
        
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

        ax.legend(loc="upper right")
        return fig, ax

    def poses_from_odom(self):
        """
        Given a DataFrame of odometry data with the following columns:
        Time | Odometry_LinearVelocity | Odometry_AngularVelocity
        Output a DataFrame of estimated pose data with the following columns:
        Time | x | y | theta
        """
        first_pose = self.gt_log['robot_pose_parsed'].iloc[0]
        x = first_pose['pos']['x']
        y = first_pose['pos']['y']
        theta = first_pose['theta']
        
        poses = []
        dt = self.env_info["Timestep"]
        NEAR_ZERO = 1e-6

        for idx, row in self.sensor_log.iterrows():
            v = row['encoder_lin_vel']
            w = row['encoder_ang_vel']
            time = row['time']

            if abs(w) < NEAR_ZERO:
                # Straight line motion
                dx = v * dt * np.cos(theta)
                dy = v * dt * np.sin(theta)
                dtheta = 0.0
            else:
                # Arc motion (proper differential drive)
                r = v / w  # turning radius
                dtheta = w * dt
                dx = r * (np.sin(theta + dtheta) - np.sin(theta))
                dy = -r * (np.cos(theta + dtheta) - np.cos(theta))
            
            # Update pose
            x += dx
            y += dy
            theta += dtheta

            # Wrap theta to [-pi, pi]
            theta = theta % (2 * np.pi)
            if theta > np.pi:
                theta -= 2 * np.pi

            poses.append({"Time": time, "x": x, "y": y, "theta": theta})

        return pd.DataFrame(poses)

    def poses_from_gt(self):
        """
        Given a DataFrame of ground truth data with the following columns:
        Time | RobotPose
        Where RobotPose data is in the form: X{xposition}Y{yposition}T{heading}
        And time data is in decaseconds.
        Output a DataFrame of ground truth pose data with the following columns:
        Time | x | y | theta
        Where time data is in seconds.
        """
        poses = []
        
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
        Given a DataFrame of GPS data with the following columns:
        Time | GPS
        Where GPS data is a Position dataclass with fields x and y.
        Output a DataFrame of GPS pose data with the following columns:
        Time | x | y | theta
        Note: theta is set to NaN since GPS doesn't measure heading.
        """
        poses = []
        
        if 'gps_x' in self.sensor_log.columns and 'gps_y' in self.sensor_log.columns:
            for idx, row in self.sensor_log.iterrows():
                if pd.notna(row['gps_x']) and pd.notna(row['gps_y']):
                    poses.append({
                        "Time": row['time'],
                        "x": row['gps_x'],
                        "y": row['gps_y'],
                        "theta": np.nan,
                    })

        return pd.DataFrame(poses)

    def plot_single_trajectory(
        self,
        label,
        pose_table: pd.DataFrame,
        color="blue",
        alpha=1.0,
        scatter=False,
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

        # Plot trajectory path
        ax.plot(
            pose_table["x"],
            pose_table["y"],
            "-",
            color=color,
            linewidth=2,
            label=label,
            alpha=alpha,
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
                color=color,
                linewidth=2,
                alpha=alpha,
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

        ax.legend(loc="upper right")
        return ax

    def draw_all(self, animate=False, fps=30, speedup=3.0, linger_seconds=2.0):
        """
        Docstring for draw_all

        :param self: Description
        """
        self.plot_env()
        self.plot_single_trajectory(
            "Ground Truth",
            self.poses_from_gt(),
            "green",
        )
        self.plot_single_trajectory(
            "Dead Reckoning",
            self.poses_from_odom(),
            "red",
        )
        
        gps_poses = self.poses_from_gps()
        if not gps_poses.empty:
            self.plot_single_trajectory(
                "GPS Only",
                gps_poses,
                "orange",
                scatter=True,
            )
        
        plt.savefig(self.output_path / "dataset_viz.png")
        print(f"Static plot saved: {self.output_path / 'dataset_viz.png'}")
        
        if animate:
            print("Generating animation...")
            self.animate_trajectories(fps=fps, speedup=speedup, linger_seconds=linger_seconds)

    def animate_trajectories(
        self,
        fps=30,
        speedup=3.0,
        linger_seconds=5.0,
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
            [], [], "-", color="green", linewidth=2, label="Ground Truth", alpha=0.8
        )
        (odom_line,) = ax.plot(
            [], [], "-", color="red", linewidth=2, label="Dead Reckoning", alpha=0.8
        )
        (gps_line,) = ax.plot([], [], "-", color="orange", linewidth=2, alpha=0.8)
        gps_scatter = ax.scatter(
            [],
            [],
            c="orange",
            s=50,
            marker="x",
            label="GPS Measurements",
            alpha=0.6,
            zorder=5,
        )

        # Initialize end marker objects (hidden initially)
        gt_end = ax.plot([], [], "s", color="green", markersize=10, alpha=0)[0]
        odom_end = ax.plot([], [], "s", color="red", markersize=10, alpha=0)[0]
        gps_end = ax.plot([], [], "s", color="orange", markersize=10, alpha=0)[0]

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
            gps_line.set_data([], [])
            gps_scatter.set_offsets(np.empty((0, 2)))
            gt_end.set_data([], [])
            odom_end.set_data([], [])
            gps_end.set_data([], [])
            time_text.set_text("")
            return (
                gt_line,
                odom_line,
                gps_line,
                gps_scatter,
                gt_end,
                odom_end,
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
                    gps_line.set_data(gps_up_to_now["x"], gps_up_to_now["y"])
                    gps_scatter.set_offsets(gps_up_to_now[["x", "y"]].values)

            return gt_line, odom_line, gps_scatter, gt_end, odom_end, time_text

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frame_indices),
            interval=1000 / fps,  # milliseconds between frames
            blit=True,
            repeat=True,
        )

        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(self.output_path / "trajectory_animation.gif", writer=writer)
        plt.close(fig)
        print(f"Animation saved: {self.output_path / 'trajectory_animation.gif'}")
        return anim
