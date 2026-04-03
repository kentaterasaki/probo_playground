import numpy as np
import random


class KalmanFilter:
    """
    A class that implements the basic Kalman Filter algorithm, which assumes linear system dynamics and Gaussian noise.

    Attributes:
        dt: the length of each timestep, in seconds
        x: the state vector for the system we are estimating
        P: the process model, describing the uncertainty in our estimate
        F: the state transition matrix, describing how our state naturally changes from timestep to timestep
        B: the control input model, describing how control inputs affect each state variable in the state vector
        Q: the process noise, modeling unexpected disturbance in state transitions
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an instance of the KalmanFilter class.

        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable
        """
        # TODO: set the timestep size to the given parameter
        self.DT: float = dt

        # TODO: set the state vector to the given prior
        self.x: np.ndarray = prior

        # TODO: set the process model to an identity matrix
        self.P: np.ndarray = np.identity(3)

        # TODO: define the motion model
        self.F: np.ndarray = np.identity(3)

        # TODO: define the control model
        self.B: np.ndarray = np.identity(3)*dt

        # TODO: define the process noise
        self.Q_stdev = 0.1
        self.Q: np.ndarray = self.get_Q()


    def predict(self, u: np.ndarray):
        """
        Predicts the next state vector and its covariance matrix using the state transition matrix and an input control vector. The Kalman Filter uses the following predict equations:

        x_t+1 = F * x_t + B * u_t
        P_t+1 = F * P * F.T + Q

        Args:
            u: the input control vector
        """
        # TODO: update the state vector using the state transition matrix and the given control input
        self.x = self.F @ self.x + self.B @ u

        # TODO: update the process model by propagating it through the state transition matrix and adding noise
        self.Q = self.get_Q()
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x, self.P

    def update(self, z, H, R):
        """
        Updates the current state prediction using observations from the environment. The Kalman Filter uses the following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        y = z - H * x (residual: error between observation and expected observation given estimated state vector)
        K = P * H.T * inv(S) (Kalman Gain: portion of total uncertainty that is from the prediction)
        S = H * P * H.T + R (total uncertainty in the system)

        Args:
            z: the given observation, AKA a measurement taken of the environment
            H: the measurement model, which relates the state space to the measurement space
            R: the measurement noise model (covariance)
        """
        # TODO: calculate the total uncertainty in the system
        S = H @ self.P @ H.T + R

        # TODO: calculate the Kalman Gain, AKA the percentage of the total uncertainty that came from the estimate rather than the measurement
        K = self.P @ H.T @ np.linalg.inv(S)

        # TODO: calculate the residual, AKA the error between the observation and what we expected the observation to be given our estimated state vector
        y = z - H @ self.x

        # TODO: update the state vector
        self.x = self.x + K @ y

        # TODO: update the process model
        self.P = (np.identity(3) - K @ H) @ self.P

        return self.x, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        stdev = self.Q_stdev
        return np.identity(3) * stdev**2