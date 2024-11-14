import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class ErrorPlotting:

    @staticmethod
    def plot_state_trajectories(body_trajectory, swing_leg_trajectory):
        """
        Plot the body and swing leg trajectories.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(body_trajectory[:, 0], label="X")
        plt.plot(body_trajectory[:, 1], label="Y")
        plt.plot(body_trajectory[:, 2], label="Z")
        plt.title("Body Trajectory")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(swing_leg_trajectory[:, 0], label="X")
        plt.plot(swing_leg_trajectory[:, 1], label="Y")
        plt.plot(swing_leg_trajectory[:, 2], label="Z")
        plt.title("Front Swing Leg Trajectory")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(swing_leg_trajectory[:, 3], label="X")
        plt.plot(swing_leg_trajectory[:, 4], label="Y")
        plt.plot(swing_leg_trajectory[:, 5], label="Z")
        plt.title("Rear Swing Leg Trajectory")
        plt.legend()

    @staticmethod
    def plot_q_error(q_desired, q_actual):
        """
        Plot the joint angles, desired joint angles, joint position error, and actuated joint angles.

        Parameters:
        q_desired (list): Desired joint angles, shape (N, num_joints).
        q_actual (list): Actual joint angles, shape (N, num_joints).
        """
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']

        plt.figure(figsize=(12, 18))
        plt.subplot(4, 1, 1)
        for joint in range(3):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{labels[joint]}]', linestyle='-')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 2)
        for joint in range(3, 6):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{labels[joint]}]', linestyle='-')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 3)
        for joint in range(6, 9):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{labels[joint]}]', linestyle='-')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 4)
        for joint in range(9, 12):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{labels[joint]}]', linestyle='-')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

    @staticmethod
    def plot_q_dot(q_dot, title):
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']

        plt.figure(figsize=(12, 18))
        plt.subplot(4, 1, 1)
        for joint in range(3):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()

        plt.subplot(4, 1, 2)
        for joint in range(3, 6):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()

        plt.subplot(4, 1, 3)
        for joint in range(6, 9):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()

        plt.subplot(4, 1, 4)
        for joint in range(9, 12):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()

    @staticmethod
    def plot_api_value(dq_error, dq_dot, output):
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']

        plt.figure(figsize=(12, 18))
        plt.subplot(3, 1, 1)
        for joint in range(12):
            plt.plot([data[joint] for data in dq_error], label=f'dq_error[{labels[joint]}]')
        plt.legend()
        plt.title('dq_error Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('dq_error')

        plt.subplot(3, 1, 2)
        for joint in range(12):
            plt.plot([data[joint] for data in dq_dot], label=f'dq_dot[{labels[joint]}]')
        plt.title('dq_dot Over Time')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('dq_dot')

        plt.subplot(3, 1, 3)
        for joint in range(12):
            plt.plot([data[joint] for data in output], label=f'output[{labels[joint]}]')
        plt.title('Output Over Time')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Output')

    @staticmethod
    def plot_state_error_trajectories(desired_state, current_state, state_error, title):
        """
        Plot the desired state, current state, and state error trajectories.

        Parameters:
        desired_state (list): Desired state values over time.
        current_state (list): Current state values over time.
        state_error (list): State error values over time.
        title (str): Title for the plot.
        """
        if title == "Body":
            labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        else:
            labels = ['x_front', 'y_front', 'z_front', 'x_rear', 'y_rear', 'z_rear']
        plt.figure(figsize=(12, 18))

        for i, label in enumerate(labels):
            plt.subplot(6, 1, i + 1)
            plt.plot([data[i] for data in desired_state], label=f'desired_state[{label}]', linestyle='--')
            plt.plot([data[i] for data in current_state], label=f'current_state[{label}]', linestyle='-')
            # plt.plot([data[i] for data in state_error], label=f'state_error[{label}]', linestyle='-.')
            plt.title(f'{title} {label.capitalize()} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'{label.capitalize()}')
            plt.legend()
            # plt.axis('equal',)