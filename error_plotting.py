import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class ErrorPlotting:
    def __init__(self):
        # Data storage for plotting
        self.q_desired_data = []
        self.q_current_data = []
        self.q_error_data = []

        # Data storage for plotting
        self.dq_desired_data = []
        self.dq_current_data = []
        self.dq_error_data = []

        self.ddq_desired_data = []
        self.ddq_current_data = []
        self.ddq_error_data = []

        # IK data storage
        self.q_err_data = []
        self.q3_dot_data = []

        # Data storage for plotting
        self.xb_data = []
        self.x2_data = []
        self.dx_b_data = []
        self.q1_dot_data = []

        self.xw_data = []
        self.x3_data = []
        self.dx_sw_data = []
        self.q2_dot_data = []
        

        self.xb_dot_data = []
        self.x2_dot_data = []
        self.dx_b_dot_data = []

        self.xw_dot_data = []
        self.x3_dot_data = []
        self.dx_sw_dot_data = []



        self.dq_cmd_data = []
        self.output_data = []

        # Inverse dynamics data storage
        self.tau_data = []
        self.Fc_data = []
        self.ddxc_data = []
        self.ddq_diff_data = []

        # check ddot_q data storage
        self.ddq_ik_data = []
        self.ddq_dik_data = []

        self.index_data = []

        self.FR_position = []   
        self.FL_position = []
        self.RR_position = []
        self.RL_position = []

        self.torque_sensor_data = []



    @staticmethod
    def plot_state_trajectories(body_trajectory, swing_leg_trajectory):
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(body_trajectory[:, 0], label="X")
        plt.plot(body_trajectory[:, 1], label="Y")
        plt.plot(body_trajectory[:, 2], label="Z")
        plt.title("Body Trajectory")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(swing_leg_trajectory[:, 0], label="X")
        plt.plot(swing_leg_trajectory[:, 1], label="Y")
        plt.plot(swing_leg_trajectory[:, 2], label="Z")
        plt.title("Front Swing Leg Trajectory")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(swing_leg_trajectory[:, 3], label="X")
        plt.plot(swing_leg_trajectory[:, 4], label="Y")
        plt.plot(swing_leg_trajectory[:, 5], label="Z")
        plt.title("Rear Swing Leg Trajectory")
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_q_error(q_desired, q_actual, q_error, title):
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
        plt.figure(figsize=(12, 18))
        plt.subplot(4, 1, 1)
        for joint in range(3):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel(f'{title}')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        for joint in range(3, 6):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel(f'{title}')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        for joint in range(6, 9):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel(f'{title}')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        for joint in range(9, 12):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{labels[joint]}]', linestyle='--')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel(f'{title}')
        plt.legend()
        plt.grid(True)

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
        plt.grid(True)

        plt.subplot(4, 1, 2)
        for joint in range(3, 6):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        for joint in range(6, 9):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        for joint in range(9, 12):
            plt.plot([qd[joint] for qd in q_dot], label=f'q_dot[{labels[joint]}]', linestyle='-.')
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q_dot')
        plt.legend()
        plt.grid(True)

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
        plt.grid(True)

        plt.subplot(3, 1, 2)
        for joint in range(12):
            plt.plot([data[joint] for data in dq_dot], label=f'dq_dot[{labels[joint]}]')
        plt.title('dq_dot Over Time')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('dq_dot')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        for joint in range(12):
            plt.plot([data[joint] for data in output], label=f'output[{labels[joint]}]')
        plt.title('Output Over Time')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Output')
        plt.grid(True)

    @staticmethod
    def plot_state_error_trajectories(desired_state, current_state, state_error, title):
        if title == "Body":
            labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        else:
            labels = ['x_front', 'y_front', 'z_front', 'x_rear', 'y_rear', 'z_rear']
        plt.figure(figsize=(12, 18))

        for i, label in enumerate(labels):
            plt.subplot(len(labels), 1, i + 1)
            plt.plot([data[i] for data in desired_state], label=f'desired_state[{label}]', linestyle='--')
            plt.plot([data[i] for data in current_state], label=f'current_state[{label}]', linestyle='-')
            plt.title(f'{title} {label.capitalize()} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'{label.capitalize()}')
            plt.legend()
            plt.grid(True)

    def plot_contact_acceleration(self, ddxc, title):
        num_subplots = len(ddxc[0]) 
        plt.figure(figsize=(12, 6 * num_subplots))
        
        for i in range(num_subplots):
            plt.subplot(num_subplots, 1, i + 1)
            plt.plot([data[i] for data in ddxc], label=f'ddxc[{i}]')
            plt.title(f'{title} {i} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'ddxc[{i}]')
            plt.legend()
            plt.grid(True)

    def plot_full_body_state(self, q_full, title):
        labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']

        plt.figure(figsize=(12, 18))
        for i, label in enumerate(labels):
            plt.subplot(len(labels), 1, i + 1)
            plt.plot([data[i] for data in q_full], label=f'q_full[{label}]')
            plt.title(f'{title} {label.capitalize()} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'{label.capitalize()}')
            plt.legend()
            plt.grid(True)

    def plot_contact_force(self, Fc, title):
        num_subplots = len(Fc[0])
        plt.figure(figsize=(12, 6 * num_subplots))
        
        for i in range(num_subplots):
            plt.subplot(num_subplots, 1, i + 1)
            plt.plot([data[i] for data in Fc], label=f'Fc[{i}]')
            plt.title(f'{title} {i} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'Fc[{i}]')
            plt.legend()
            plt.grid(True)

    def plot_torque(self, tau, title):
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']

        plt.figure(figsize=(12, 18))
        for i, label in enumerate(labels):
            plt.subplot(len(labels), 1, i + 1)
            plt.plot([data[i] for data in tau], label=f'tau[{label}]')
            plt.title(f'{title} {label.capitalize()} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'{label.capitalize()}')
            plt.legend()
            plt.grid(True)
    
    def plot_index_data(self, index_data, title):
        plt.figure(figsize=(12, 6))
        plt.plot(index_data)
        plt.title(f'{title} Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Index Value')
        plt.legend(['Index'])
        plt.grid(True)

    def plot_foot_location(self, FL, FR, RL, RR, title):
        labels = ['x', 'y', 'z']
        plt.figure(figsize=(12, 18))
        
        for i, label in enumerate(labels):
            plt.subplot(len(labels), 1, i + 1)
            plt.plot([data[i] for data in FL], label='FL_foot')
            plt.plot([data[i] for data in FR], label='FR_foot')
            plt.plot([data[i] for data in RL], label='RL_foot')
            plt.plot([data[i] for data in RR], label='RR_foot')
            plt.title(f'{title} {label.capitalize()} Over Time')
            plt.xlabel('Iteration')
            plt.ylabel(f'{label.capitalize()}')
            plt.legend()
            plt.grid(True)
