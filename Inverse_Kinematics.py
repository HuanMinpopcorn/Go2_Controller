import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg as sp
from Kinematics import Kinematics
from Send_motor_cmd import send_motor_commands


from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import time
import matplotlib.pyplot as plt
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from matplotlib.animation import FuncAnimation
import config

# import unitree_mujoco.simulate_python.unitree_mujoco as sim 


class InverseKinematic(Kinematics):
    """
    Inverse Kinematic Controller for Unitree Robots using MuJoCo simulation.
    This class extends Kinematics and controls robot joints based on trajectory planning.
    """

    def __init__(self, xml_path, cmd, step_size=config.SIMULATE_DT):
        super().__init__(xml_path)
        self.step_size = step_size
        self.swing_time =  10  # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing

        # Robot parameters
        self.body_height = 0.25
        self.swing_height = 0.075
        self.velocity = 0.0

        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]

    def check_joint_limits(self, joint_angles):
        """
        Ensure the joint angles are within the limits defined in the MuJoCo model.
        """
        for i, angle in enumerate(joint_angles):
            lower_limit, upper_limit = self.model.jnt_range[i, :]
            joint_angles[i] = np.clip(angle, lower_limit, upper_limit)
        return joint_angles

    def get_required_state(self):
        """
        Get the required positions of contact and swing legs, and the body state.
        """
        contact_positions = [self.get_body_state(leg)["position"] for leg in self.contact_legs]
        swing_positions = [self.get_body_state(leg)["position"] for leg in self.swing_legs]

        body_state = self.get_body_state(self.body_frame_name)
        body_position = body_state["position"]
        body_orientation = body_state["orientation"]
        body_state_combined = np.hstack((body_position, body_orientation))
        # x1 : contact leg positions, x2 : body state, x3 : swing leg positions

        return np.hstack(contact_positions), body_state_combined, np.hstack(swing_positions)

    def get_required_jacobian(self):
        """
        Compute Jacobians for the contact legs, body, and swing legs.
        """
        J1 = np.vstack([self.get_jacobian(leg)["J_pos"] for leg in self.contact_legs])
        J2 = np.zeros((6, self.model.nv))
        body_jacobian = self.get_jacobian(self.body_frame_name)
        J2[:3, :] = body_jacobian["J_pos"]
        J2[3:, :] = body_jacobian["J_rot"]
        J3 = np.vstack([self.get_jacobian(leg)["J_pos"] for leg in self.swing_legs])

        return J1, J2, J3

    def compute_desired_value(self):
        """
        Generate trajectories for the body and swing legs over the swing phase.
        """
        print("Computing desired trajectories...")
        body_trajectory, swing_leg_trajectory = [], []

        # Get initial body and leg positions
        body_state = self.get_body_state(self.body_frame_name)
        body_position = body_state["position"]
        body_orientation = body_state["orientation"]
        body_configuration = np.hstack((body_position, body_orientation))
        # Initialize swing leg positions
        swing_leg_positions = [
            np.copy(self.get_body_state(leg_name)["position"])
            for leg_name in self.swing_legs
        ]
        swing_leg_positions = np.hstack(swing_leg_positions)
        # print("body_configuration", body_configuration)
        # print("swing_leg_positions", swing_leg_positions)
        swing_leg_positions_initial = np.copy(swing_leg_positions)
        body_configuration_initial = np.copy(body_configuration)
        for i in range(self.K):
            body_configuration[0] = body_configuration_initial[0] + self.velocity * i * self.step_size
            body_configuration[2] = body_configuration_initial[2] + 0.05 * np.sin(np.pi * i / (2 * self.K))
            body_trajectory.append(body_configuration.copy())

            # Generate swing leg trajectories

            swing_leg_positions[0] = swing_leg_positions_initial[0] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[1] = swing_leg_positions_initial[1]  # Keep Y position constant
            swing_leg_positions[2] = swing_leg_positions_initial[2] + self.swing_height * np.sin(np.pi * i / self.K)
            swing_leg_positions[3] = swing_leg_positions_initial[3] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[4] = swing_leg_positions_initial[4]  # Keep Y position constant
            swing_leg_positions[5] = swing_leg_positions_initial[5] + self.swing_height * np.sin(np.pi * i / self.K)

            swing_leg_trajectory.append(swing_leg_positions.copy())

        # Plot the trajectories
        body_trajectory = np.array(body_trajectory)
        swing_leg_trajectory = np.array(swing_leg_trajectory)
      
        return np.array(body_trajectory), np.array(swing_leg_trajectory)

    def cubic_spline(self, t, tf, xf):
        """
        Generate cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3

    import matplotlib.pyplot as plt

    def calculate(self):
        """
        Main loop to compute and update joint angles in real-time,
        including a trot gait with proper leg phasing.
        """
   

        # intial the API gain 
        # kp = 120
        # kd = 5
        kp = 45
        # kp = 50
        kd = 4

        # kd = 4
        kc = 1
        # Gain for joint position error
        # kb : 2.5 - 3
        kb = 4.955
        # kb = 5.8103
        # Gain for joint velocity error
        # ks = 15
        ks = 0

        m = self.model.nv
        i = 0
        trail = 0
        running_time = 0
        leg_pair_in_swing = True

        # Data storage for plotting
        q_desired_data = []
        q_current_data = []
        q_err_data = []
        q3_dot_data = []


        # Data storage for plotting
        xb_data = []
        x2_data = []
        dx_b_data = []
        q1_dot_data = []

        xw_data = []
        x3_data = []
        dx_sw_data = []
        q2_dot_data = []

        dq_cmd_data = []
        dq_error_data = []
        dq_dot_data = []
        ouput_data = []


        print("Starting Trot Gait...")
        print("initial body state", self.get_body_state(self.body_frame_name))
        intial_joint_angles = self.joint_state_reader.joint_angles.copy()
        # initial desired trajectory 
        x_b, x_sw = self.compute_desired_value()
        while True:
            i = (i + 1) % self.K  # Loop over the swing cycle duration
            phase = np.tanh(running_time / 1.2)
            joint_angles = self.joint_state_reader.joint_angles
            x1, x2, x3 = self.get_required_state() 
            # x1 : contact leg positions, x2 : body state, x3 : swing leg positions
            J1, J2, J3 = self.get_required_jacobian()
            # J1 : contact leg Jacobian, J2 : body Jacobian, J3 : swing leg Jacobian

            config =  (intial_joint_angles - joint_angles) # shape (12, 1)
            q_err = kc * np.hstack((np.zeros(6), config)).reshape(-1, 1) # shape (18, 1)
            # print("q_err", q_err.T)
        
            # print(x_b[i].T - x2)
            
            dx_b = kb * (x_b[i].T - x2).reshape(-1, 1) # body state : xyz + rpy 
            dx_sw = ks * (x_sw[i].T - x3).reshape(-1, 1)
            # print("dx_sw", dx_sw.T)

            xb_data.append(x_b[i].T)
            x2_data.append(x2.flatten())
            dx_b_data.append(dx_b.flatten())

            xw_data.append(x_sw[i].T)
            x3_data.append(x3.flatten())
            dx_sw_data.append(dx_sw.flatten())

        
            
        
            N1 = np.eye(m) - np.linalg.pinv(J1, rcond=1e-4) @ J1      
            J_21 = J2 @ N1
            N_21 = np.eye(m) - np.linalg.pinv(J_21, rcond=1e-4) @ J_21      
     
            q1_dot = np.linalg.pinv(J_21, rcond=1e-4) @ dx_b
            # q1_dot = np.zeros(18).reshape(-1, 1)
            q2_dot = np.linalg.pinv(J3 @ N_21, rcond=1e-4) @ (dx_sw - J3 @ q1_dot)
            N_321 = np.eye(m) - np.linalg.pinv(J3 @ N_21, rcond=1e-4) @ J3 @ N_21
            q3_dot = N_321 @ q_err
            q_dot = q1_dot + q2_dot + q3_dot
            # q_dot = q1_dot + q3_dot
            dq_cmd = q_dot[6:].flatten()
            new_joint_angles = joint_angles + dq_cmd  


            
            # Append more data for plotting 
            # check for q3 dot
            # q_desired_data.append(self.change_q_order(stand_up_joint_pos))
            # q_current_data.append(joint_angles.copy())
            # q_err_data.append(q_err[6:].copy())
            q3_dot_data.append(self.change_q_order(q3_dot[6:].flatten()))
            q2_dot_data.append(self.change_q_order(q2_dot[6:].flatten()))
            q1_dot_data.append(self.change_q_order(q1_dot[6:].flatten()))

            # new_joint_angles = self.check_joint_limits(new_joint_angles)
            # gravity_torque = np.array(self.data.qfrc_bias[6:]).flatten()

            # cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles) * phase + (1-phase) * self.change_q_order(intial_joint_angles), self.change_q_order(dq_cmd))
            cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles), self.change_q_order(dq_cmd))
            dq_error = kp * (self.change_q_order(new_joint_angles) - self.data.sensordata[:12])
            # dq_error = kp * (self.change_q_order(new_joint_angles) * phase + (1-phase) * self.change_q_order(intial_joint_angles) - self.data.sensordata[12:24])
            dq_error_data.append(dq_error)
            dq_dot= kd * (self.change_q_order(dq_cmd) - self.data.sensordata[12:24])
            dq_dot_data.append(dq_dot)
            ouput_data.append(dq_error + dq_dot)

            trail += 1
            # print("Trail: ", trail)
            if trail > 1000:  # Replace or remove as needed
                break

        self.plot_api_value(dq_error_data, dq_dot_data, ouput_data)
        self.plot_q_dot(q3_dot_data, "q3_dot")
        self.plot_q_dot(q2_dot_data, "q2_dot")
        self.plot_q_dot(q1_dot_data , "q1_dot")
        self.plot_state_error_trajectories(xb_data, x2_data, dx_b_data, "Body")
        self.plot_state_error_trajectories(xw_data, x3_data, dx_sw_data, "Swing")
        # self.plot_q_error(q_desired_data, q_current_data) # all shape is (N, 12)
        plt.show()


    def transition_legs(self):
        """
        Swap the swing and contact legs for the next cycle.
        """
        self.swing_legs, self.contact_legs = self.contact_legs, self.swing_legs
        # print("Legs transitioned: Swing legs ->", self.swing_legs, ", Contact legs ->", self.contact_legs)

    def change_q_order(self, q):
        """
        Change the order of the joint angles.
        """
        return np.array(
            [
                q[3], q[4], q[5], q[0], q[1], q[2], q[9], q[10], q[11], q[6], q[7], q[8]
            ]
        )
    def plot_state_trajectories(self, body_trajectory, swing_leg_trajectory):
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

        plt.tight_layout()


    def plot_q_error(self, q_desired, q_actual):
        """
        Plot the joint angles, desired joint angles, joint position error, and actuated joint angles.

        Parameters:
        q_desired (list): Desired joint angles, shape (N, num_joints).
        q_actual (list): Actual joint angles, shape (N, num_joints).
        q_error (list): Joint position errors, shape (N, num_joints).
        q_actuated (list): Actuated joint angles, shape (N, num_joints).
        """
        # Plot q2_dot data for the last twelve joints
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
        
        plt.figure()
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

    def plot_q_dot(self, q_dot, title):
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
        
        plt.figure()
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
    def plot_api_value(self, dq_error, dq_dot, output):
     
        labels = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
        
        plt.figure()
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
    def plot_state_error_trajectories(self, desired_state, current_state, state_error, title):
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

        plt.tight_layout()



# Example Usage
if __name__ == "__main__":


    robot_scene = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    cmd = send_motor_commands()
    ik = InverseKinematic(robot_scene, cmd)
    ik.start_joint_updates()
    cmd.move_to_initial_position()
    # ik.compute_desired_value()
    ik.calculate()