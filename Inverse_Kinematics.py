import mujoco
import numpy as np
import scipy.linalg as sp
from Kinematics import Kinematics
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import time
import matplotlib.pyplot as plt

class InverseKinematic(Kinematics):
    def __init__(self, xml_path, step_size=0.001, tol=1e-3):
        super().__init__(xml_path)  # Initialize the parent class
        self.step_size = step_size
        self.tol = tol

        self.body_height = 0.25
        self.swing_height = 0.075
        self.swing_time = 0.25
        self.step_length = 0.1
        self.velocity = 0.1

        self.body_frame_name = "Base_Link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
        
        self.K = int(self.swing_time / self.step_size)

        self.joint_angles = self.get_current_joint_angles()
        # self.x1, self.x2, self.x3 = np.zeros((6, 1))  # Initialize state variables
        # self.J1, self.J2, self.J3 = np.zeros((6, ))  # Initialize Jacobians

    def check_joint_limits(self, joint_angles):
        for i in range(len(joint_angles)):
            lower_limit, upper_limit = self.model.jnt_range[i, :]
            joint_angles[i] = np.clip(joint_angles[i], lower_limit, upper_limit)
        return joint_angles

    def get_required_state(self):
        x1 = []  # Contact leg positions
        x3 = []  # Swing leg positions
        x2 = np.zeros(6)  # Body state (position + orientation)

        for contact_leg in self.contact_legs:
            contact_position = self.get_body_state(contact_leg)["position"]
            x1.append(contact_position)

        x1 = np.hstack(x1) # Reshape to (3 * num_contact_legs, 1)

        base_state = self.get_body_state(self.body_frame_name)
        base_position = base_state["position"]
        base_orientation = base_state["orientation"]
        x2[:3] = base_position
        x2[3:] = base_orientation

        for swing_leg in self.swing_legs:
            swing_position = self.get_body_state(swing_leg)["position"]
            x3.append(swing_position)

        x3 = np.hstack(x3)  # Reshape to (3 * num_swing_legs, 1)

        return x1, x2, x3

    def get_required_jacobian(self):
        J1 = []
        J2 = np.zeros((6, self.model.nv))
        J3 = []
        J4 = []

        for contact_leg in self.contact_legs:
            jacobian_contact = self.get_jacobian(contact_leg)
            J1.append(jacobian_contact["J_pos"])
        
        J1 = np.vstack(J1)

        jacobian_body = self.get_jacobian(self.body_frame_name)
        J2[:3, :] = jacobian_body["J_pos"]
        J2[3:, :] = jacobian_body["J_rot"]


        for swing_leg in self.swing_legs:
            swing_jacobian = self.get_jacobian(swing_leg)
            J3.append(swing_jacobian["J_pos"])
            
        J3 = np.vstack(J3)
       
        return J1, J2, J3

    def compute_desired_value(self):
        body_trajectory = []
        swing_trajectory_1 = []
        swing_trajectory_2 = []

        for i in range(self.K):
            body_state = self.get_body_state(self.body_frame_name)
            body_position = body_state["position"]
            body_orientation = body_state["orientation"]
            body_state_combined = np.hstack((body_position, body_orientation))
            body_state_combined[0] += self.velocity * i * self.step_size
            body_trajectory.append(body_state_combined.copy())

            swing_leg_position_1 = self.get_body_state(self.swing_legs[0])["position"]
            swing_leg_position_1[0] += self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_position_1[2] += self.swing_height * np.sin(np.pi * i / self.K)
            swing_trajectory_1.append(swing_leg_position_1.copy())

            swing_leg_position_2 = self.get_body_state(self.swing_legs[1])["position"]
            swing_leg_position_2[0] += self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_position_2[2] += self.swing_height * np.sin(np.pi * i / self.K)
            swing_trajectory_2.append(swing_leg_position_2.copy())

            swing_Traj = np.hstack((np.array(swing_trajectory_1), np.array(swing_trajectory_2)))

        return np.array(body_trajectory), swing_Traj

    def calculate(self, initial_joint_angles, x_b, x_sw):
        joint_angles = initial_joint_angles.reshape(-1, 1)  # Ensure shape: (nv, 1)
        q_dot = np.zeros_like(joint_angles)
        joint_trajectory = [joint_angles.copy()]
        q_dot_trajectory = []
        x1, x2, x3 = self.get_required_state()
        J1, J2, J3 = self.get_required_jacobian()
        print("Contact leg positions (x1):", x1)
        print("Body state (x2):", x2)
        print("Swing leg positions (x3):", x3)
        for i in range(self.K):
    
            dx_b = (x_b[i].T - x2).reshape(-1, 1)  # Shape: (6, 1)
            dx_sw = (x_sw[i].T - x3).reshape(-1, 1)  # Shape: (6, 1)

            N1 = sp.null_space(J1)  
            J_21 = J2 @ N1  
            N_21 = sp.null_space(J_21.T @ J_21)  

            print(f"Step {i}:")
            print("x_b:", x_b[i].shape)
            print("x_sw:", x_sw[i].shape)
            print("x1:", x1.shape)
            print("x2:", x2.shape)
            print("x3:", x3.shape)
            print("J1:", J1.shape)
            print("J2:", J2.shape)
            print("J3:", J3.shape)
            print("dx_sw shape:", dx_sw.shape)
            print("dx_b shape:", dx_b.shape)
            print("N1 shape:", N1.shape)
            print("J_21 shape:", J_21.shape)
            print("N_21 shape:", N_21.shape)
            print("J3 @ N1 @ N_21 shape:", np.linalg.pinv(J3 @ N1 @ N_21).shape)
            
            # Calculate joint velocities

            # q_dot = np.linalg.pinv(J_21) @ dx_b + np.linalg.pinv(J3[:,6:] @ N_21) @ (dx_sw - J3 @ np.linalg.pinv(J_21) @ dx_b)
            q1_dot = np.linalg.pinv(J_21) @ dx_b
            
            print("q1_dot shape:", q1_dot.shape)
            q2_dot = np.linalg.pinv(J3 @ N1 @ N_21) @ (dx_sw - J3 @ N1 @ q1_dot)
            print("q2_dot shape:", q2_dot.shape)
            q_dot = q1_dot + q2_dot
            new_joint_angles = joint_angles + q_dot * self.step_size
            new_joint_angles = self.check_joint_limits(new_joint_angles)
            joint_angles = new_joint_angles

            joint_trajectory.append(joint_angles.copy())
            q_dot_trajectory.append(q_dot.copy())

        return joint_trajectory, q_dot_trajectory

    def cubic_spline(self, t, tf, xf):
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3

# Example Usage
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    ik = InverseKinematic(ROBOT_SCENE)
    ik.start_joint_updates()
    time.sleep(1.0)

    # Compute desired trajectories
    x_b, x_sw = ik.compute_desired_value()

    # Get initial joint angles
    initial_joint_angles = ik.get_current_joint_angles()
    # Calculate joint angles and velocities
    q_Traj, q_dot_Traj = ik.calculate(initial_joint_angles, x_b, x_sw)
    print("New Joint Angles:", q_Traj)

    # # Plotting the body trajectory
    # plt.figure(figsize=(12, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(x_b[:, 0], x_b[:, 2], label='Body Trajectory')
    # plt.xlabel('X Position')
    # plt.ylabel('Z Position')
    # plt.title('Body Trajectory')
    # plt.legend()

    # # Plotting the first swing leg trajectory
    # plt.subplot(3, 1, 2)
    # plt.plot(x_sw_1[:, 0], x_sw_1[:, 2], label='Swing Leg 1 Trajectory', color='r')
    # plt.xlabel('X Position')
    # plt.ylabel('Z Position')
    # plt.title('Swing Leg 1 Trajectory')
    # plt.legend()

    # # Plotting the second swing leg trajectory
    # plt.subplot(3, 1, 3)
    # plt.plot(x_sw_2[:, 0], x_sw_2[:, 2], label='Swing Leg 2 Trajectory', color='g')
    # plt.xlabel('X Position')
    # plt.ylabel('Z Position')
    # plt.title('Swing Leg 2 Trajectory')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
