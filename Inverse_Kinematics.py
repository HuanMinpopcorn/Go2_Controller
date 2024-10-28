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

        self.body_frame_name = "base_link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
        
        self.K = int(self.swing_time / self.step_size)

        self.joint_angles = self.get_current_joint_angles()
        self.x1, self.x2, self.x3 = self.get_required_state()
        self.J1, self.J2, self.J3 = self.get_required_jacobian()

    def check_joint_limits(self, joint_angles):
        for i in range(len(joint_angles)):
            lower_limit, upper_limit = self.model.jnt_range[i]
            joint_angles[i] = np.clip(joint_angles[i], lower_limit, upper_limit)
        return joint_angles

    def get_required_state(self):
        x1 = []
        x2 = np.zeros(6)
        x3 = []
        x4 = []

        for contact_leg in self.contact_legs:
            contact_state = self.get_body_state(contact_leg)
            contact_position = contact_state["position"]
            x1.append(contact_position)
        
        x1 = np.hstack(x1)

        body_state = self.get_body_state(self.body_frame_name)
        body_position = body_state["position"]
        body_orientation = body_state["orientation"]
        x2[:3] = body_position
        x2[3:] = body_orientation

        for swing_leg in self.swing_legs:
            swing_state = self.get_body_state(swing_leg)
            swing_position = swing_state["position"]
            x3.append(swing_position)

        x3 = np.hstack(x3)

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
        joint_angles = initial_joint_angles.copy()
        q_dot = np.zeros_like(joint_angles)
        joint_trajectory = [joint_angles.copy()]
        q_dot_trajectory = []
        for i in range(self.K):
            x_b = x_b[i]
            x_sw = x_sw[i]
         
            print("x_b:", x_b.shape, "x_sw:", x_sw.shape)
            self.x1, self.x2, self.x3 = self.get_required_state()
            self.J1, self.J2, self.J3 = self.get_required_jacobian()
            print("x1:", self.x1.shape, "x2:", self.x2.shape, "x3:", self.x3.shape)
            print("J1:", self.J1.shape, "J2:", self.J2.shape, "J3:", self.J3.shape)
            
            
            N1 = sp.null_space(self.J1)
  
            print("N1 shape:", N1.shape)
            print("J2 shape:", self.J2.shape)
            J_21 = self.J2 @ N1
            N_21 = sp.null_space(J_21)
            
            print("J_21:", J_21.shape)
            
            dx_b = (x_b - self.x2).reshape(-1, 1)
            dx_sw = (x_sw - self.x3).reshape(-1, 1)
            print("J3:,", self.J3.shape)
            print("N_21:", N_21.shape) 
            print((self.J3 @ N_21).shape)


            q_dot = np.linalg.pinv(J_21) @ dx_b  + np.linalg.pinv(self.J3 @ N_21) @ (dx_sw - self.J3 @ np.linalg.pinv(J_21) @ dx_b)
            
            # update the new joint angles 
            new_joint_angles = joint_angles + q_dot*self.step_size
            new_joint_angles = self.check_joint_limits(new_joint_angles)
            joint_angles = new_joint_angles
            joint_trajectory.append(joint_angles)
            q_dot_trajectory.append(q_dot.copy())
        return joint_trajectory , q_dot_trajectory

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

    # Example of computing desired values 
    x_b, x_sw = ik.compute_desired_value()
    print("Body Trajectory:", x_b.shape)
    print("Swing Leg Trajectory:", x_sw.shape)



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

    # Calculate the joint angles 
    initial_joint_angles = ik.get_current_joint_angles()
    q_Traj,q_dot_Traj = ik.calculate(initial_joint_angles, x_b, x_sw)
    print("New Joint Angles:", q_Traj)

