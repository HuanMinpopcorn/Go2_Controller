import mujoco
import numpy as np
from Kinematics import Kinematics
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import time
class InverseKinematic(Kinematics):
    def __init__(self, xml_path, step_size=0.01, tol=1e-3):
        super().__init__(xml_path)  # Initialize the parent class
        self.step_size = step_size
        self.tol = tol
        self.body_height = 0.25
        self.swing_height = 0.075
        self.swing_time = 0.25
        self.step_length = 0.1
        self.body_frame_name = "base_link"

    def check_joint_limits(self, joint_angles):
        for i in range(len(joint_angles)):
            lower_limit, upper_limit = self.model.jnt_range[i]
            joint_angles[i] = np.clip(joint_angles[i], lower_limit, upper_limit)
        return joint_angles

    def get_required_state(self, swing_legs, contact_legs):
        x1 = []
        x2 = np.zeros(6)
        x3 = []
        x4 = []

        for contact_leg in contact_legs:
            contact_state = self.get_body_state(contact_leg)
            contact_position = contact_state["position"]
            x1.append(contact_position)
        
        x1 = np.vstack(x1)

        body_state = self.get_body_state(self.body_frame_name)
        body_position = body_state["position"]
        body_orientation = body_state["orientation"]
        x2[:3] = body_position
        x2[3:] = body_orientation

        swing_state_1 = self.get_body_state(swing_legs[0])
        x3.append(swing_state_1["position"])
        swing_state_2 = self.get_body_state(swing_legs[1])
        x4.append(swing_state_2["position"])

        x3 = np.vstack(x3)
        x4 = np.vstack(x4)

        return x1, x2, x3, x4

    def get_required_jacobian(self, swing_legs, contact_legs):
        J1 = []
        J2 = np.zeros((6, self.model.nv))
        J3 = []
        J4 = []

        for contact_leg in contact_legs:
            jacobian_contact = self.get_jacobian(contact_leg)
            J1.append(jacobian_contact["J_pos"])
        
        J1 = np.vstack(J1)

        jacobian_body = self.get_jacobian(self.body_frame_name)
        J2 = jacobian_body["J_pos"]

        swing_jacobian_1 = self.get_jacobian(swing_legs[0])
        J3.append(swing_jacobian_1["J_pos"])
        swing_jacobian_2 = self.get_jacobian(swing_legs[1])
        J4.append(swing_jacobian_2["J_pos"])

        J3 = np.vstack(J3)
        J4 = np.vstack(J4)

        return J1, J2, J3, J4

    def compute_desired_value(self, velocity, swing_time, swing_height, body_height, swing_legs):
        time_step = 0.01
        K = int(swing_time / time_step)
        for i in range(K):
            body_state = self.get_body_state(self.body_frame_name)["position"]
            body_state[0] += velocity * i * time_step

            swing_leg_position_1 = self.get_body_state(swing_legs[0])["position"]
            swing_leg_position_1[0] += self.cubic_spline(i, K, velocity * swing_time)
            swing_leg_position_1[2] += swing_height * np.sin(np.pi * i / K)

            swing_leg_position_2 = self.get_body_state(swing_legs[1])["position"]
            swing_leg_position_2[0] += self.cubic_spline(i, K, velocity * swing_time)
            swing_leg_position_2[2] += swing_height * np.sin(np.pi * i / K)

        return body_state, np.vstack([swing_leg_position_1, swing_leg_position_2])

    def calculate(self, goal_position, initial_joint_angles):
        joint_angles = initial_joint_angles.copy()
        q_dot = np.zeros_like(joint_angles)
        
        swing_legs = ["FL_foot", "RR_foot"]
        contact_legs = ["FR_foot", "RL_foot"]
        
        J1, J2, J3, J4 = self.get_required_jacobian(swing_legs, contact_legs)
        x1, x2, x3, x4 = self.get_required_state(swing_legs, contact_legs)

        body_state, swing_leg_position = self.compute_desired_value(0.1, self.swing_time, self.swing_height, self.body_height, swing_legs)
        
        N1 = np.linalg.null(J1)
        N2 = np.linalg.null(J2)
        N3 = np.linalg.null(J3)

        J_21 = J2 @ N1
        N_21 = np.linalg.null(J_21)

        x_sw = np.concatenate((x3, x4), axis=0)
        J_sw = np.concatenate((J3, J4), axis=0)

        q_dot = np.linalg.pinv(J2 @ N1) @ (x2 - body_state) + np.linalg.pinv(J_sw @ N_21) @ ((x2 - body_state) - J_sw @ np.linalg.pinv(J2 @ N1) @ (x2 - body_state))
        return joint_angles + q_dot

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
    swing_legs = ["FL_foot", "RR_foot"]
    contact_legs = ["FR_foot", "RL_foot"]

    # current x1 x2 x3 x4
    while True:
        x1,x2,x3,x4 = ik.get_required_state(swing_legs, contact_legs)
        J1, J2, J3, J4 = ik.get_required_jacobian(swing_legs, contact_legs)
        time.sleep(1.0)
        x_b , x_sw = ik.compute_desired_value(0.1, ik.swing_time, ik.swing_height, ik.body_height, swing_legs)
        print("xb:", xb)
        print("x swing:", x_sw)
        