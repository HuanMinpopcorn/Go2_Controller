import mujoco
import numpy as np
import scipy.linalg as sp
from Kinematics import Kinematics
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import time
import matplotlib.pyplot as plt
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

class InverseKinematic(Kinematics):
    """
    Inverse Kinematic Controller for Unitree Robots using MuJoCo simulation.
    This class extends Kinematics and controls robot joints based on trajectory planning.
    """

    def __init__(self, xml_path, crc, pub, step_size=0.01, tol=1e-3):
        super().__init__(xml_path)
        self.step_size = step_size
        self.swing_time = 0.25  # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing
        self.tol = tol

        # Robot parameters
        self.body_height = 0.25
        self.swing_height = 0.075
        self.velocity = 0.1

        self.body_frame_name = "Base_Link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]

        # Initialize publisher and CRC utilities
        self.pub = pub  
        self.crc = crc  

        # Initialize motor command message
        self.cmd = self._initialize_motor_commands()

        # Initialize joint angles
        # self.joint_angles = self.get_current_joint_angles()

    def _initialize_motor_commands(self):
        """
        Initialize motor command message with default settings.
        """
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head = [0xFE, 0xEF]
        cmd.level_flag = 0xFF  # Low-level control

        for motor in cmd.motor_cmd:
            motor.mode = 0x01  # PMSM mode
            motor.q = 0.0
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        return cmd

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
        body_trajectory, swing_trajectory_1, swing_trajectory_2 = [], [], []

        for i in range(self.K):
            body_position = self.get_body_state(self.body_frame_name)["position"]
            body_orientation = self.get_body_state(self.body_frame_name)["orientation"]
            body_position[0] += self.velocity * i * self.step_size
            body_trajectory.append(np.hstack((body_position, body_orientation)))

            for swing_leg_trajectory, leg_name in zip(
                [swing_trajectory_1, swing_trajectory_2], self.swing_legs
            ):
                leg_position = self.get_body_state(leg_name)["position"]
                leg_position[0] += self.cubic_spline(i, self.K, self.velocity * self.swing_time)
                leg_position[2] += self.swing_height * np.sin(np.pi * i / self.K)
                swing_leg_trajectory.append(leg_position.copy())

        swing_Traj = np.hstack((swing_trajectory_1, swing_trajectory_2))
        return np.array(body_trajectory), swing_Traj

    def cubic_spline(self, t, tf, xf):
        """
        Generate cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3

    def calculate(self, x_b, x_sw):
        """
        Main loop to compute and update joint angles in real-time.
        """
        i = 0
        while True:
            i = min(i + 1, self.K - 1)  # Prevent index out of bounds
            
            joint_angles = self.get_current_joint_angles() # Get current joint angles from mujoco
            x1, x2, x3 = self.get_required_state()
            J1, J2, J3 = self.get_required_jacobian()


            print(f"Current joint angles: {joint_angles}")
            print(f"Desired body position: {x_b[i]}")
            print(f"Desired swing position: {x_sw[i]}")

            dx_b = (x_b[i].T - x2).reshape(-1, 1)
            dx_sw = (x_sw[i].T - x3).reshape(-1, 1)

            N1 = sp.null_space(J1)
            J_21 = J2 @ N1
            N_21 = sp.null_space(J_21)

            q1_dot = np.linalg.pinv(J_21) @ dx_b
            q2_dot = np.linalg.pinv(J3 @ N1 @ N_21) @ (dx_sw - J3 @ N1 @ q1_dot)
            q_dot = q1_dot + q2_dot
            
            new_joint_angles = joint_angles.reshape(-1,1) + q_dot * self.step_size

            # new_joint_angles = self.check_joint_limits(joint_angles.reshape(-1,1) + q_dot * self.step_size)
            print(f"New joint angles: {new_joint_angles.T}")
            # print(f"New joint angles: {new_joint_angles.shape}")
            self.send_motor_commands(new_joint_angles, q_dot)

    def send_motor_commands(self, new_joint_angles, q_dot):
        """
        Send motor commands to the robot using the publisher.
        """

        for i, angle in enumerate(new_joint_angles):
            self.cmd.motor_cmd[i].q = angle
            self.cmd.motor_cmd[i].dq = q_dot[i]
            self.cmd.motor_cmd[i].kp = 10.0
            self.cmd.motor_cmd[i].kd = 3.5

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
        time.sleep(self.step_size)



    def move_to_initial_position(self):
        """
        Smoothly transition to the initial standing position and maintain it.
        """
        # Define initial and final joint positions for smooth transition
        stand_up_joint_pos = np.array([
            0.052, 1.12, -2.10, -0.052, 1.12, -2.10,
            0.052, 1.12, -2.10, -0.052, 1.12, -2.10
        ], dtype=float)

        stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ], dtype=float)


        running_time = 0.0
        try:
            while True:
                # Check if the robot is at the target body height
                running_time += self.step_size
                
                # Smoothly transition to the initial position
                phase = np.tanh(running_time / 1.2)
                for i in range(12):
                    self.cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (
                        1 - phase) * stand_down_joint_pos[i]
                    self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0  # Gradual stiffness
                    self.cmd.motor_cmd[i].kd = 3.5

                self.cmd.crc = self.crc.Crc(self.cmd)
                self.pub.Write(self.cmd)
                time.sleep(self.step_size)
        except KeyboardInterrupt:
            # Gracefully stop the joint update thread on exit
            pass

# Example Usage
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    crc = CRC()
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    robot_scene = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    ik = InverseKinematic(robot_scene, crc, pub)
    ik.start_joint_updates()
    ik.move_to_initial_position()

    # Compute desired trajectories
    try : 
        x_b, x_sw = ik.compute_desired_value()
        print("Desired body trajectory:", x_b)
        print("Desired swing trajectory:", x_sw)
    except KeyboardInterrupt:
        pass
    # joint_angles = ik.get_current_joint_angles()
    # print(f"Current joint angles: {joint_angles}")
    # print(f"Desired body position: {x_b[1]}")
    # print(f"Desired swing position: {x_sw[1]}")
    # Calculate joint angles and update in real-time
    ik.calculate(x_b, x_sw)
