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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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
        self.velocity = 0.0

        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
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
            body_trajectory.append(body_configuration.copy())

            # Generate swing leg trajectories
               
            swing_leg_positions[0] = swing_leg_positions_initial[0] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[2] = swing_leg_positions_initial[2] + self.swing_height * np.sin(np.pi * i / self.K)
            swing_leg_positions[3] = swing_leg_positions_initial[3] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[5] = swing_leg_positions_initial[5] + self.swing_height * np.sin(np.pi * i / self.K)

            swing_leg_trajectory.append(swing_leg_positions.copy())

        # Plot the trajectories
        body_trajectory = np.array(body_trajectory)
        swing_leg_trajectory = np.array(swing_leg_trajectory)

        # plt.figure(figsize=(12, 6))

        # Plot body trajectory
        # plt.subplot(1, 2, 1)
        # plt.plot(body_trajectory[:, 0], body_trajectory[:, 2], label='Body Trajectory')
        # plt.xlabel('X Position (m)')
        # plt.ylabel('Z Position (m)')
        # plt.title('Body Trajectory')
        # plt.legend()
        # plt.grid()

        # # Plot swing leg trajectory
        # plt.subplot(1, 2, 2)
        # plt.plot(swing_leg_trajectory[:, 0], swing_leg_trajectory[:, 2], label='Swing Leg Trajectory (FL)')
        # plt.plot(swing_leg_trajectory[:, 3], swing_leg_trajectory[:, 5], label='Swing Leg Trajectory (RR)')
        # plt.xlabel('X Position (m)')
        # plt.ylabel('Z Position (m)')
        # plt.title('Swing Leg Trajectory')
        # plt.legend()
        # plt.grid()

        # plt.tight_layout()
        # plt.show()
        return np.array(body_trajectory), np.array(swing_leg_trajectory)

    def cubic_spline(self, t, tf, xf):
        """
        Generate cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3

    def calculate(self):
        """
        Main loop to compute and update joint angles in real-time,
        including a trot gait with proper leg phasing.
        """
        x_b, x_sw = self.compute_desired_value()
        kp = 5 # Proportional gain
        m = self.model.nv
        i = 0

        # State variable to keep track of which leg pair is swinging
        leg_pair_in_swing = True

        print("Starting Trot Gait...")
        while i <  5:
            # i = (i + 1) % self.K  # Loop over the swing cycle duration
            i = min(i + 1, self.K)  # Increment index but keep it within bounds

            # # Toggle leg pairs at the end of each phase
            # if i == 0:
            #     leg_pair_in_swing = not leg_pair_in_swing
            #     self.transition_legs()

            # Select the active swing and stance leg pairs based on the phase
            # if leg_pair_in_swing:
            #     active_swing_legs = self.swing_legs
            #     active_stance_legs = self.contact_legs
            # else:
            #     active_swing_legs = self.contact_legs
            #     active_stance_legs = self.swing_legs

            # Get current joint angles and required state
            joint_angles = self.get_current_joint_angles()
            x1, x2, x3 = self.get_required_state()
            J1, J2, J3 = self.get_required_jacobian()

            stand_up_joint_pos = np.array([
                0.052, 1.12, -2.10, -0.052, 1.12, -2.10,
                0.052, 1.12, -2.10, -0.052, 1.12, -2.10
            ], dtype=float)
            

            # Initial joint position error
            config = 20* (stand_up_joint_pos - joint_angles)
            q_err_temp = config.reshape(-1, 1)
            zero_vector = np.zeros((6, 1)).reshape(-1, 1)
            q_err = np.vstack((zero_vector, q_err_temp))

            # Desired body position - current body position
            dx_b = kp * (x_b[i].T - x2).reshape(-1, 1)
            # Desired swing position - current swing position
            dx_sw = (x_sw[i].T - x3).reshape(-1, 1)
            
            print("x1", x1)
            print("x2", x2)
            print("x3", x3)
            print("x_b i-1: ", x_b[10].T)
            
            print("x_b: ", x_b[1].T)
            print("x_sw i-1: ", x_sw[10].T)
            print("x_sw: ", x_sw[1].T)
            print("dx_b: ", dx_b.T)
            print("dx_sw: ", dx_sw.T)


            N1 = np.eye(m) - np.linalg.pinv(J1) @ J1
            # N1 = J1 @ np.linalg.pinv(J1.T @ J1) @ J1.T
            # print("N1: ", N1.shape)
            # N1_min = sp.null_space(J1)
            # print("N1_min: ", N1_min.shape)
            # print("N1_lee: ", N1_lee.shape)

            print("J2: ", J2.shape)
            print("N1: ", N1.shape)
            J_21 = J2 @ N1

            N_21 = np.eye(m) - np.linalg.pinv(J_21) @ J_21
            # N_21 = J_21 @ np.linalg.pinv(J_21.T @ J_21) @ J_21.T           # Compute joint velocities
            
            q1_dot = np.linalg.pinv(J_21) @ dx_b
            q2_dot = np.linalg.pinv(J3 @ N_21) @ (dx_sw - J3 @ q1_dot)
            q3_dot = N_21 @ q_err
            # q_dot = q1_dot + q2_dot + q3_dot
            q_dot = q1_dot + q2_dot
            dq_cmd = q_dot[6:m].flatten()

            # Compute new joint angles
            new_joint_angles = joint_angles + dq_cmd * self.step_size
            # Check joint limits
            # new_joint_angles = self.check_joint_limits(new_joint_angles)
            
            # # Get required torques
            # # Use MuJoCo's inverse dynamics function to compute required torques
            # self.data.qpos[:] = 0  # Reset accelerations
            # self.data.qpos[7:] = new_joint_angles 
            # # print(self.data.qpos.shape)
            # self.data.qvel[:] = 0  # Reset velocities
            # self.data.qvel[6:] = dq_cmd
            
            
            # mujoco.mj_inverse(self.model, self.data)
            # required_torques = self.data.qfrc_inverse[6:]*0.01
            required_torques = [2.77,0.8,6.85,-2.77,0.8,6.85,
                                3.28,0.384,6.96,-3.28,0.384,6.96]
            # Send motor commands
            # time.sleep(self.step_size*10)
            # self.send_motor_commands(new_joint_angles, dq_cmd, required_torques)
           
            
            print("q1_dot: ", q1_dot.T)
            print("q2_dot: ", q2_dot.T)
            # print("q3_dot: ", q3_dot.T)
            
            print("Motor commands sent.", dq_cmd)
            print("Joint angles updated.", new_joint_angles)
            print("prev joint angles: ", joint_angles)
            print("command data", self.data.ctrl)
            print("iteration: ", i) 
            # print("Required torques: ", required_torques)
            
            self.send_motor_commands(new_joint_angles, dq_cmd, required_torques)
    def transition_legs(self):
        """
        Swap the swing and contact legs for the next cycle.
        """
        self.swing_legs, self.contact_legs = self.contact_legs, self.swing_legs
        # print("Legs transitioned: Swing legs ->", self.swing_legs, ", Contact legs ->", self.contact_legs)

    def send_motor_commands(self, new_joint_angles, q_dot, torque=0.0):
        """
        Send motor commands to the robot using the publisher.
        """

        for i, angle in enumerate(new_joint_angles):
            self.cmd.motor_cmd[i].q  = angle
            self.cmd.motor_cmd[i].dq = q_dot[i]
            self.cmd.motor_cmd[i].kp = 10
            self.cmd.motor_cmd[i].kd = 5
            self.cmd.motor_cmd[i].tau = torque[i]

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

        print("Transitioning to initial position...")
        print("Press Ctrl+ C to enter Trot Gait ...")
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
    def apply_gravity_compensation(self):
        """
        Apply gravity compensation torques to keep the robot upright in MuJoCo.
        """
        # Get the gravity compensation forces/torques from qfrc_bias
        # qfrc_bias includes the effects of gravity, Coriolis, and centrifugal forces
        gravity_compensation = self.data.qfrc_bias

        # Apply these torques to the actuators
        self.data.ctrl[:] = gravity_compensation[6:]
        print("Gravity compensation applied.", gravity_compensation)
            

# Example Usage
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    crc = CRC()
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    robot_scene = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    ik = InverseKinematic(robot_scene, crc, pub)
    ik.start_joint_updates()
    # ik.apply_gravity_compensation()
    ik.move_to_initial_position()
    
    ik.calculate()
