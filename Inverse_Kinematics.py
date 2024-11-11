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


# import unitree_mujoco.simulate_python.unitree_mujoco as sim 


class InverseKinematic(Kinematics):
    """
    Inverse Kinematic Controller for Unitree Robots using MuJoCo simulation.
    This class extends Kinematics and controls robot joints based on trajectory planning.
    """

    def __init__(self, xml_path, cmd, step_size=0.01):
        super().__init__(xml_path)
        self.step_size = step_size
        self.swing_time = 0.25  # Duration of swing phase
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
            swing_leg_positions[2] = swing_leg_positions_initial[2] + self.swing_height * np.sin(np.pi * i / self.K )
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
        
        kd = 3.5
        m = self.model.nv
        i = 0

        # State variable to keep track of which leg pair is swinging
        leg_pair_in_swing = True

        print("Starting Trot Gait...")
        # while i < self.K:
        while True:
            i = (i + 1) % self.K  # Loop over the swing cycle duration
            phase = i / self.K  # Phase variable for cubic spline trajectory
            kp = 50.0 * phase + (1 - phase) * 20.0  # Gradual stiffness


            # # i = min(i + 1, self.K -1)  # Increment index but keep it within bounds

            # Toggle leg pairs at the end of each phase
            if i == 0:
                leg_pair_in_swing = not leg_pair_in_swing
                self.transition_legs()
                x_b , x_sw = self.compute_desired_value()

            # Select the active swing and stance leg pairs based on the phase
            if leg_pair_in_swing:
                active_swing_legs = self.swing_legs
                active_stance_legs = self.contact_legs
            else:
                active_swing_legs = self.contact_legs
                active_stance_legs = self.swing_legs
                

            # Get current joint angles and required state
            joint_angles = self.joint_state_reader.joint_angles  #current joint angle in the order of FL, FR, RL, RR
            x1, x2, x3 = self.get_required_state()
            J1, J2, J3 = self.get_required_jacobian()

            stand_up_joint_pos = np.array([
                0.052, 1.12, -2.10, -0.052, 1.12, -2.10,
                0.052, 1.12, -2.10, -0.052, 1.12, -2.10
            ], dtype=float) # stand up joint angle order FR, FL, RR, RL

        
            # Initial joint position error
            config = 1 * (self.change_q_order(stand_up_joint_pos) - joint_angles)
            q_err_temp = config.reshape(-1, 1)
            zero_vector = np.zeros((6, 1)).reshape(-1, 1)
            q_err = np.vstack((zero_vector, q_err_temp))

            # Desired body position - current body position
            dx_b = (x_b[i].T - x2).reshape(-1, 1)
            # Desired swing position - current swing position
            dx_sw = 10 * (x_sw[i].T - x3).reshape(-1, 1)
            # dx_sw = np.zeros((6, 1))
            print("dx_sw: ", dx_sw.T)

            N1 = np.eye(m) - np.linalg.pinv(J1,rcond=1e-4) @ J1      
            J_21 = J2 @ N1
            N_21 = np.eye(m) - np.linalg.pinv(J_21,rcond=1e-4) @ J_21      
            np.set_printoptions(precision=3, suppress=True)

            # print("J_21 inv: ", np.linalg.pinv(J_21,rcond=1e-4))

            q1_dot = np.linalg.pinv(J_21,rcond=1e-4) @ dx_b

            q2_dot = np.linalg.pinv(J3 @ N_21,rcond=1e-4) @ (dx_sw - J3 @ q1_dot)
            N_321 = np.eye(m) - np.linalg.pinv(J3 @ N_21,rcond=1e-4) @ J3 @ N_21
            q3_dot = N_321 @ q_err
            q_dot = q1_dot + q2_dot + q3_dot
            # q_dot = q1_dot + q3_dot 
            # q_dot = q1_dot 

            dq_cmd = q_dot[6:].flatten()

            # Compute new joint angles
            new_joint_angles = joint_angles + dq_cmd

            # gravity_torque = [3.09, 0.556, 6.61, -3.09, 0.561, 6.63, 0.616, 0.712, 7.2, -0.613, 0.7121, 7.21]
            gravity_torque = np.array(self.data.qfrc_bias[6:]).flatten()
            print("gravity_torque: ", gravity_torque)

            # Ensure joint angles are within limits
            cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles), self.change_q_order(dq_cmd))
            # cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles), self.change_q_order(dq_cmd), gravity_torque)
            # cmd.send_motor_commands(kp, kd, (new_joint_angles),  (dq_cmd))


            print("q1_dot: ", q1_dot[6:].T)
            print("q2_dot: ", q2_dot[6:].T)
            print("q3_dot: ", q3_dot[6:].T)

         
            # sensor data order FR, FL, RR, RL
            # cmd order FR, FL, RR, RL


            # new joint angles, dq, joint angles order FL, FR, RL, RR
            print("Motor commands sent.", (dq_cmd))
            print("Joint angles updated.", (new_joint_angles) )
            print("current joint angles: ",(joint_angles) )

            print("---------------------------------------------------")
            print("dq error" , kp * ((new_joint_angles) - self.change_q_order(self.data.sensordata[:12])))
            # print("dq error" , kp * ((new_joint_angles) - self.data.sensordata[:12]))
            print("---------------------------------------------------")
            print("dq_dot: ", kd * ((dq_cmd) - self.change_q_order(self.data.sensordata[12:24])))
            print("---------------------------------------------------")
            print("ctrl ouput", self.change_q_order((kp * ((new_joint_angles) - self.change_q_order(self.data.sensordata[:12])) + kd * ((dq_cmd) - self.change_q_order(self.data.sensordata[12:24])))))
            # print("dq_dot: ", kd * ((dq_cmd) - self.data.sensordata[12:24]))

    
            print("iteration: ", i) 
            print("---------------------------------------------------")

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

# Example Usage
if __name__ == "__main__":


    robot_scene = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    cmd = send_motor_commands()
    ik = InverseKinematic(robot_scene, cmd)
    ik.start_joint_updates()
    cmd.move_to_initial_position()
    ik.calculate()