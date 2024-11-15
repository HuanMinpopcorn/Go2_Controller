import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg as sp
from Kinematics import Kinematics
from Send_motor_cmd import send_motor_commands
from error_plotting import ErrorPlotting

import matplotlib.pyplot as plt
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import time

from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import config




class InverseKinematic(Kinematics):
    """
    Inverse Kinematic Controller for Unitree Robots using MuJoCo simulation.
    This class extends Kinematics and controls robot joints based on trajectory planning.
    """

    def __init__(self, xml_path, cmd, step_size=config.SIMULATE_DT):
        super().__init__(xml_path)
        self.step_size = step_size
        self.swing_time =  0.25 # 0.25 # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing
  

        # Robot parameters
        self.body_height = 0.25
        self.swing_height = 0.075
        self.velocity = 0.0

        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]

    def initialize(self):
        """
        Initialize the robot state and joint angles.
        """
        self.initial_joint_angles = self.joint_state_reader.joint_angles.copy()

        # Get initial body and leg positions
        self.initial_body_state = self.get_body_state(self.body_frame_name).copy()
        self.initial_body_position = self.initial_body_state["position"]
        self.initial_body_orientation = self.initial_body_state["orientation"]
        self.initial_body_configuration = np.hstack((self.initial_body_position, self.initial_body_orientation))
        self.desired_body_configuration = self.initial_body_configuration.copy()
        # Initialize swing leg positions
        self.initial_swing_leg_positions = [
            np.copy(self.get_body_state(leg_name)["position"])
            for leg_name in self.swing_legs
        ]
        self.initial_swing_leg_positions = np.hstack(self.initial_swing_leg_positions)

        # Initialize the contact leg positions
        self.initial_contact_leg_positions = [  
            np.copy(self.get_body_state(leg_name)["position"])
            for leg_name in self.contact_legs
        ]
        self.initial_contact_leg_positions = np.hstack(self.initial_contact_leg_positions)


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

    def compute_desired_body_state(self):
        """
        Generate trajectory for the body over the swing phase.
        """
        print("Computing desired body state...")
        
        self.desired_body_configuration[0] += self.velocity * self.step_size
  
        return self.desired_body_configuration

    def compute_desired_swing_leg_trajectory(self):
        """
        Generate trajectories for the swing legs over the swing phase.
        """
        print("Computing desired swing leg trajectories...")
        swing_leg_trajectory = []

        # Initialize swing leg positions
        swing_leg_positions = [
            np.copy(self.get_body_state(leg_name)["position"])
            for leg_name in self.swing_legs
        ]
        swing_leg_positions = np.hstack(swing_leg_positions)

        swing_leg_positions_initial = np.copy(swing_leg_positions)

        for i in range(self.K):
            swing_leg_positions[0] = swing_leg_positions_initial[0] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[1] = swing_leg_positions_initial[1]  # Keep Y position constant
            swing_leg_positions[2] = swing_leg_positions_initial[2] + self.swing_height * np.sin(np.pi * i / self.K)
            swing_leg_positions[3] = swing_leg_positions_initial[3] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[4] = swing_leg_positions_initial[4]  # Keep Y position constant
            swing_leg_positions[5] = swing_leg_positions_initial[5] + self.swing_height * np.sin(np.pi * i / self.K)

            swing_leg_trajectory.append(swing_leg_positions.copy())

        return np.array(swing_leg_trajectory)

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
   

        # intial the API gain 
        kp = 250 + 50
        kd = 10
        # intial the gain for the body and swing leg
        kc = 1
        kb = 1 
        ks = 1


        m = self.model.nv
        i = 0
        trail = 0
        running_time = 0
        

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

        
        x_sw = self.compute_desired_swing_leg_trajectory()
        leg_pair_in_swing = True
        while True:
            i = (i + 1) % self.K  # Loop over the swing cycle duration
 
            if i == 0:
                # over one cycle
                self.transition_legs()
                # time.sleep(self.step_size)
                # x_b = self.compute_desired_body_trajectory()
                x_sw = self.compute_desired_swing_leg_trajectory()
                # print("Transitioning legs...")
            
            x_b = self.compute_desired_body_state()        # update the body state
            joint_angles = self.joint_state_reader.joint_angles
            x1, x2, x3 = self.get_required_state() 
            # x1 : contact leg positions, x2 : body state, x3 : swing leg positions
            J1, J2, J3 = self.get_required_jacobian()
            # J1 : contact leg Jacobian, J2 : body Jacobian, J3 : swing leg Jacobian

            config =  (self.initial_joint_angles - joint_angles) # shape (12, 1)
            q_err = kc * np.hstack((np.zeros(6), config)).reshape(-1, 1) # shape (18, 1)
            # print("q_err", config)
            
            # --------------------------------
            #
            #  Case 1: full-contact and standing up and down test
            # Jc = np.vstack([J1, J3])
            # Nc = np.eye(m) - np.linalg.pinv(Jc, rcond=1e-4) @ Jc
            # Jb_c = J2 @ Nc
            # Nb_c = np.eye(m) - np.linalg.pinv(Jb_c, rcond=1e-4) @ Jb_c

            # dx_b = kb * (x_b[i].T - x2).reshape(-1, 1) # body state : xyz + rpy 

            # q1_dot = np.linalg.pinv(Jb_c, rcond=1e-4) @ dx_b
            # q2_dot = Nb_c @ (q_err - q1_dot)

            # q_dot = q1_dot + q2_dot
            # # q_dot = q1_dot + q3_dot
            # dq_cmd = q_dot[6:].flatten()
            # new_joint_angles = joint_angles + dq_cmd  
            # ---------------------------------    
   

            # Case 2: three contact and 1 swing leg
            Jc = J1
            Nc = np.eye(m) - np.linalg.pinv(Jc, rcond=1e-5) @ Jc

            Jb_c = J2 @ Nc
            Nb_c = Nc - np.linalg.pinv(Jb_c, rcond=1e-5) @ Jb_c

            Jsw_bc = J3 @ Nb_c        
            Nsw_bc = Nb_c - np.linalg.pinv(Jsw_bc, rcond=1e-5) @ Jsw_bc

            dx_b = kb * (x_b - x2).reshape(-1, 1)
            dx_sw = ks * (x_sw[i].T - x3).reshape(-1, 1)

            q1_dot = np.linalg.pinv(Jb_c, rcond=1e-4) @ dx_b
            q2_dot = np.linalg.pinv(Jsw_bc, rcond=1e-4) @ (dx_sw - J3 @ q1_dot)
            q3_dot = Nsw_bc @ (q_err - q1_dot - q2_dot)

            q_dot = q1_dot + q2_dot + q3_dot


            dq_cmd = q_dot[6:].flatten()
            new_joint_angles = joint_angles + dq_cmd  

            xb_data.append(x_b.T)
            x2_data.append(x2.flatten())
            dx_b_data.append(dx_b.flatten())

            xw_data.append(x_sw[i].T)
            x3_data.append(x3.flatten())
            dx_sw_data.append(dx_sw.flatten())         

            q3_dot_data.append(self.change_q_order(q3_dot[6:].flatten()))
            q2_dot_data.append(self.change_q_order(q2_dot[6:].flatten()))
            q1_dot_data.append(self.change_q_order(q1_dot[6:].flatten()))

            cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles), self.change_q_order(dq_cmd))
            dq_error = kp * (self.change_q_order(new_joint_angles) - self.data.sensordata[:12])
            
            dq_error_data.append(dq_error)
            dq_dot = kd * (self.change_q_order(dq_cmd) - self.data.sensordata[12:24])
            dq_dot_data.append(dq_dot)
            ouput_data.append(dq_error + dq_dot)

            trail += 1
            if trail > 5000:
                break

        ErrorPlotting.plot_api_value(dq_error_data, dq_dot_data, ouput_data)
        ErrorPlotting.plot_q_dot(q3_dot_data, "q3_dot")
        ErrorPlotting.plot_q_dot(q2_dot_data, "q2_dot")
        ErrorPlotting.plot_q_dot(q1_dot_data , "q1_dot")
        ErrorPlotting.plot_state_error_trajectories(xb_data, x2_data, dx_b_data, "Body")
        ErrorPlotting.plot_state_error_trajectories(xw_data, x3_data, dx_sw_data, "Swing")
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
    



# Example Usage
if __name__ == "__main__":

    cmd = send_motor_commands()
    ik = InverseKinematic(config.ROBOT_SCENE, cmd)
    ik.start_joint_updates()
    cmd.move_to_initial_position()
    ik.initialize()
    ik.calculate()