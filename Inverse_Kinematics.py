import mujoco
import numpy as np

from Forward_Kinematics import ForwardKinematic
from Send_motor_cmd import send_motor_commands
from error_plotting import ErrorPlotting
import matplotlib.pyplot as plt
import time
import Simulation.config as config
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import multiprocessing
from multiprocessing import Process, Pipe
import time 
from tqdm import tqdm

class InverseKinematic(ForwardKinematic):
    """
    Inverse Kinematic Controller for Unitree Robots using MuJoCo simulation.
    This class extends Kinematics and controls robot joints based on trajectory planning.
    """

    def __init__(self, swing_legs, contact_legs):   
        super().__init__(config.ROBOT_SCENE)
        # mujoco.mj_step(self.model, self.data)

        # Initialize the leg positions
        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
        self.num_actuated_joints = self.model.nv - 6
   
        # Initialize the API
        self.cmd = send_motor_commands()
        self.ErrorPlotting = ErrorPlotting()

        # intial the API gain 
        self.kp = 200
        self.kd = 8
        self.swing_phase = 0
        # self.walk_phase = "double_standing" # intial phase
        # intialize output for Inverse Dynamics shape (12, 1)
    
        self.previous_qd = np.zeros((12, 1))
        self.previous_dqd = np.zeros((12, 1))

        self.swing_legs = swing_legs
        self.contact_legs = contact_legs
    # TODO: Get Initialize the robot state, joint angles, and leg positions.
    def initialize(self):
        """
        Initialize the robot state and joint angles.
        """
        # print("Initializing inverse kinematic parameter...")
        self.qd = np.zeros((12, 1))   
        self.dqd = np.zeros((12, 1))
        self.ddqd = np.zeros((12, 1))
        self.ddq_cmd = np.zeros((12, 1))
 
        # self.tau = self.joint_toque.copy()
        self.tau = np.zeros((12, 1))
        
        self.initial_joint_angles = self.joint_angles.copy()
        self.initial_joint_velocity = self.joint_velocity.copy()

       
        self.initial_body_configuration = self.get_required_state()["body"].copy()   
        self.initial_swing_leg_positions = self.get_required_state()["swing_leg"].copy()
        self.initial_contact_leg_positions = self.get_required_state()["contact_leg"].copy()

        self.initial_body_jacobian = np.vstack((self.get_jacobian(self.body_frame_name)["J_pos"], self.get_jacobian(self.body_frame_name)["J_rot"]))[:,6:18]
        self.initial_body_velocity = self.initial_body_jacobian @ self.initial_joint_velocity

        self.initial_swing_leg_jacobian = np.vstack([self.get_jacobian(leg)["J_pos"] for leg in self.swing_legs])[:,6:18]
        self.initial_swing_leg_velocity = self.initial_swing_leg_jacobian @ self.initial_joint_velocity

        return  self.initial_body_configuration, self.initial_swing_leg_positions, self.initial_contact_leg_positions, self.initial_body_velocity, self.initial_swing_leg_velocity

      
    def check_joint_limits(self, joint_angles):
        """
        Ensure the joint angles are within the limits defined in the MuJoCo model.
        """
        for i, angle in enumerate(joint_angles):
            lower_limit, upper_limit = self.model.jnt_range[i, :]
            joint_angles[i] = np.clip(angle, lower_limit, upper_limit)
        return joint_angles
    # TODO: Get the required state for inverse kinematic calculation, including contact and swing leg positions, and body state.
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

        return {
            "contact_leg": np.hstack(contact_positions),
            "body": body_state_combined,
            "swing_leg": np.hstack(swing_positions)
        }
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

        return {
            "contact_leg": J1,
            "body": J2,
            "swing_leg": J3
        }
    def get_required_jacobian_dot(self):
        """
        Compute Jacobians for the contact legs, body, and swing legs.
        """
        J1_dot = np.vstack([self.get_jacobian_dot(leg)["Jp_dot"] for leg in self.contact_legs])
        J2_dot = np.zeros((6, self.model.nv))
        body_jacobian = self.get_jacobian_dot(self.body_frame_name)
        J2_dot[:3, :] = body_jacobian["Jp_dot"]
        J2_dot[3:, :] = body_jacobian["Jr_dot"]
        J3_dot = np.vstack([self.get_jacobian_dot(leg)["Jp_dot"] for leg in self.swing_legs])

        return {
            "contact_leg": J1_dot,
            "body": J2_dot,
            "swing_leg": J3_dot
        }

    def desired_joint_position_and_velocity(self,J1, J2, J3, x1, x2, x3, x_b, x_sw, i, joint_angles):

        config =  (self.initial_joint_angles - joint_angles)            # shape (12, 1)
        q_err = np.hstack((np.zeros(6), config))    # shape (18, 1)

        m = self.model.nv
        Jc = J1
        Nc = np.eye(m) - np.linalg.pinv(Jc, rcond=1e-6) @ Jc

        Jb_c = J2 @ Nc
        Nb_c = Nc - np.linalg.pinv(Jb_c, rcond=1e-6) @ Jb_c

        Jsw_bc = J3 @ Nb_c        
        Nsw_bc = Nb_c - np.linalg.pinv(Jsw_bc, rcond=1e-6) @ Jsw_bc
        # print("Body  shape:", x_b[i].T.shape, "Swing leg  shape:", x_sw[i].T.shape)
        dx_b = (x_b[i].T - x2) # desired body state error - dotx_b
        dx_sw = (x_sw[i].T - x3) # desired swing leg state error - dotx_sw

        q2_dot = np.linalg.pinv(Jb_c, rcond=1e-6) @ dx_b  # body q_dot
        q3_dot = np.linalg.pinv(Jsw_bc, rcond=1e-6) @ (dx_sw - J3 @ q2_dot) # swing leg q_dot
        q1_dot = Nsw_bc @ (q_err - q2_dot - q3_dot)

        q_dot = q1_dot + q2_dot + q3_dot            # update dqd
        # q_dot = q3_dot + q2_dot 
        
        dq_cmd = q_dot[6:].flatten()                # shape (12, 1)
        new_joint_angles = joint_angles + dq_cmd    # update qd
   

        self.ErrorPlotting.xb_data.append(x_b[i].T)
        self.ErrorPlotting.x2_data.append(x2.flatten())
        self.ErrorPlotting.dx_b_data.append(dx_b.flatten())

        self.ErrorPlotting.xw_data.append(x_sw[i].T)
        self.ErrorPlotting.x3_data.append(x3.flatten())
        self.ErrorPlotting.dx_sw_data.append(dx_sw.flatten())         

     
        return new_joint_angles, dq_cmd #Kinematic order 
    
    def desired_joint_acceleration(self, J1, J2, J3, J1_dot, J2_dot, J3_dot, i, x_b_dot, x_sw_dot, joint_angles_velcity):
        """
        Compute desired joint acceleration using KinWBC null-space projections
        Returns:
            ddqd: Desired joint acceleration (12x1 vector)
        """
        # Initialize variables
        ddqd_desired = np.zeros((self.model.nv-6, 1))  # num_actuated_joints = 12
        vel_error = (self.initial_joint_velocity - joint_angles_velcity) # shape (12, 1)
        dq_err = np.hstack((np.zeros((6)), vel_error))  # shape (18, 1)

        Jc = J1
        Nc = np.eye(self.model.nv) - np.linalg.pinv(Jc, rcond=1e-5) @ Jc

        Jb_c = J2 @ Nc
        Nb_c = Nc - np.linalg.pinv(Jb_c, rcond=1e-5) @ Jb_c

        Jsw_bc = J3 @ Nb_c        
        Nsw_bc = Nb_c - np.linalg.pinv(Jsw_bc, rcond=1e-5) @ Jsw_bc

        # Compute body and swing leg velocities
        full_q_velocity = np.zeros(self.model.nv)

        full_q_velocity[:3] = np.array(self.robot_velocity).reshape(-1)
        full_q_velocity[3:6] = np.array(self.body_angular_velocity).reshape(-1)
        full_q_velocity[6:18] = np.array(joint_angles_velcity).reshape(-1)
        x2_dot = J2 @ full_q_velocity     # Body velocity
        x3_dot = J3 @ full_q_velocity     # Swing leg velocity

        # Compute velocity error
        dx_b_dot = x_b_dot[i] - x2_dot  # Body velocity error
        dx_sw_dot = x_sw_dot[i] - x3_dot  # Swing leg velocity error

        J1_dotq = J1_dot @ full_q_velocity
        J2_dotq = J2_dot @ full_q_velocity
        J3_dotq = J3_dot @ full_q_velocity

        # \delta q_dot =  J(xdot_des - xdot) - J_dot * q_dot
        # Compute desired joint acceleration
        q2_ddot = np.linalg.pinv(Jb_c, rcond=1e-4) @ (dx_b_dot - J2_dotq) # Body joint acceleration
        q3_ddot = np.linalg.pinv(Jsw_bc, rcond=1e-4) @ (dx_sw_dot - J3_dotq - J3 @ q2_ddot)  # Swing leg joint acceleration
        # q1_ddot = Nsw_bc @ (dq_err  - q3_ddot - q2_ddot)  # Joint acceleration for the remaining joints
        ddqd_desired = q3_ddot + q2_ddot
        # ddqd_desired = q1_ddot + q2_ddot + q3_ddot

        self.ErrorPlotting.xb_dot_data.append(x_b_dot[i])
        self.ErrorPlotting.x2_dot_data.append(x2_dot.flatten())
        self.ErrorPlotting.dx_b_dot_data.append(dx_b_dot.flatten())

        self.ErrorPlotting.xw_dot_data.append(x_sw_dot[i])
        self.ErrorPlotting.x3_dot_data.append(x3_dot.flatten())
        self.ErrorPlotting.dx_sw_dot_data.append(dx_sw_dot.flatten())

        return ddqd_desired[6:]  # Return only actuated joints (12x1)

    def calculate(self, x_sw, x_b, x_sw_dot, x_b_dot, index):
        """
        Main loop to compute and update joint angles in real-time,
        including a trot gait with proper leg phasing.
        """
        
                    
        joint_angles = self.joint_angles # get the current joint angles
        joint_angles_velcity = self.joint_velocity # get the current joint angles
        
    
        self.required_state = self.get_required_state()          # get the required state for IK
        self.x1 = self.required_state["contact_leg"]
        self.x2 = self.required_state["body"]
        self.x3 = self.required_state["swing_leg"]
        # x1 : contact leg positions, x2 : body state, x3 : swing leg positions

        
        self.J1 = self.get_required_jacobian()["contact_leg"]
        self.J2 = self.get_required_jacobian()["body"]
        self.J3 = self.get_required_jacobian()["swing_leg"]
        # J1 : contact leg Jacobian, J2 : body Jacobian, J3 : swing leg Jacobian
        self.J1_dot = self.get_required_jacobian_dot()["contact_leg"]
        self.J2_dot = self.get_required_jacobian_dot()["body"]
        self.J3_dot = self.get_required_jacobian_dot()["swing_leg"]
        # J1_dot : contact leg Jacobian, J2_dot : body Jacobian, J3_dot : swing leg Jacobian

        # calculate the desired joint angles and joint velocities
        self.qd, self.dqd = self.desired_joint_position_and_velocity(self.J1, self.J2, self.J3, 
                                                            self.x1, self.x2, self.x3, 
                                                            x_b, x_sw, 
                                                            index, joint_angles)
        
        self.ddqd = self.desired_joint_acceleration(self.J1, self.J2, self.J3,
                                                            self.J1_dot, self.J2_dot, self.J3_dot,
                                                            index, x_b_dot, x_sw_dot,joint_angles_velcity)
        
   

        dq_error = self.kp * (self.qd.reshape(-1, 1) - self.joint_angles.reshape(-1, 1))
        dq_dot_error = self.kd * ((self.dqd.reshape(-1, 1)) - self.joint_velocity.reshape(-1, 1))
        # print("dq_error", dq_error)
        # print("dq_dot_error", dq_dot_error)
        self.ddq_cmd = self.ddqd.reshape(-1, 1) + dq_error.reshape(-1, 1) + dq_dot_error.reshape(-1, 1)  # desired joint acceleration
        
       
        
        # get the desired q qdot qddot and current q, qdot 
        self.ErrorPlotting.q_desired_data.append(self.qd)
        self.ErrorPlotting.q_current_data.append(self.joint_angles)
        self.ErrorPlotting.q_error_data.append(dq_error)

        self.ErrorPlotting.dq_desired_data.append(self.dqd)
        self.ErrorPlotting.dq_current_data.append(self.joint_velocity)
        self.ErrorPlotting.dq_error_data.append(dq_dot_error)
    
        self.ErrorPlotting.ddq_desired_data.append(self.ddqd)
        self.ErrorPlotting.ddq_current_data.append(self.data.sensordata[self.num_actuated_joints * 2 :self.num_actuated_joints * 3])
        self.ErrorPlotting.ddq_error_data.append(self.ddqd - self.data.sensordata[self.num_actuated_joints * 3 :self.num_actuated_joints * 4])

        self.ErrorPlotting.FL_position.append(self.get_body_state("FL_foot")["position"])
        self.ErrorPlotting.FR_position.append(self.get_body_state("FR_foot")["position"])
        self.ErrorPlotting.RL_position.append(self.get_body_state("RL_foot")["position"])
        self.ErrorPlotting.RR_position.append(self.get_body_state("RR_foot")["position"])

        return self.qd, self.dqd, self.ddqd, self.ddq_cmd
    def send_command_ik(self):
        """
        Send the computed torques to the robot.
        """
        # print("Sending command API...")
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        self.tau = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - self.data.qfrc_inverse.reshape(-1, 1))
        # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.change_q_order(self.tau))
        self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd))

            
 
    def change_q_order(self, q):
        """
        Change the order of the joint angles.
        """
        return np.array(
            [
                q[3], q[4], q[5], q[0], q[1], q[2], q[9], q[10], q[11], q[6], q[7], q[8]
            ]
        )

    def transition_legs(self):
        """
        Swap the swing and contact legs for the next cycle.
        """
        self.swing_legs, self.contact_legs = self.contact_legs, self.swing_legs
        if self.swing_phase == 0:
            self.swing_phase = 1
        else:
            self.swing_phase = 0

    def plot_error_ik(self):
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.q_desired_data, 
                                        self.ErrorPlotting.q_current_data,
                                        self.ErrorPlotting.q_error_data,
                                        "qd_send")
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.dq_desired_data, 
                                        self.ErrorPlotting.dq_current_data,
                                        self.ErrorPlotting.dq_error_data,
                                        "dqd_send")
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.ddq_desired_data,
                                        self.ErrorPlotting.ddq_current_data,
                                        self.ErrorPlotting.ddq_error_data,
                                        "ddqd_send")
     
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_data, 
                                                         self.ErrorPlotting.x2_data,
                                                         self.ErrorPlotting.dx_b_data, 
                                                         "Body")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_data, 
                                                         self.ErrorPlotting.x3_data, 
                                                         self.ErrorPlotting.dx_sw_data, 
                                                         "Swing")
        self.ErrorPlotting.plot_foot_location(
                              self.ErrorPlotting.FL_position,   
                              self.ErrorPlotting.FR_position, 
                              self.ErrorPlotting.RL_position, 
                              self.ErrorPlotting.RR_position, 
                              "foot location") # FL, FR, RL, RR,
        # self.ErrorPlotting.plot_torque(self.ErrorPlotting.tau_data,"joint toque IK")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_dot_data, 
                 self.ErrorPlotting.x2_dot_data,
                 self.ErrorPlotting.dx_b_dot_data, 
                 "Body Velocity.")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_dot_data, 
                 self.ErrorPlotting.x3_dot_data, 
                 self.ErrorPlotting.dx_sw_dot_data, 
                 "Swing_Velocity .")
        
        # plot the command output 
        # self.ErrorPlotting.plot_api_value()

