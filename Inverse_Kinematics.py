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

    def __init__(self):
        super().__init__(config.ROBOT_SCENE)
        mujoco.mj_step(self.model, self.data)
        # running the FK
        # self.start_joint_updates()
        self.step_size = config.SIMULATE_DT
        self.swing_time =  0.25 # 0.25 # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing

        self.num_actuated_joints = self.model.nv - 6

        # Robot parameters
        self.body_height = 0.225 
        self.swing_height = 0.075
        self.swing_height = 0.0
        self.velocity = 0  # Forward velocity

        # Initialize the leg positions
        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
        # self.swing_legs = ["RR_foot"]
        # self.contact_legs = ["FL_foot", "FR_foot", "RL_foot"]
        # Initialize the API
        self.cmd = send_motor_commands()
        self.ErrorPlotting = ErrorPlotting()

        # intial the API gain 
        self.kp = 400 
        self.kd = 10
        # intial the gain for the body and swing leg
        self.kc = 1
        self.kb = 1
        self.ks = 1

        # intialize output for Inverse Dynamics shape (12, 1)
        self.qd = np.zeros((12, 1))   
        self.dqd = np.zeros((12, 1))
        self.ddqd = np.zeros((12, 1))
        self.ddq_cmd = np.zeros((12, 1))
 
        self.tau = np.zeros((12,1))
        

    # TODO: Get Initialize the robot state, joint angles, and leg positions.
    def initialize(self):
        """
        Initialize the robot state and joint angles.
        """
        # print("Initializing inverse kinematic parameter...")
        self.initial_joint_angles = self.joint_angles.copy()
        self.initial_joint_velocity = self.joint_velocity.copy()
    
        # print("Initial joint angles:", self.data.qpos)
    


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
    # TODO: Check the joint limits and ensure the joint angles are within the limits.
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

    def compute_desired_body_state(self):
        """
        Generate trajectory for the body over the swing phase.
        """
        # print("Computing desired body state...")
        body_moving_trajectory = []
        initial_body_configuration = self.get_required_state()["body"].copy()
    
        
        desired_body_configuration = np.copy(initial_body_configuration)
        for i in range(self.K):
            t = i / self.K  # Normalized time step
            desired_body_configuration[0] = initial_body_configuration[0] +  self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            desired_body_configuration[1] = initial_body_configuration[1] 
            desired_body_configuration[2] = self.body_height # Set desired body height
            # desired_body_configuration[2] = self.body_height # Set desired body height
        
            body_moving_trajectory.append(desired_body_configuration.copy())
        
        return np.array(body_moving_trajectory)

    def compute_desired_swing_leg_trajectory(self):
        """
        Generate trajectories for the swing legs over the swing phase.
        """
        # print("Computing desired swing leg trajectories...")
        swing_leg_trajectory = []
        swing_leg_positions_initial = self.get_required_state()["swing_leg"].copy()
        swing_leg_positions = np.copy(swing_leg_positions_initial)
        
        for i in range(self.K):
            for leg_index in range(len(self.swing_legs)):
                swing_leg_positions[3 * leg_index + 0] = swing_leg_positions_initial[3 * leg_index + 0] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
                swing_leg_positions[3 * leg_index + 1] = swing_leg_positions_initial[3 * leg_index + 1]  # Keep Y position constant
                swing_leg_positions[3 * leg_index + 2] = swing_leg_positions_initial[3 * leg_index + 2] + self.swing_height * np.sin(np.pi * i / self.K)

            swing_leg_trajectory.append(swing_leg_positions.copy())

        return np.array(swing_leg_trajectory)
        
    def compute_desired_body_state_velocity_trajectory(self):
        body_velocity_trajectory = []
        body_jacobian_dot = np.vstack((self.get_jacobian_dot(self.body_frame_name)["Jp_dot"], self.get_jacobian_dot(self.body_frame_name)["Jr_dot"]))
        initial_body_velocity = body_jacobian_dot @ self.data.qvel.copy().reshape(-1, 1)
        desired_body_velocity = np.copy(initial_body_velocity)
        for i in range(self.K):
            desired_body_velocity[0] = initial_body_velocity[0] + self.diff_cubic_spline(i, self.K, self.velocity * self.swing_time)
            desired_body_velocity[1] = initial_body_velocity[1]
            desired_body_velocity[2] = initial_body_velocity[2]
            body_velocity_trajectory.append(desired_body_velocity.copy())
        return np.array(body_velocity_trajectory)

    def compute_desired_swing_leg_velocity_trajectory(self):
        swing_leg_velocity_trajectory = []
        swing_leg_velocity_initial = np.zeros((len(self.swing_legs) * 3, 1))
        swing_leg_velocity = np.copy(swing_leg_velocity_initial)
        swing_leg_velocity_initial = np.vstack([self.get_jacobian(leg)["J_pos"] @ self.data.qvel.copy().reshape(-1, 1) for leg in self.swing_legs])
        for i in range(self.K):
            for leg_index in range(len(self.swing_legs)):
                swing_leg_velocity[3 * leg_index + 0] = swing_leg_velocity_initial[3 * leg_index + 0] + self.diff_cubic_spline(i, self.K, self.velocity * self.swing_time)
                swing_leg_velocity[3 * leg_index + 1] = swing_leg_velocity_initial[3 * leg_index + 1]
                swing_leg_velocity[3 * leg_index + 2] = swing_leg_velocity_initial[3 * leg_index + 2] + self.swing_height * np.pi / self.K * np.cos(np.pi * i / self.K)
            swing_leg_velocity_trajectory.append(swing_leg_velocity.copy())
        return np.array(swing_leg_velocity_trajectory)
    
    def cubic_spline(self, t, tf, xf):
        """
        Generate cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3
    def diff_cubic_spline(self, t, tf, xf):
        """
        Generate cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return 2 * a2 * t + 3 * a3 * t**2

    def InverseKinematic_formulation(self,J1, J2, J3, x1, x2, x3, kb, kc, ks, x_b, x_sw, i, joint_angles):

        config =  (self.initial_joint_angles - joint_angles)            # shape (12, 1)
        q_err = kc * np.hstack((np.zeros(6), config)).reshape(-1, 1)    # shape (18, 1)

        m = self.model.nv
        Jc = J1
        Nc = np.eye(m) - np.linalg.pinv(Jc, rcond=1e-6) @ Jc

        Jb_c = J2 @ Nc
        Nb_c = Nc - np.linalg.pinv(Jb_c, rcond=1e-6) @ Jb_c

        Jsw_bc = J3 @ Nb_c        
        Nsw_bc = Nb_c - np.linalg.pinv(Jsw_bc, rcond=1e-6) @ Jsw_bc
        # print("Body  shape:", x_b[i].T.shape, "Swing leg  shape:", x_sw[i].T.shape)
        dx_b = kb * (x_b[i].T - x2).reshape(-1, 1)  # desired body state error - dotx_b
        dx_sw = ks * (x_sw[i].T - x3).reshape(-1, 1) # desired swing leg state error - dotx_sw

        q2_dot = np.linalg.pinv(Jb_c, rcond=1e-6) @ dx_b  # body q_dot
        q3_dot = np.linalg.pinv(Jsw_bc, rcond=1e-6) @ (dx_sw - J3 @ q2_dot) # swing leg q_dot
        q1_dot = Nsw_bc @ (q_err - q2_dot - q3_dot)

        q_dot = q1_dot + q2_dot + q3_dot            # update dqd
        # q_dot = q1_dot + q2_dot 
        
        dq_cmd = q_dot[6:].flatten()                # shape (12, 1)
        new_joint_angles = joint_angles + dq_cmd    # update qd
   

        self.ErrorPlotting.xb_data.append(x_b[i].T)
        self.ErrorPlotting.x2_data.append(x2.flatten())
        self.ErrorPlotting.dx_b_data.append(dx_b.flatten())

        self.ErrorPlotting.xw_data.append(x_sw[i].T)
        self.ErrorPlotting.x3_data.append(x3.flatten())
        self.ErrorPlotting.dx_sw_data.append(dx_sw.flatten())         

        self.ErrorPlotting.q3_dot_data.append(self.change_q_order(q3_dot[6:].flatten()))
        self.ErrorPlotting.q2_dot_data.append(self.change_q_order(q2_dot[6:].flatten()))
        self.ErrorPlotting.q1_dot_data.append(self.change_q_order(q1_dot[6:].flatten()))
        return new_joint_angles, dq_cmd #Kinematic order 
    
    def desired_joint_acceleration(self, J1, J2, J3, J1_dot, J2_dot, J3_dot, x1, x2, x3, i, x_b_dot, x_sw_dot, joint_angles_velcity):
        """
        Compute desired joint acceleration using KinWBC null-space projections
        Returns:
            ddqd: Desired joint acceleration (12x1 vector)
        """
        # Initialize variables
        ddqd_desired = np.zeros((self.model.nv, 1))  # nv = number of degrees of freedom

        vel_error =  (self.initial_joint_velocity - joint_angles_velcity)            # shape (12, 1)
        dq_err = np.hstack((np.zeros(6), vel_error)).reshape(-1, 1)
      

        Jc = J1
        Nc = np.eye(self.model.nv) - np.linalg.pinv(Jc, rcond=1e-5) @ Jc

        Jb_c = J2 @ Nc
        Nb_c = Nc - np.linalg.pinv(Jb_c, rcond=1e-5) @ Jb_c

        Jsw_bc = J3 @ Nb_c        
        Nsw_bc = Nb_c - np.linalg.pinv(Jsw_bc, rcond=1e-5) @ Jsw_bc

        dx_b_dot = x_b_dot[i] - J2_dot @ self.data.qvel.copy().reshape(-1, 1)  # Body velocity error
        dx_sw_dot = x_sw_dot[i] - J3_dot @ self.data.qvel.copy().reshape(-1, 1)  # Swing leg velocity error
        # Compute desired joint acceleration
        q1_ddot = np.linalg.pinv(Jb_c, rcond=1e-4) @ (dx_b_dot - J2_dot @ self.data.qvel.copy().reshape(-1, 1) )  # Body joint acceleration
        q2_ddot = np.linalg.pinv(Jsw_bc, rcond=1e-4) @ (dx_sw_dot - J3_dot @ self.data.qvel.copy().reshape(-1, 1))  # Swing leg joint acceleration
        q3_ddot = Nsw_bc @ (dq_err - q1_ddot - q2_ddot)  # Joint acceleration for the remaining joints
        # print("q1_ddot", q1_ddot.shape)
        # print("q2_ddot", q2_ddot.shape)
        ddqd_desired = q1_ddot + q2_ddot + q3_ddot
        # ddqd_desired = q1_ddot + q2_ddot
            
            
        return ddqd_desired[6:]  # Return only actuated joints (12x1)

    def calculate(self,x_sw, x_b, x_sw_dot, x_b_dot, i):
        """
        Main loop to compute and update joint angles in real-time,
        including a trot gait with proper leg phasing.
        """


        # # Initialize the loop variables
        # i = 0
        # trail = 0

        index = i % self.K
        
        # print("compute inverse kinematics...")

        # # x_sw = self.compute_desired_swing_leg_trajectory()
        # # x_b = self.compute_desired_body_state() # update the body state for the next cycle
        # # x_sw_dot = self.compute_desired_swing_leg_velocity_trajectory()
        # # x_b_dot = self.compute_desired_body_state_velocity_trajectory()

        
        # # while not self._stop_event.is_set():  # Use event flag instead of while True

        # if index < self.K - 1:
        #     index += 1
        # else:
        #     self.transition_legs()
        #     index = 0
        #     # Smooth transition of legs
        #     self.transition_legs()
        #     self.initialize()
        #     x_sw = self.compute_desired_swing_leg_trajectory() 
        #     x_b = self.compute_desired_body_state()  # update the body state for the next cycle
        #     x_sw_dot = self.compute_desired_swing_leg_velocity_trajectory()
        #     x_b_dot = self.compute_desired_body_state_velocity_trajectory()
            # Smoothly blend the new trajectories with the current state
            # x_sw = 0.5 * (x_sw + self.x3)
            # x_b = 0.5 * (x_b + self.x2)
            # x_sw_dot = 0.5 * (x_sw_dot + self.joint_velocity[:len(x_sw_dot)])
            # x_b_dot = 0.5 * (x_b_dot + self.joint_velocity[:len(x_b_dot)])
            # # print("Transitioning legs...")
        
                    
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
        self.qd, self.dqd = self.InverseKinematic_formulation(self.J1, self.J2, self.J3, 
                                                            self.x1, self.x2, self.x3, 
                                                            self.kb, self.kc, self.ks, 
                                                            x_b, x_sw, 
                                                            index, joint_angles)
        
        self.ddqd = self.desired_joint_acceleration(self.J1, self.J2, self.J3,
                                                            self.J1_dot, self.J2_dot, self.J3_dot,
                                                            self.x1, self.x2, self.x3,
                                                            index, x_b_dot, x_sw_dot,joint_angles_velcity)

        
        dq_error = self.kp * (self.change_q_order(self.qd) - self.data.sensordata[:self.num_actuated_joints])
        dq_dot_error = self.kd * (self.change_q_order(self.dqd) - self.data.sensordata[self.num_actuated_joints:self.num_actuated_joints * 2])
        self.ddq_cmd = self.ddqd.reshape(-1, 1) + self.change_q_order(dq_error.reshape(-1, 1)) + self.change_q_order(dq_dot_error.reshape(-1, 1))  # desired joint acceleration

        # PD controller 
        # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd))

        # calculate the contact foot acceleration
        # ddxc = self.J1 @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + self.J1_dot @ self.data.qvel.copy().reshape(-1, 1)
        # self.ErrorPlotting.ddxc_data.append(ddxc.flatten())
        # # calculate the tau cmd
        # M = np.zeros((self.model.nv, self.model.nv))
        # mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        # B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        # S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        # tau_1 = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - self.data.qfrc_inverse.reshape(-1, 1))
        # Fc_list = []  # Contact forces (constraint space)
        # Jc_list = []  # Contact Jacobians (constraint space)
        # efc_J = self.data.efc_J
        # Jc = self.data.efc_J.reshape(-1, self.model.nv)  # Shape: (n_constraints, nv)
        # efc_shape = Jc.shape
        # # Loop through all active contacts

        # ncon = self.data.ncon   
        # for i in range(ncon):
        #     contact = self.data.contact
        #     start = contact.efc_address[i]  # Starting index in efc_force/efc_J
        #     if start == -1:
        #         continue
        #     dim = contact.dim[i]            # Number of constraints (e.g., 1 for normal, 3 for normal + friction)
            
        #     # Extract contact forces for this contact (e.g., normal + tangential friction)
        #     Fc_i = self.data.efc_force[start:start + dim]
            
        #     # Extract Jacobian rows for this contact (shape: [dim, nv])
        #     Jc_i = Jc[start:start + dim,:]
            
        #     Fc_list.append(Fc_i)
        #     Jc_list.append(Jc_i)

        # # Combine results across all contacts
        # if Fc_list:
        #     Fc = np.concatenate(Fc_list).reshape(-1, 1)  # (n_constraints, 1)
        #     Jc = np.vstack(Jc_list)                      # (n_constraints, nv)
        # else:
        #     Fc = np.array([]).reshape(0, 1)
        #     Jc = np.empty((0, self.model.nv))

        # Fc_shape = Fc.shape
        # Jc_shape =  Jc.shape
        # toq_shape = (Jc.T @ Fc).shape
        # # print("contact torque shape:", Jc.T @ Fc)
        
        
        
        # 
        #         # tau_2 = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - Jc.T @ Fc)
        #         # # tau_diff = tau_1 - tau_2

        #         # # self.ErrorPlotting.tau_data.append(tau_1)
        #         # self.ErrorPlotting.tau_data.append(tau_2)
        #         # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd))

        #         self.cmd.send_motor_commands(self.kp, self.kd, 
        #                                      self.change_q_order(self.qd), 
        #                                      self.change_q_order(self.dqd), 
        #                                      self.change_q_order(tau_1))
        #         # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.change_q_order(tau_2))

        #         # print(self.ddqd , self.tau, self.data.ctrl[:])
        # get the desired q qdot qddot and current q, qdot 
        self.ErrorPlotting.q_desired_data.append(self.change_q_order(self.qd))
        self.ErrorPlotting.q_current_data.append(self.joint_angles)
        self.ErrorPlotting.q_error_data.append(dq_error)

        self.ErrorPlotting.dq_desired_data.append(self.change_q_order(self.dqd))
        self.ErrorPlotting.dq_current_data.append(self.joint_velocity)
        self.ErrorPlotting.dq_error_data.append(dq_dot_error)
    
        self.ErrorPlotting.ddq_desired_data.append(self.change_q_order(self.ddqd))
        self.ErrorPlotting.ddq_current_data.append(self.change_q_order(self.data.sensordata[self.num_actuated_joints * 2 :self.num_actuated_joints * 3]))
        self.ErrorPlotting.ddq_error_data.append(self.change_q_order(self.ddqd - self.data.sensordata[self.num_actuated_joints * 3 :self.num_actuated_joints * 4]))
        
        #         # self.ErrorPlotting.tau_data.append(self.tau.flatten())
        #         #  Send `ddqd` through the pipe
        #         # print("Sending data through pipe...")
        #         # data_to_send = {
        #         #     "qd": self.qd.tolist(),
        #         #     "dqd": self.dqd.tolist(),
        #         #     "ddqd": self.ddqd.tolist(),
        #         #     "ddqd_desired": self.ddq_cmd.tolist(),
        #         #     "J_contact": self.J1.tolist(),
        #         #     "contact_legs": self.contact_legs,
        #         #     "kp": self.kp, 
        #         #     "kd": self.kd,
        #         #     "timestamp": time.time()
        #         # }
        #         # if self.running:
        #         # # Send ddqd through pipe
        #         #     if self.conn.poll():
        #         #         _ = self.conn.recv()  # Clear any stale messages
        #         #     self.conn.send(data_to_send)  # Convert numpy array to list
                    
        #         # time.sleep(0.001)  # Reduced sleep for better responsiveness
            
        #     except (BrokenPipeError, EOFError, KeyboardInterrupt):
        #             print("Connection closed, exiting calculation loop")
        #             plt.close()
        #             break
    
            
            # trail += 1
            # for _ in tqdm(range(5000), desc="Running Inverse Kinematics"):
            #     if self._stop_event.is_set():
            #         break
        # call the error plotting class for plotting the data

        # plot ddq_cmd, dq_desired, q_desired  
        # self.ErrorPlotting.plot_q_error(self.ErrorPlotting.q_desired_data, 
        #                                 self.ErrorPlotting.q_current_data,
        #                                 self.ErrorPlotting.q_error_data,
        #                                 "qd_send")
        # self.ErrorPlotting.plot_q_error(self.ErrorPlotting.dq_desired_data, 
        #                                 self.ErrorPlotting.dq_current_data,
        #                                 self.ErrorPlotting.dq_error_data,
        #                                 "dqd_send")
        # self.ErrorPlotting.plot_q_error(self.ErrorPlotting.ddq_desired_data,
        #                                 self.ErrorPlotting.ddq_current_data,
        #                                 self.ErrorPlotting.ddq_error_data,
        #                                 "ddqd_send")
        # self.ErrorPlotting.plot_contact_acceleration(self.ErrorPlotting.ddxc_data, "contact foot acceleration")

       
        # self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q3_dot_data, "q3_dot")
        # self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q2_dot_data, "q2_dot")
        # self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q1_dot_data , "q1_dot")
        # self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_data, 
        #                                                  self.ErrorPlotting.x2_data,
        #                                                  self.ErrorPlotting.dx_b_data, 
        #                                                  "Body")
        # self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_data, 
        #                                                  self.ErrorPlotting.x3_data, 
        #                                                  self.ErrorPlotting.dx_sw_data, 
        #                                                  "Swing")
        
        # self.ErrorPlotting.plot_torque(self.ErrorPlotting.tau_data,"joint toque")

        # plt.show()
        # time.sleep(20)
        # plt.close()
        # self.conn.close()  # Proper cleanup

    # def stop(self):
    #     """Signal the process to stop"""
    #     self._stop_event.set()

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
    # def ik_main(self):
    #     """
    #     Main function to run the Inverse Kinematic controller.
    #     """
    #     # self.start_joint_updates()
    #     self.cmd.move_to_initial_position()
    #     self.initialize()
   
        

# Modified main execution pattern
def ik_process():
    ChannelFactoryInitialize(1, "lo")
    ik = InverseKinematic()
    # ik.main()

if __name__ == "__main__":
    ik_process()

