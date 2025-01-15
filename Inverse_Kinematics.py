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

class InverseKinematic(ForwardKinematic):
    """
    Inverse Kinematic Controller for Unitree Robots using MuJoCo simulation.
    This class extends Kinematics and controls robot joints based on trajectory planning.
    """

    def __init__(self):
        super().__init__(config.ROBOT_SCENE)
        mujoco.mj_step(self.model, self.data)
        self.step_size = config.SIMULATE_DT
        self.swing_time =  0.25 # 0.25 # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing


        # Robot parameters
        self.body_height = 0.25
        self.swing_height = 0.075
        self.velocity = 0.0

        # Initialize the leg positions
        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
        
        # Initialize the API
        self.cmd = send_motor_commands()
        
        # intialize the data storage for plotting
        self.ErrorPlotting = ErrorPlotting()

        # intial the API gain 
        self.kp = 250 + 50 + 100
        self.kd = 10 
        # intial the gain for the body and swing leg
        self.kc = 1
        self.kb = 1 
        self.ks = 1


    # TODO: Get Initialize the robot state, joint angles, and leg positions.
    def initialize(self):
        """
        Initialize the robot state and joint angles.
        """
        print("Initializing inverse kinematic parameter...")
        self.initial_joint_angles = self.joint_state_reader.joint_angles.copy()

        # Get initial body and leg positions
        self.initial_body_state = self.get_body_state(self.body_frame_name).copy() # 
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

        # intialize output for Inverse Dynamics shape (12, 1)
        self.qd = np.zeros((12, 1))   
        self.dqd = np.zeros((12, 1))
        self.ddqd = np.zeros((12, 1))
        self.q = np.zeros((12, 1))
        self.dq = np.zeros((12, 1))
        self.tau = np.zeros((12,1))

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

    def compute_desired_body_state(self):
        """
        Generate trajectory for the body over the swing phase.
        """
        # print("Computing desired body state...")
        
        self.desired_body_configuration[0] += self.velocity * self.step_size
  
        return self.desired_body_configuration

    def compute_desired_swing_leg_trajectory(self):
        """
        Generate trajectories for the swing legs over the swing phase.
        """
        # print("Computing desired swing leg trajectories...")
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

    def InverseKinematic_formulation(self,J1, J2, J3, x1, x2, x3, kb, kc, ks, x_b, x_sw, i, joint_angles):

        config =  (self.initial_joint_angles - joint_angles)            # shape (12, 1)
        q_err = kc * np.hstack((np.zeros(6), config)).reshape(-1, 1)    # shape (18, 1)
        m = self.model.nv
        Jc = J1
        Nc = np.eye(m) - np.linalg.pinv(Jc, rcond=1e-5) @ Jc

        Jb_c = J2 @ Nc
        Nb_c = Nc - np.linalg.pinv(Jb_c, rcond=1e-5) @ Jb_c

        Jsw_bc = J3 @ Nb_c        
        Nsw_bc = Nb_c - np.linalg.pinv(Jsw_bc, rcond=1e-5) @ Jsw_bc

        self.dx_b = kb * (x_b - x2).reshape(-1, 1)  # desired body state error - dotx_b
        self.dx_sw = ks * (x_sw[i].T - x3).reshape(-1, 1) # desired swing leg state error - dotx_sw
        q1_dot = np.linalg.pinv(Jb_c, rcond=1e-4) @ self.dx_b  # body q_dot
        q2_dot = np.linalg.pinv(Jsw_bc, rcond=1e-4) @ (self.dx_sw - J3 @ q1_dot) # swing leg q_dot
        q3_dot = Nsw_bc @ (q_err - q1_dot - q2_dot)

        q_dot = q1_dot + q2_dot + q3_dot            # update dqd
        
        dq_cmd = q_dot[6:].flatten()                # shape (12, 1)
        new_joint_angles = joint_angles + dq_cmd    # update qd

    

        self.ErrorPlotting.xb_data.append(x_b.T)
        self.ErrorPlotting.x2_data.append(x2.flatten())
        self.ErrorPlotting.dx_b_data.append(self.dx_b.flatten())

        self.ErrorPlotting.xw_data.append(x_sw[i].T)
        self.ErrorPlotting.x3_data.append(x3.flatten())
        self.ErrorPlotting.dx_sw_data.append(self.dx_sw.flatten())         

        self.ErrorPlotting.q3_dot_data.append(self.change_q_order(q3_dot[6:].flatten()))
        self.ErrorPlotting.q2_dot_data.append(self.change_q_order(q2_dot[6:].flatten()))
        self.ErrorPlotting.q1_dot_data.append(self.change_q_order(q1_dot[6:].flatten()))
        return new_joint_angles, dq_cmd #Kinematic order 

    def calculate(self):
        """
        Main loop to compute and update joint angles in real-time,
        including a trot gait with proper leg phasing.
        """


        # Initialize the loop variables
        i = 0
        trail = 0
        running_time = 0
        
        print("Starting Trot Gait...")

        x_sw = self.compute_desired_swing_leg_trajectory()
        leg_pair_in_swing = True
        
        while True:
            i = (i + 1) % self.K  # Loop over the swing cycle duration
 
            if i == 0:
                # over one cycle
                self.transition_legs()
                x_sw = self.compute_desired_swing_leg_trajectory()
                # print("Transitioning legs...")
            
            x_b = self.compute_desired_body_state()             # update the body state for the next cycle
            joint_angles = self.joint_state_reader.joint_angles # get the current joint angles
            
           
            self.required_state = self.get_required_state()          # get the required state for IK
            self.x1 = self.required_state["contact_leg"]
            self.x2 = self.required_state["body"]
            self.x3 = self.required_state["swing_leg"]
            # x1 : contact leg positions, x2 : body state, x3 : swing leg positions

            self.required_jacobian = self.get_required_jacobian()
            self.J1 = self.required_jacobian["contact_leg"]
            self.J2 = self.required_jacobian["body"]
            self.J3 = self.required_jacobian["swing_leg"]
            # J1 : contact leg Jacobian, J2 : body Jacobian, J3 : swing leg Jacobian

            # calculate the desired joint angles and joint velocities
            self.qd, self.dqd = self.InverseKinematic_formulation(self.J1, self.J2, self.J3, 
                                                                self.x1, self.x2, self.x3, 
                                                                self.kb, self.kc, self.ks, 
                                                                x_b, x_sw, 
                                                                i, joint_angles)
            # calculate ctrl output through PD controller 
            dq_error = self.kp * (self.change_q_order(self.qd) - self.data.sensordata[:12])
            dq_dot = self.kd * (self.change_q_order(self.dqd) - self.data.sensordata[12:24])
            
            
            self.ErrorPlotting.dq_error_data.append(dq_error)
            self.ErrorPlotting.dq_dot_data.append(dq_dot)
            self.ErrorPlotting.output_data.append(dq_error + dq_dot)
            # TODO update the output for the Inverse Dynamics
            self.ddqd = dq_error + dq_dot # desired joint acceleration
            self.q = self.change_q_order(self.data.sensordata[:12]) # current joint angles in kinematic order
            self.dq = self.change_q_order(self.data.sensordata[12:24]) # current joint velocities in kinematic order

            # TODO Feed the desired joint angles and joint velocities to the Inverse Dynamics
            # self.InverseDynamic_formulation(self.model, self.data, self.q, self.dq, self.ddqd, self.qd, self.dqd)
            # tau = InverseDynamic.compute_torque(self.model, self.data, self.q, self.dq, self.ddqd, self.qd, self.dqd)
            
            # send qd and dqd to the API 
            self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd))
            # print(self.ddqd , self.tau, self.data.ctrl[:])
            self.data.ctrl[:] = self.ddqd + self.tau.T
            # data storage for plotting
            self.ErrorPlotting.q_desired_data.append(self.change_q_order(self.qd))
            self.ErrorPlotting.q_current_data.append(self.change_q_order(self.q ))
            # update running steps 
            trail += 1
            if trail > 5000: # 5000 steps 
                break
        # call the error plotting class for plotting the data
        self.ErrorPlotting.plot_api_value(self.ErrorPlotting.dq_error_data, self.ErrorPlotting.dq_dot_data, self.ErrorPlotting.output_data)
        self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q3_dot_data, "q3_dot")
        self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q2_dot_data, "q2_dot")
        self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q1_dot_data , "q1_dot")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_data, self.ErrorPlotting.x2_data, self.ErrorPlotting.dx_b_data, "Body")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_data, self.ErrorPlotting.x3_data, self.ErrorPlotting.dx_sw_data, "Swing")
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.q_desired_data, self.ErrorPlotting.q_current_data) 
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
    ChannelFactoryInitialize(1, "lo")
    # cmd = send_motor_commands()
    ik = InverseKinematic()
    ik.start_joint_updates()
    ik.cmd.move_to_initial_position()
    ik.initialize()
    ik.calculate()
