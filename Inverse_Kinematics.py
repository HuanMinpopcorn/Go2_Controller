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
from matplotlib.animation import FuncAnimation


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
      
        return np.array(body_trajectory), np.array(swing_leg_trajectory)

    def cubic_spline(self, t, tf, xf):
        """
        Generate cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3

    import matplotlib.pyplot as plt

    def calculate(self):
        """
        Main loop to compute and update joint angles in real-time,
        including a trot gait with proper leg phasing.
        """
        # Initialize the command publishe3.5
        x_b, x_sw = self.compute_desired_value()

        # intial the API gain 

        kp = 50
        kd = 3.5
        # Gain for joint position error
        kb = 1

        m = self.model.nv
        i = 0
        trail = 0
        running_time = 0
        leg_pair_in_swing = True

        
        q_desired_data = []
        q_current_data = []
        q_err_data = []
        q3_dot_data = []




        # Data storage for plotting
        dx_sw_data = []
        q1_dot_data = []
        q2_dot_data = []
        q3_dot_data = []
        dq_cmd_data = []

        dq_error_data = []
        dq_dot_data = []

        x3_data = []

        print("Starting Trot Gait...")
        intial_joint_angles = self.joint_state_reader.joint_angles.copy()
        while True:
            i = (i + 1) % self.K  # Loop over the swing cycle duration
            # print("i", i)  
            running_time += self.step_size
            phase = np.tanh(running_time / 1.2)
            # kp = 50 * phase + (1 - phase) * 20 # Gradual stiffness 
            kp = 69 # Gradual stiffness
            kd =  3.5 # Gradual damping
          
            # if i == 0:
            #     leg_pair_in_swing = not leg_pair_in_swing
            #     self.transition_legs()
            #     x_b, x_sw = self.compute_desired_value()

            # if leg_pair_in_swing:
            #     active_swing_legs = self.swing_legs
            #     active_stance_legs = self.contact_legs
            # else:
            #     active_swing_legs = self.contact_legs
            #     active_stance_legs = self.swing_legs

            joint_angles = self.joint_state_reader.joint_angles
            x1, x2, x3 = self.get_required_state() 
            # x1 : contact leg positions, x2 : body state, x3 : swing leg positions
            J1, J2, J3 = self.get_required_jacobian()
            # J1 : contact leg Jacobian, J2 : body Jacobian, J3 : swing leg Jacobian

            stand_up_joint_pos = np.array([
                0.052, 1.12, -2.10, -0.052, 1.12, -2.10,
                0.052, 1.12, -2.10, -0.052, 1.12, -2.10
            ], dtype=float)

            config = (self.change_q_order(stand_up_joint_pos) - joint_angles) # shape (12, 1)
           
            q_err = np.hstack((np.zeros(6), config)).reshape(-1, 1) # shape (18, 1)
            

            dx_b = kb * (x_b[i].T - x2).reshape(-1, 1)
            dx_sw = (x_sw[i].T - x3).reshape(-1, 1)
            
        
            N1 = np.eye(m) - np.linalg.pinv(J1, rcond=1e-4) @ J1      
            J_21 = J2 @ N1
            N_21 = np.eye(m) - np.linalg.pinv(J_21, rcond=1e-4) @ J_21      

            q1_dot = np.linalg.pinv(J_21, rcond=1e-4) @ dx_b
            q2_dot = np.linalg.pinv(J3 @ N_21, rcond=1e-4) @ (dx_sw - J3 @ q1_dot)
            N_321 = np.eye(m) - np.linalg.pinv(J3 @ N_21, rcond=1e-4) @ J3 @ N_21
            q3_dot = N_321 @ q_err
            q_dot = q3_dot + q1_dot
            # q_dot = q1_dot + q2_dot 
            dq_cmd = q_dot[6:].flatten()
   
            
            # Append more data for plotting
            # q1_dot_data.append(q1_dot.copy())
            # q2_dot_data.append(q2_dot.copy())
            # q3_dot_data.append(q3_dot.copy())
            # dq_cmd_data.append(dq_cmd.copy())
            # Append data for plotting
            # dx_sw_data.append(dx_sw.copy())
            # x3_data.append(x3.copy())
            q_desired_data.append(self.change_q_order(stand_up_joint_pos))
            q_current_data.append(joint_angles.copy())
            q_err_data.append(q_err[6:].copy())
            q3_dot_data.append(q3_dot[6:].T.copy())

            
            
            new_joint_angles = joint_angles + dq_cmd 
            # new_joint_angles = self.check_joint_limits(new_joint_angles)
            # gravity_torque = np.array(self.data.qfrc_bias[6:]).flatten()

            # cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles) * phase + (1-phase) * self.change_q_order(intial_joint_angles), self.change_q_order(dq_cmd))
            cmd.send_motor_commands(kp, kd, self.change_q_order(new_joint_angles), self.change_q_order(q_dot))
            dq_error = kp * ((new_joint_angles) - self.change_q_order(self.data.sensordata[:12]))
            dq_error_data.append(dq_error)
            dq_dot= kd * ((dq_cmd) - self.change_q_order(self.data.sensordata[12:24]))
            dq_dot_data.append(dq_dot)

            # print("dq error" ,dq_error )
            # print("dq error" , kp * ((new_joint_angles) - self.data.sensordata[:12]))
            # print("---------------------------------------------------")
            # print("dq_dot: ", kd * ((dq_cmd) - self.change_q_order(self.data.sensordata[12:24])))
            # cmd.move_to_initial_position(self.change_q_order(joint_angles), self.change_q_order(new_joint_angles))

            # Additional debug prints can stay here if needed

            # Break after a few iterations for demonstration purposes (or add your condition)
            # print("new_joint_angles", new_joint_angles) 
            # print(self.change_q_order(dq_cmd) * kd)
            trail += 1
            # print("Trail: ", trail)
            if trail > 2000:  # Replace or remove as needed
                break
        # #plot the error data
        # print(len(q_desired_data))
        # print(len(q_current_data))
        # print(len(q_err_data))
        # print(len(q3_dot_data))
        self.plot_api_value(dq_error_data, dq_dot_data)
        self.plot_q_error(q_desired_data, q_current_data, q_err_data, q3_dot_data) # all shape is (N, 12)


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
    def plot_state_trajectories(self, body_trajectory, swing_leg_trajectory):
        """
        Plot the body and swing leg trajectories.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(body_trajectory[:, 0], label="X")
        plt.plot(body_trajectory[:, 1], label="Y")
        plt.plot(body_trajectory[:, 2], label="Z")
        plt.title("Body Trajectory")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(swing_leg_trajectory[:, 0], label="X")
        plt.plot(swing_leg_trajectory[:, 1], label="Y")
        plt.plot(swing_leg_trajectory[:, 2], label="Z")
        plt.title("Front Swing Leg Trajectory")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(swing_leg_trajectory[:, 3], label="X")
        plt.plot(swing_leg_trajectory[:, 4], label="Y")
        plt.plot(swing_leg_trajectory[:, 5], label="Z")
        plt.title("Rear Swing Leg Trajectory")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_q_error(self, q_desired, q_actual, q_error, q_actuated):
        """
        Plot the joint angles, desired joint angles, joint position error, and actuated joint angles.

        Parameters:
        q_desired (list): Desired joint angles, shape (N, num_joints).
        q_actual (list): Actual joint angles, shape (N, num_joints).
        q_error (list): Joint position errors, shape (N, num_joints).
        q_actuated (list): Actuated joint angles, shape (N, num_joints).
        """
        # # Plot dx_sw data for all axes
        # plt.figure()
        # for axis in range(dx_sw_data[0].shape[0]):
        #     plt.plot([data[axis, 0] for data in dx_sw_data], label=f'dx_sw[{axis}]')
        # plt.title('dx_sw Over Time')
        # plt.xlabel('Iteration')
        # plt.ylabel('dx_sw')
        # plt.legend()

        # plt.figure()
        # for axis in range(6):
        #     plt.plot([data[axis] for data in x3_data], label=f'x2[{axis}]')
        # plt.title('x2 Over Time')
        # plt.xlabel('Iteration')
        # plt.ylabel('x2')    
        # plt.legend()


        # Plot q2_dot data for the last twelve joints
        plt.figure()
        plt.subplot(4, 1, 1)
        for joint in range(3):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{joint}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{joint}]' , linestyle='-')
            # plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 2)
        for joint in range(3, 6):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{joint}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{joint}]' , linestyle='-')
            # plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 3)
        for joint in range(6, 9):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{joint}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{joint}]' , linestyle='-')
            # plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 4)
        for joint in range(9, 12):
            plt.plot([qd[joint] for qd in q_desired], label=f'q_desired[{joint}]', linestyle='--')
            plt.plot([qa[joint] for qa in q_actual], label=f'q_actual[{joint}]' , linestyle='-')
            # plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_config Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.figure()
        plt.subplot(4, 1, 1)
        for joint in range(3):
 
            plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_error Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 2)
        for joint in range(3, 6):

            plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_error Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 3)
        for joint in range(6, 9):
        
            plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_error Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()

        plt.subplot(4, 1, 4)
        for joint in range(9, 12):
            plt.plot([qe[joint] for qe in q_error], label=f'q_error[{joint}]', linestyle='-.')
        plt.title('q_error Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('q')
        plt.legend()
        plt.show()

    def plot_api_value(self, dq_error, dq_dot):
        plt.figure()
        plt.subplot(2, 1, 1)
        for joint in range(12):
            plt.plot([data[joint] for data in dq_error], label=f'dq_error[{joint}]')
        plt.title('dq_error Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('dq_error')

        plt.subplot(2, 1, 2)
        for joint in range(12):
            plt.plot([data[joint] for data in dq_dot], label=f'dq_dot[{joint}]')
        plt.title('dq_dot Over Time')
        plt.legend()
        plt.show()

# Example Usage
if __name__ == "__main__":


    robot_scene = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    cmd = send_motor_commands()
    ik = InverseKinematic(robot_scene, cmd)
    ik.start_joint_updates()
    cmd.move_to_initial_position()
    # ik.compute_desired_value()
    ik.calculate()