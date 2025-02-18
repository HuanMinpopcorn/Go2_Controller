import mujoco
import numpy as np
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from read_JointState import read_JointState
from read_TaskSpace import read_TaskSpace
from Simulation import config

from Send_motor_cmd import send_motor_commands

class ForwardKinematic:
    def __init__(self, xml_path=config.ROBOT_SCENE):
        """
        Initializes the MuJoCo model and data.

        Parameters:
            xml_path (str): Path to the MuJoCo XML file for the robot.
        """
        # TODO: Initialize the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # TODO: Initialize joint state and task space readers
        self.joint_state_reader = read_JointState()
        self.task_space_reader = read_TaskSpace()

   
        self.update_thread = None  # Thread for continuous updates
        self.running = False  # Control flag for the thread
        # self.start_joint_updates()

        self.counter = 0

    def set_joint_angles(self):
        """
        feed the topic's info to mujoco calculation 
        offline from the mujoco simulation 
        """
        # TODO: Set the joint angles and robot state in the MuJoCo data structure
        self.data.qpos[:3] = self.robot_position
        self.data.qpos[3:7] = self.imu_data
        self.data.qpos[7:19] = self.joint_angles

        self.data.qvel[:3] = self.robot_velocity
        self.data.qvel[3:6] = self.imu_velocity
        self.data.qvel[6:19] = self.joint_velocity
        

    def get_body_state(self, body_name):
        """
        Retrieves the position and orientation of the specified body in global frame.

        Parameters:
            body_name (str): Name of the body to retrieve the state for.

        Returns:
            dict: Contains 3D position and 3x3 orientation matrix of the body.
        """
        # TODO: Get the body state (position and orientation) from MuJoCo data
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_name == "world" or body_name == "base_link":
            position = np.copy(self.data.xpos[body_id])
        else:
            position = np.copy(self.get_foot_position_in_hip_frame(body_id))
        position = np.copy(self.data.xpos[body_id])
        orientation_quat = np.copy(self.data.xquat[body_id])
        orientation = self.convert_quat_to_euler(orientation_quat)

        return {"position": position, "orientation": orientation}

    def get_jacobian(self, body_name):
        """
        Computes the Jacobian matrix for a given body.

        Parameters:
            body_name (str): Name of the body to compute the Jacobian for.

        Returns:
            dict: Contains the translational and rotational Jacobians.
        """
        # TODO: Compute the Jacobian matrix for the specified body
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        J_pos = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, body_id)
        
        if self.check_Jacobian_singularity(J_pos) or self.check_Jacobian_singularity(J_rot):
            print(f"Jacobian for {body_name} is singular.")

        return {"J_pos": J_pos, "J_rot": J_rot}
    def get_jacobian_dot(self, body_name):
        """
        Computes the Jacobian matrix for a given body.

        Parameters:
            body_name (str): Name of the body to compute the Jacobian for.

        Returns:
            dict: Contains the translational and rotational Jacobians.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        Jp_dot = np.zeros((3, self.model.nv))   
        Jr_dot = np.zeros((3, self.model.nv))

        mujoco.mj_jacDot(self.model, self.data, Jp_dot, Jr_dot, np.zeros(3),body_id)

        return {"Jp_dot": Jp_dot, "Jr_dot": Jr_dot}
    def get_foot_position_in_hip_frame(self, body_id):
        """
        Computes the position of a body in the hip frame.

        Parameters:
            body_id (int): ID of the body.

        Returns:
            np.ndarray: Position of the foot body in the hip frame.
        """
        # TODO: Compute the position of the foot in the hip frame
        hip_state = self.data.xpos[body_id - 3]
        foot_state = self.data.xpos[body_id]
        foot_position = foot_state - hip_state
        return foot_position

    def check_Jacobian_singularity(self, jacobian):
        """
        Check if the Jacobian matrix is singular.

        Parameters:
            jacobian (np.ndarray): The Jacobian matrix to check.

        Returns:
            bool: True if the Jacobian matrix is singular.
        """
        # TODO: Check for singularity in the Jacobian matrix
        u, s, vh = np.linalg.svd(jacobian)
        return np.isclose(s, 0.0).any()

    def convert_quat_to_euler(self, quat):
        """
        Convert quaternion to Euler angles.

        Parameters:
            quat (np.ndarray): Quaternion to convert.

        Returns:
            np.ndarray: Euler angles.
        """
        # TODO: Convert quaternion to Euler angles
        q0, q1, q2, q3 = quat
        sin_pitch = 2 * (q0 * q2 - q3 * q1)
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        pitch = np.arcsin(sin_pitch)
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        
        roll = np.clip(roll, -np.pi/2, np.pi/2)
        pitch = np.clip(pitch, -np.pi/2, np.pi/2)
        yaw = np.clip(yaw, -np.pi, np.pi)
        
        return np.array([roll, pitch, yaw]).round(3)

    def update_joint_angles(self):
        """
        Continuously updates the joint angles in the background.
        """
        # TODO: Continuously update joint angles and run forward kinematics
        while self.running:

            # update the sensor data from unitree sdk
            self.robot_position = self.task_space_reader.robot_state.position
            self.robot_velocity = self.task_space_reader.robot_state.velocity
        

            self.joint_angles = self.joint_state_reader.joint_angles
            self.joint_velocity = self.joint_state_reader.joint_velocity
            self.joint_toque = self.joint_state_reader.joint_tau
            
            self.imu_data = self.joint_state_reader.imu_data
            self.imu_velocity = self.joint_state_reader.imu_gyroscope
            self.imu_acc = self.joint_state_reader.imu_accelerometer

            
      
            self.set_joint_angles()
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_inverse(self.model, self.data)
            # mujoco.mj_comPos(self.model, self.data) # Map inertias and motion dofs to global frame centered at CoM.
            # mujoco.mj_crb(self.model, self.data)# Run composite rigid body inertia algorithm (CRB).
            # mujoco.mj_comVel(self.model, self.data)
            
            mujoco.mj_collision(self.model, self.data)
            
            time.sleep(config.SIMULATE_DT)

    def start_joint_updates(self):
        """Starts a background thread to continuously update joint angles."""
        # TODO: Start a background thread for updating joint angles
        print("Starting joint updates...")
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.update_joint_angles)
            self.update_thread.start()

    def stop_joint_updates(self):
        """Stops the background joint update thread."""
        # TODO: Stop the background thread for updating joint angles
        if self.running:
            self.running = False
            self.update_thread.join()

    def print_joint_data(self):
        """
        Prints the joint angles and velocities.
        """
        # TODO: Print the joint angles and velocities
        # np.set_printoptions(precision=5, suppress=True)
        print("\n=== Joint Angles ===")
        print(self.data.qpos) 
        print("\n=== Joint Velocities ===")
        print(self.data.qvel)

    def print_kinematics(self, body_name):
        """
        Prints the kinematic data of a given body.

        Parameters:
            body_name (str): Name of the body (e.g., 'FL_foot').
        """
        # TODO: Print the kinematic data (position, orientation, Jacobian) of the specified body
        state = self.get_body_state(body_name)
        jacobian = self.get_jacobian(body_name)
        jacobian_dot = self.get_jacobian_dot(body_name)
        # np.set_printoptions(precision=5, suppress=True)
        print(f"\n{body_name} Position: {state['position']}")
        print(f"{body_name} Orientation:\n{state['orientation']}")
        print(f"{body_name} Jacobian:\n{jacobian['J_pos']}")
        print(f"{body_name} Jacobian Dot:\n{jacobian_dot['Jp_dot']}")

    def print_general_framework(self):
        """
        Prints the general framework of the robot.
        """
        # TODO: Print the general framework of the robot (number of coordinates, DOF, constraints, etc.)
        # print("\n=== General Framework ===")
        
        print("===n_q number of position coordinates==")
        print(self.model.nq)
        print("===n_V number of DOF ==")
        print(self.model.nv)
        # print("===n_C number of active constraints==")
        # print(self.data.nefc)
        # print("===bias force==")
        # print(self.data.qfrc_bias)
        # print("===self.data.qfrc_actuator==")
        # print(self.data.qfrc_actuator)
        # print("===self.data.qfrc_constraint==")
        # print(self.data.qfrc_applied)
        # print("===self.data.qfrc_==")
        # print("===constraint residual ===")
        # print(self.data.efc_pos)
        # print("===constraint force ===")
        # print(self.data.efc_force)

        # print("Centroid Momentum of Inertia")
        # print("===center of mass nbody x 3 ===")
        # print(self.data.subtree_com)
        # print("===com-based motion axis of each dof  ===")
        # print(self.data.cdof)
        # print("===com-based body inertia and mass  ===")
        # print(self.data.cinert)
        # print(self.data.subtree_com.shape)
        # print(self.data.cdof.shape)
        # print(self.data.cinert.shape)
        # check the sensor data 
        print("===My model sensor data ===")
        print(self.data.sensordata[:12])
        print("===sensor joint vel ===")
        print(self.data.sensordata[12:24])
        print("===sensor joint torque ===")
        print(self.data.sensordata[24:36])
        print("===sensor imu data ===")
        print(self.data.sensordata[36:40])
        print("===sensor imu gyro ===")
        print(self.data.sensordata[40:43])
        print("===sensor imu acc ===")
        print(self.data.sensordata[43:46])
        print("===sensor frame data ===")
        print(self.data.sensordata[46:49])
        print("===sensor frame vel data ===")
        print(self.data.sensordata[49:52])

        

# Example Usage
if __name__ == "__main__":



    ChannelFactoryInitialize(1, "lo")
    fk = ForwardKinematic()
    # print(fk.print_joint_data())
    # time.sleep(1.0)
