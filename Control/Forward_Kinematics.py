import mujoco
import numpy as np
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from Interface.read_JointState import read_JointState
from Interface.read_TaskSpace import read_TaskSpace
from Simulation import config

from Interface.Send_motor_cmd import send_motor_commands

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
        self.lock = threading.Lock()
        self.start_joint_updates()

    def set_joint_angles(self):
        """
        feed the topic's info to mujoco calculation 
        offline from the mujoco simulation 
        """
        self.lock.acquire()
        # TODO: Set the joint angles and robot state in the MuJoCo data structure
        self.data.qpos[:3] = self.robot_position
        self.data.qpos[3:7] = self.body_quat
        self.data.qpos[7:19] = self.joint_angles

        self.data.qvel[:3] = self.robot_velocity
        self.data.qvel[3:6] = self.body_angular_velocity
        self.data.qvel[6:19] = self.joint_velocity
        self.lock.release()

    def get_body_state(self, body_identifier):
        """
        Retrieves the position and orientation of the specified body in global frame.

        Parameters:
            body_name (str): Name of the body to retrieve the state for.

        Returns:
            dict: Contains 3D position and 3x3 orientation matrix of the body.
        """
        # Get the body state (position and orientation) from MuJoCo data
               # Determine if the identifier is a name or an ID
        if isinstance(body_identifier, str):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_identifier)
        else:
            body_id = body_identifier
        position = np.copy(self.data.xpos[body_id])
        orientation_quat = np.copy(self.data.xquat[body_id])
        orientation = self.convert_quat_to_euler(orientation_quat)
        return {"position": position, "orientation": orientation}

    

    def get_jacobian(self, body_identifier):
        """
        Computes the Jacobian matrix for a given body.

        Parameters:
            body_identifier (str or int): Name or ID of the body to compute the Jacobian for.

        Returns:
            dict: Contains the translational and rotational Jacobians.
        """
        # Determine if the identifier is a name or an ID
        if isinstance(body_identifier, str):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_identifier)
        else:
            body_id = body_identifier

        J_pos = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, body_id)
        
        if self.check_Jacobian_singularity(J_pos) or self.check_Jacobian_singularity(J_rot) and body_id != 0:
            print(f"Jacobian for body ID {body_id} is singular.")

        return {"J_pos": J_pos, "J_rot": J_rot}
    def get_jacobian_dot(self, body_identifier):
        """
        Computes the Jacobian matrix for a given body.

        Parameters:
            body_name (str): Name of the body to compute the Jacobian for.

        Returns:
            dict: Contains the translational and rotational Jacobians.
        """
         # Determine if the identifier is a name or an ID
        if isinstance(body_identifier, str):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_identifier)
        else:
            body_id = body_identifier

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
    def _get_contact_Jacobian(self, ncon):
        """
        Computes the Jacobian matrix for the contact points.

        Parameters:
            ncon (int): Number of contacts.

        Returns:
            np.ndarray: Contact Jacobian matrix.
        """
        # Get contact Jacobians
        J_c = np.zeros((3 * ncon, self.model.nv))  # multiple stack contact Jacobian
        for i in range(ncon):
            contact = self.data.contact[i]
            geom1_body_id = self.model.geom_bodyid[contact.geom1]
            geom2_body_id = self.model.geom_bodyid[contact.geom2]

            # get the body Jacobian
            if geom1_body_id != 0:
                J_c_single = self.get_jacobian(geom1_body_id)["J_pos"]
            else:
                J_c_single = self.get_jacobian(geom2_body_id)["J_pos"]

            J_c[3 * i:3 * (i + 1), :] = J_c_single
            # print(f"Contact ID: {i}, Geom1 Body ID: {geom1_body_id}, Geom2 Body ID: {geom2_body_id}")
        # print(f"Contact Jacobian for contact {ncon}:\n{J_c.shape}")

        return J_c
    
    def _get_contact_Jacobian_dot(self, ncon):
        """
        Computes the Jacobian matrix for the contact points.

        Parameters:
            ncon (int): Number of contacts.

        Returns:
            np.ndarray: Contact Jacobian matrix.
        """
        # Get contact Jacobians
        J_c_dot = np.zeros((3 * ncon, self.model.nv))
        for i in range(ncon):
            contact = self.data.contact[i]
            geom1_body_id = self.model.geom_bodyid[contact.geom1]
            geom2_body_id = self.model.geom_bodyid[contact.geom2]
            # get the body Jacobian
            if geom1_body_id != 0:
                J_c_dot_single = self.get_jacobian_dot(geom1_body_id)["Jp_dot"]
            else:
                J_c_dot_single = self.get_jacobian_dot(geom2_body_id)["Jp_dot"]
            J_c_dot[3 * i:3 * (i + 1), :] = J_c_dot_single
        # print(f"Contact Jacobian dot for contact {ncon}:\n{J_c_dot.shape}")
        return J_c_dot


    def update_joint_angles(self):
        """
        Continuously updates the joint angles in the background.
        """
        # TODO: Continuously update joint angles and run forward kinematics
        while self.running:
            
            self.lock.acquire()
            # update the sensor data from unitree sdk
            self.robot_position = self.task_space_reader.robot_state.position
            self.robot_velocity = self.task_space_reader.robot_state.velocity
            
            # update the joint data from unitree sdk
            self.joint_angles = self.joint_state_reader.joint_angles
            self.joint_velocity = self.joint_state_reader.joint_velocity
            self.joint_toque = self.joint_state_reader.joint_tau
            
            # update the imu data from unitree sdk
            self.body_quat = self.joint_state_reader.imu_data
            self.body_angular_velocity = self.joint_state_reader.imu_gyroscope
            self.body_acc = self.joint_state_reader.imu_accelerometer
            

            # update the contact location and jacobian
            
            self.ncon = self.data.ncon
            self.Jc = self._get_contact_Jacobian(self.ncon)
            self.Jc_dot = self._get_contact_Jacobian_dot(self.ncon)
            # print(f"Contact Jacobian for contact {self.ncon}:\n{self.Jc.shape}")
            if self.Jc.shape[0] != 3 * self.ncon or self.Jc_dot.shape[0] != 3 * self.ncon:
                raise ValueError(f"Shape mismatch: Jc shape {self.Jc.shape[0]}, Jc_dot shape {self.Jc_dot.shape[0]}, expected {3 * self.ncon}")
            self.lock.release() 

            # Set joint angles
            self.set_joint_angles()
            # Run forward kinematics
            mujoco.mj_forward(self.model, self.data)
            # Run Inverse Dynamics
            mujoco.mj_inverse(self.model, self.data)
            # mujoco.mj_comPos(self.model, self.data) # Map inertias and motion dofs to global frame centered at CoM.
            # mujoco.mj_crb(self.model, self.data)# Run composite rigid body inertia algorithm (CRB).
            # mujoco.mj_comVel(self.model, self.data)
            # self.print_joint_data()
            
            mujoco.mj_collision(self.model, self.data)
            # print("collision",self.data.ncon)
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
    cmd = send_motor_commands()
    cmd.move_to_initial_position()
    # time.sleep(1.0)
    # print(fk.print_joint_data())
    # time.sleep(1.0)
