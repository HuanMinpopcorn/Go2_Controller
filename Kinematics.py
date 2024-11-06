import mujoco
import numpy as np
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from read_JointState import read_JointState


class Kinematics:
    def __init__(self, xml_path):
        """
        Initializes the MuJoCo model and data.

        Parameters:
            xml_path (str): Path to the MuJoCo XML file for the robot.
        """
        # ChannelFactoryInitialize(1, "lo")  # Initialize communication channel
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.joint_angles = np.zeros(self.model.nq)  # Store current joint angles
        self.joint_state_reader = read_JointState()  # Initialize joint state reader
        self.update_thread = None  # Thread for continuous updates
        self.running = False  # Control flag for the thread
        
    
    def set_joint_angles(self, joint_angles):
        """
        Sets the joint angles in the MuJoCo qpos array.

        Parameters:
            joint_angles (np.ndarray): Array of joint positions to be set.
        """
        self.joint_angles = joint_angles
        self.data.qpos[:len(joint_angles)] = joint_angles

    def run_fk(self):
        """Runs forward kinematics to update positions and orientations."""
        mujoco.mj_forward(self.model, self.data)

    def get_body_state(self, body_name):
        """
        Retrieves the position and orientation of the specified body.

        Parameters:
            body_name (str): Name of the body to retrieve the state for.

        Returns:
            dict: Contains 3D position and 3x3 orientation matrix of the body.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        position = np.copy(self.data.xpos[body_id])  # 3D position
        orientation_matrix = np.copy(self.data.xmat[body_id].reshape(3, 3))  # 3x3 rotation matrix
        orientation = np.zeros(3)
        
        # orientation[1] = np.arctan2(-orientation_matrix[2, 0], np.sqrt(orientation_matrix[2, 1]**2 + orientation_matrix[2, 2]**2))  # Pitch
        # Check for singularity (gimbal lock)
        if np.isclose(orientation_matrix[2, 2], 1.0):
            orientation[1] = 0  # Pitch
            orientation[0] = np.arctan2(orientation_matrix[0, 1], orientation_matrix[0, 0])  # Roll
            orientation[2] = 0  # Yaw
            # print("Singularity detected: gimbal lock at pitch = 0")
        elif np.isclose(orientation_matrix[2, 2], -1.0):
            orientation[1] = np.pi  # Pitch
            orientation[0] = np.arctan2(orientation_matrix[0, 1], orientation_matrix[0, 0])  # Roll
            orientation[2] = 0  # Yaw
            # print("Singularity detected: gimbal lock at pitch = 180 degrees")
        else:
            orientation[1] = np.arctan2(np.sqrt(orientation_matrix[2, 0]**2 + orientation_matrix[2, 1]**2), orientation_matrix[2, 2])  # Pitch
            orientation[0] = np.arctan2(orientation_matrix[1, 2]/np.sin(orientation[1]), orientation_matrix[0, 2]/np.sin(orientation[1]))  # Roll
            orientation[2] = np.arctan2(orientation_matrix[2, 1]/np.sin(orientation[1]), -orientation_matrix[2, 0]/np.sin(orientation[1]))  # Yaw

        return {"position": position, "orientation": orientation}

    def get_jacobian(self, body_name):
        """
        Computes the Jacobian matrix for a given body.

        Parameters:
            body_name (str): Name of the body to compute the Jacobian for.

        Returns:
            dict: Contains the translational and rotational Jacobians.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        J_pos = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, body_id)
        
        if self.check_Jacobian_singularity(J_pos) or self.check_Jacobian_singularity(J_rot):
            print(f"Jacobian for {body_name} is singular.")

        return {"J_pos": J_pos, "J_rot": J_rot}


    def print_kinematics(self, body_name):
        """
        Prints the kinematic data of a given body.

        Parameters:
            body_name (str): Name of the body (e.g., 'FL_foot').
        """
        state = self.get_body_state(body_name)
        jacobian = self.get_jacobian(body_name)
        print(f"\n{body_name} Position: {state['position']}")
        print(f"{body_name} Orientation:\n{state['orientation']}")
        # print(f"{body_name} qpos: {self.data.qpos}")
        

    def update_joint_angles(self):
        """
        Continuously updates the joint angles in the background.
        """
        while self.running:
            self.joint_angles = self.joint_state_reader.joint_angles
            self.set_joint_angles(self.joint_angles)
            self.run_fk()
            time.sleep(0.01)  # Control update frequency

    def start_joint_updates(self):
        """Starts a background thread to continuously update joint angles."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.update_joint_angles)
            self.update_thread.start()

    def stop_joint_updates(self):
        """Stops the background joint update thread."""
        if self.running:
            self.running = False
            self.update_thread.join()

    def get_current_joint_angles(self):
        """
        Retrieves the latest joint angles.

        Returns:
            np.ndarray: Array of current joint angles.
        """
        return self.joint_state_reader.joint_angles
    def check_Jacobian_singularity(self, jacobian):
        """
        Check if the Jacobian matrix is singular.

        Parameters:
            jacobian (np.ndarray): The Jacobian matrix to check.

        Returns:
            bool: True if the Jacobian matrix is singular.
        """
        u, s, vh = np.linalg.svd(jacobian)
        if np.isclose(s, 0.0).any():
            return True
        else:
            return False
# Example Usage
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    # Initialize the Kinematics class with the XML file for the Go2 robot
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    kinematics = Kinematics(ROBOT_SCENE)
    
    kinematics.start_joint_updates()

    print("Joint Names and IDs:")
    for i in range(kinematics.model.njnt):
        joint_name = mujoco.mj_id2name(kinematics.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"Joint {i}: {joint_name}")

    # print("Body Names and IDs:")
    # for i in range(kinematics.model.nbody):
    #     body_name = mujoco.mj_id2name(kinematics.model, mujoco.mjtObj.mjOBJ_BODY, i)
    #     print(f"Body {i}: {body_name}")
    try:
        while True:
            # Print kinematics for 'FL_foot'
            frame = ["world", "base_link", "FL_foot", "RR_foot", "FR_foot", "RL_foot"]
            for i in frame:
                kinematics.print_kinematics(i)
            print("\n=== Joint Angles ===")
            print(kinematics.get_current_joint_angles())

            time.sleep(1.0)  # Adjust the frequency as needed
    except KeyboardInterrupt:
        # Gracefully stop the joint update thread on exit
        kinematics.stop_joint_updates()
