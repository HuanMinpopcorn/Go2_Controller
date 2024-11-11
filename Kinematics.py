import mujoco
import numpy as np
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from read_JointState import read_JointState
from read_TaskSpace import read_TaskSpace
from PhysicalSim import PhysicalSim



class Kinematics:
    # def __init__(self,sim):
    
        
    #     # read the model and data from the sim 
    #     self.model = sim.mj_model
    #     self.data =  sim.mj_data
    #     self.joint_angles = sim.mj_data.qpos[7:]
    #     self.robot_position = sim.mj_data.xpos
    #     # id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    #     # self.diff = self.data.xpos[id] - self.data.site_xpos
    #     # self.diff_quat = self.data.xmat[id] - self.data.site_xmat

    def __init__(self, xml_path):
        """
        Initializes the MuJoCo model and data.

        Parameters:
            xml_path (str): Path to the MuJoCo XML file for the robot.
        # """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_state_reader = read_JointState()  # Initialize joint state reader
        self.joint_angles = self.joint_state_reader.joint_angles
        self.task_space_reader = read_TaskSpace()
        self.robot_state = self.task_space_reader.robot_state
        self.imu_data = self.joint_state_reader.imu_data
        self.update_thread = None  # Thread for continuous updates
        self.running = False  # Control flag for the thread

    def set_joint_angles(self):
        """
        Sets the joint angles in the MuJoCo qpos array.
        """
        self.data.qpos[:3] = self.robot_state.position 
        # print(f"robot_state.position: {self.robot_state.position}") 
        self.data.qpos[3:7] = self.imu_data
        self.data.qpos[7:] = self.joint_angles
    

    def run_fk(self):
        """
        Runs forward kinematics to update the robot's positions and orientations 
        based on current joint angles. Provides debug output for the forward 
        kinematics process and error checking.
        """
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
        # print(f"body_id: {body_id}")
        position = np.copy(self.data.xpos[body_id])  # 3D position
        orientation_quat = np.copy(self.data.xquat[body_id])  # Quaternion orientation
        orientation = self.convert_quat_to_euler(orientation_quat)  # Euler angles

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
        np.set_printoptions(precision=5, suppress=True)
        print(f"\n{body_name} Position: {state['position']}")
        print(f"{body_name} Orientation:\n{state['orientation']}")
    

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

    def convert_quat_to_euler(self, quat):
        """
        Convert quaternion to Euler angles.

        Parameters:
            quat (np.ndarray): Quaternion to convert.

        Returns:
            np.ndarray: Euler angles.
        """
        q0, q1, q2, q3 = quat
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(2*(q0*q2 - q3*q1))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return np.array([roll, pitch, yaw])

    def update_joint_angles(self):
        """
        Continuously updates the joint angles in the background.
        """
        while self.running:
            self.joint_angles = self.joint_state_reader.joint_angles
            self.imu_data = self.joint_state_reader.imu_data
            self.robot_state = self.task_space_reader.robot_state
            self.set_joint_angles()
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

    

# Example Usage
if __name__ == "__main__":

    MODE = "SDK"
    if MODE == "SDK":
        # ==== SDK Mode ====
        ChannelFactoryInitialize(1, "lo")
        # Initialize the Kinematics class with the XML file for the Go2 robot
        ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
        # run the forward kinematics    
        fk = Kinematics(ROBOT_SCENE)
        fk.start_joint_updates()
        # === end ===
    else:
        # ==== Simulation Mode ====
        # "===simulation mode==" 
        # import sim from PysicalSim
        sim = PhysicalSim()
        sim.start()
        fk = Kinematics(sim)

        # # ====== end =====

    try:
        while True:
            # Print kinematics for 'FL_foot'
            frame = ["world", "base_link", "FL_foot", "RR_foot", "FR_foot", "RL_foot"]
            for i in frame:
                fk.print_kinematics(i)
            print("\n=== Joint Angles ===")
            print(np.array(fk.data.qpos[7:]))
            print("\n=== sensor Data ===")
            print(np.array(fk.data.sensordata[:12]))
            print("\n=== model gravity Data ===")
            print(np.array(fk.data.qfrc_bias[6:]))

            # print(f"difference" , fk.diff, fk.diff_quat)

            time.sleep(1.0)  # Adjust the frequency as needed
    except KeyboardInterrupt:
        # Gracefully stop the joint update thread on exit
        fk.stop_joint_updates()
