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

MOTOR_SENSOR_NUM = 3
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
        
        self.joint_state_reader = read_JointState()  # Initialize joint state reader
        self.joint_angles = self.joint_state_reader.joint_angles
        
        self.task_space_reader = read_TaskSpace()
        self.robot_state = self.task_space_reader.robot_state

        self.imu_data = self.joint_state_reader.imu_data

        
        self.update_thread = None  # Thread for continuous updates
        self.running = False  # Control flag for the thread


        self.num_motor = self.model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
    def set_joint_angles(self):
        """
        Sets the joint angles in the MuJoCo qpos array.
        """
        self.data.qpos[:3] = self.robot_state[:3]
        self.data.qpos[3:7] = self.imu_data
        self.data.qpos[7:] = self.joint_angles
        
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
        np.set_printoptions(precision=3, suppress=True)
        print(f"\n{body_name} Position: {state['position']}")
        print(f"{body_name} Orientation:\n{state['orientation']}")
        

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

    # def read_the_imu_data(self):
    #     """
    #     Read the sensor data from the robot.

    #     Returns:
    #         np.ndarray: Array of sensor data.
    #     """

    #     # read the frame sensor data 
    #     imu_pos = np.copy(self.data.sensordata[self.dim_motor_sensor + 10 : self.dim_motor_sensor+13])
    #     imu_vel = np.copy(self.data.sensordata[self.dim_motor_sensor + 13 : self.dim_motor_sensor + 16])
    #     imu_quat = np.copy(self.data.sensordata[self.dim_motor_sensor + 0 : self.dim_motor_sensor + 4])
    #     imu_gyro = np.copy(self.data.sensordata[self.dim_motor_sensor + 4 : self.dim_motor_sensor + 7])
    #     imu_acc = np.copy(self.data.sensordata[self.dim_motor_sensor + 7 : self.dim_motor_sensor + 10])
    #     # print(f"IMU Position: {imu_pos}")
    #     # print(f"IMU Velocity: {imu_vel}")
    #     # print(f"IMU Quaternion: {imu_quat}")
    #     # print(f"IMU Gyro: {imu_gyro}")
    #     # print(f"IMU Acceleration: {imu_acc}")

    #     return {"pos": imu_pos, "vel": imu_vel, "quat": imu_quat, "gyro": imu_gyro, "acc": imu_acc}

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


    

# Example Usage
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    # Initialize the Kinematics class with the XML file for the Go2 robot
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    fk = Kinematics(ROBOT_SCENE)


    
    fk.start_joint_updates()
    try:
        while True:
            # Print kinematics for 'FL_foot'
            frame = ["world", "base_link", "FL_foot", "RR_foot", "FR_foot", "RL_foot"]
            for i in frame:
                fk.print_kinematics(i)
            print("\n=== Joint Angles ===")
            # print(kinematics.get_current_joint_angles())
            # print("\n=== Sensor Data ===")
            print(f"qpos: {fk.data.qpos}")
            # print(f"qpos0: {kinematics.model.qpos0}")
            # print(f"   
            # kinematics.read_the_imu_data()
        
            time.sleep(1.0)  # Adjust the frequency as needed
    except KeyboardInterrupt:
        # Gracefully stop the joint update thread on exit
        fk.stop_joint_updates()
