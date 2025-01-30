import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
import numpy as np

class read_JointState:
    def __init__(self):
        self.joint_angles_temp = np.zeros(12)
        self.joint_angles = np.zeros(12)


        self.joint_velocity = np.zeros(12)
        self.joint_tau = np.zeros(12)

        self.imu_data = np.zeros(4)
        self.imu_gyroscope = np.zeros(3)
        self.imu_accelerometer = np.zeros(3)
    
        
        
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        """
        Callback to handle low state data and update joint state for each leg.
        """


        for i in range(12):
            self.joint_angles_temp[i] = msg.motor_state[i].q
            self.joint_velocity[i] = msg.motor_state[i].dq
            self.joint_tau[i] = msg.motor_state[i].tau_est

        for j in range(4):
            self.imu_data[j] = msg.imu_state.quaternion[j] # IMU data

        for j in range(3):
            self.imu_gyroscope[j] = msg.imu_state.gyroscope[j] # IMU data
            self.imu_accelerometer[j] = msg.imu_state.accelerometer[j] # IMU data
         
        # change the order of the joint angles in urdf file order 
        self.joint_angles = np.array([self.joint_angles_temp[3], self.joint_angles_temp[4], 
        self.joint_angles_temp[5], self.joint_angles_temp[0], self.joint_angles_temp[1], 
        self.joint_angles_temp[2], self.joint_angles_temp[9], self.joint_angles_temp[10], 
        self.joint_angles_temp[11], self.joint_angles_temp[6], self.joint_angles_temp[7], self.joint_angles_temp[8]])

        # change the order of the joint velocities
        self.joint_velocity = np.array([self.joint_velocity[3], self.joint_velocity[4], 
        self.joint_velocity[5], self.joint_velocity[0], self.joint_velocity[1], 
        self.joint_velocity[2], self.joint_velocity[9], self.joint_velocity[10], 
        self.joint_velocity[11], self.joint_velocity[6], self.joint_velocity[7], self.joint_velocity[8]])

        # change the order of the joint accelerations
        self.joint_tau = np.array([self.joint_tau[3], self.joint_tau[4], 
        self.joint_tau[5], self.joint_tau[0], self.joint_tau[1], 
        self.joint_tau[2], self.joint_tau[9], self.joint_tau[10], 
        self.joint_tau[11], self.joint_tau[6], self.joint_tau[7], self.joint_tau[8]])

    
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_JointState class
    joint_state_reader = read_JointState()
   
    # Keep the program running to continue receiving data
    while True:
        time.sleep(1.0)
        
        print("\n=== Joint States ordinary ===")
        print(joint_state_reader.joint_angles_temp)
        print("\n=== joint Data after change the order ===")
        print(joint_state_reader.joint_angles)
        print("\n=== joint velocity  ===")
        print(joint_state_reader.joint_velocity)
        print("\n=== joint acc  ===")
        print(joint_state_reader.joint_tau)
        print("===============================================")
        print(joint_state_reader.imu_data)
        print("===============================================")
        print(joint_state_reader.imu_gyroscope)
        print("===============================================")
        print(joint_state_reader.imu_accelerometer)
        print("===============================================")
