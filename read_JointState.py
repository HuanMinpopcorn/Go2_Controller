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
        self.imu_data = np.zeros(4)
    
        
        
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        """
        Callback to handle low state data and update joint state for each leg.
        """
        for i in range(12):
            self.joint_angles_temp[i] = msg.motor_state[i].q
            
        for j in range(4):
            self.imu_data[j] = msg.imu_state.quaternion[j] # IMU data
  
        # change the order of the joint angles
        self.joint_angles = np.array([self.joint_angles_temp[3], self.joint_angles_temp[4], 
        self.joint_angles_temp[5], self.joint_angles_temp[0], self.joint_angles_temp[1], 
        self.joint_angles_temp[2], self.joint_angles_temp[9], self.joint_angles_temp[10], 
        self.joint_angles_temp[11], self.joint_angles_temp[6], self.joint_angles_temp[7], self.joint_angles_temp[8]])


if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_JointState class
    joint_state_reader = read_JointState()
   
    # Keep the program running to continue receiving data
    while True:
        time.sleep(1.0)
        joint_angles = joint_state_reader.joint_angles
        print("\n=== Joint States ===")
        print(joint_angles)
        
