import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
import numpy as np

class read_JointState:
    def __init__(self):
        self.joint_angles = np.zeros(12)
        self.imu_data = np.zeros(4)
        
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        """
        Callback to handle low state data and update joint state for each leg.
        """
        for i in range(12):
            self.joint_angles[i] = msg.motor_state[i].q
        for j in range(4):
            self.imu_data[j] = msg.imu_state.quaternion[j] # IMU data

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
        
