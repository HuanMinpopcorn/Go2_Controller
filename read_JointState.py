import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
import numpy as np


class read_JointState:
    def __init__(self):
        self.joint_angles = np.zeros(12)
        
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.low_state_handler, 10)
    def low_state_handler(self, msg: LowState_):
        """
        Callback to handle high state data and print joint state for each leg.
        """
        for i in range(12):
            self.joint_angles[i] = msg.motor_state[i].q
        return self.joint_angles


if __name__ == "__main__":

    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_JointState class
    joint_state_reader = read_JointState()
   
    # print(joint_angles)

    # Keep the program running to continue receiving data
    while True:
        time.sleep(1.0)
        joint_angles = joint_state_reader.joint_angles
        print("\n=== Joint States ===")
        print(joint_angles)
        
