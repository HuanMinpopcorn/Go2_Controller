import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
import numpy as np

from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

class read_TaskSpace:
    def __init__(self):
 
        sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sub.Init(self.high_state_handler, 10)
        self.robot_state = unitree_go_msg_dds__SportModeState_()


    def high_state_handler(self, msg: SportModeState_):
        """
        Callback to handle high state data and update leg positions.
        """
        global robot_state
        self.robot_state = msg

    # the robot state include
    # position
    # velocity
    # yaw_speed
    # foot_position_body
    # foot_speed_body



if __name__ == "__main__":

    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_TaskSpace class
    task_space_reader = read_TaskSpace()
    robot_state = unitree_go_msg_dds__SportModeState_()
    while True:
        time.sleep(1.0)
        robot_state = task_space_reader.robot_state
        print("\n=== Task States read the postion x and velocity v===")
        print(robot_state.position)
        print("===============================================")
        print(robot_state.foot_position_body)

    # Keep the program running to continue receiving data
   