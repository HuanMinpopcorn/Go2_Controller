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
        self.robot_state = np.zeros(6)
        self.foot_positions = np.zeros(12)

    def high_state_handler(self, msg: SportModeState_):
        """
        Callback to handle high state data and update leg positions.
        """
        # read the body postion x and body velocity v
        for i in range(3):
            self.robot_state[i] = msg.position[i]
            self.robot_state[i+3] = msg.velocity[i]
        
        foot_positions = np.zeros(12)
        for i in range(12):
            self.foot_positions[i] = msg.foot_position_body[i]


if __name__ == "__main__":

    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_TaskSpace class
    task_space_reader = read_TaskSpace()
    while True:
        time.sleep(1.0)
        robot_state = task_space_reader.robot_state
        print("\n=== Task States read the postion x and velocity v===")
        np.set_printoptions(precision=3, suppress=True)
        print(robot_state)
        print("===============================================")
        print(task_space_reader.foot_positions)

    # Keep the program running to continue receiving data
   