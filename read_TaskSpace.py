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

    def high_state_handler(self, msg: SportModeState_):
        """
        Callback to handle high state data and update leg positions.
        """
        
        for i in range(3):
            self.robot_state[i] = msg.position[i]
            self.robot_state[i+3] = msg.velocity[i]
        
        #    print(f"X state: {msg.position[0]:.3f} meters")
        #     print(f"Y state: {msg.position[1]:.3f} meters")
        #     print(f"Z state: {msg.position[2]:.3f} meters")

        return self.robot_state


if __name__ == "__main__":

    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_TaskSpace class
    task_space_reader = read_TaskSpace()
    while True:
        time.sleep(1.0)
        robot_state = task_space_reader.robot_state
        print("\n=== Task States ===")
        print(robot_state)

    # Keep the program running to continue receiving data
   