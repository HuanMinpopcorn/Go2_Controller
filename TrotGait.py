import time
import numpy as np
import threading


from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)



from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ 
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

from unitree_sdk2py.utils.crc import CRC

from Kinematics import Kinematics 
from read_JointState import read_JointState
from read_TaskSpace import read_TaskSpace


class TrotGaitController:
    def __init__(self, pub, crc, dt=0.01):
        """
        Initialize the trot gait controller.
        """
        self.pub = pub  # Publisher for Unitree commands
        self.crc = crc  # CRC utility for message integrity
        self.dt = dt  # Control loop time step (500Hz)

        self.body_height = 0.25  # Desired body height
        self.swing_height = 0.075  # Height of the foot during swing
        self.swing_time = 0.25  # Time for one swing phase
        self.step_length = 0.1  # Length of each step

        self.lock = threading.Lock()  # Thread-safe lock for high state access
        self.high_state_read = False  # Flag to track high state read

        # Initialize motor command message
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF  # Set to low-level control

        # Initialize all motor commands
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
    def move_to_initial_position(self):
        """
        Smoothly transition to the initial standing position and maintain it.
        """
        # Define initial and final joint positions for smooth transition
        stand_up_joint_pos = np.array([
            0.052, 1.12, -2.10, -0.052, 1.12, -2.10,
            0.052, 1.12, -2.10, -0.052, 1.12, -2.10
        ], dtype=float)

        stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ], dtype=float)


        running_time = 0.0
        while True:
            # Check if the robot is at the target body height
            running_time += self.dt
            
            # Smoothly transition to the initial position
            phase = np.tanh(running_time / 1.2)
            for i in range(12):
                self.cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (
                    1 - phase) * stand_down_joint_pos[i]
                self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0  # Gradual stiffness
                self.cmd.motor_cmd[i].kd = 3.5

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            time.sleep(self.dt)
        

                                                                                                                                                                                                                                                                              


def main():
    """
    Main function to initialize and start the controller.
    """
    crc = CRC()
    ChannelFactoryInitialize(1, "lo")

    # initialize the channel publisher
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    controller = TrotGaitController(pub, crc)
    controller.move_to_initial_position()

    # controller.subscribe_high_state()



if __name__ == '__main__':
    main()
