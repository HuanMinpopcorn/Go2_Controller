from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import mujoco
import config
import threading
import sys
import numpy as np
import time

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35

# STAND_UP_JOINT_POS = np.array([
#             0.052, 1.12, -2.10, -0.052, 1.12, -2.10,
#             0.052, 1.12, -2.10, -0.052, 1.12, -2.10
#         ], dtype=float)

STAND_UP_JOINT_POS = np.array([
            0.062, 1.02, -1.80, -0.062, 1.02, -1.80,
            0.062, 1.02, -1.80, -0.062, 1.02, -1.80
        ], dtype=float)

STAND_DOWN_JOINT_POS = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ], dtype=float)

class send_motor_commands():
    def __init__(self):
        # ChannelFactoryInitialize(1, "lo")
    

        self.low_cmd_pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_) 
        self.low_cmd_pub.Init()
        self.cmd = self._initialize_motor_commands()
        self.crc = CRC()
        self.step_size = config.SIMULATE_DT
        # self.step_size = 0.001

    def _initialize_motor_commands(self):
        """
        Initialize motor command message with default settings.
        """
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head = [0xFE, 0xEF]
        cmd.level_flag = 0xFF  # Low-level control

        for motor in cmd.motor_cmd:
            motor.mode = 0x01  # PMSM mode
            motor.q = 0.0
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        return cmd

    def send_motor_commands(self, kp, kd,new_joint_angles, q_dot, torque=np.zeros(12)):
        """
        Send motor commands to the robot using the publisher.
        """
        for i in range(12):
            self.cmd.motor_cmd[i].q  = new_joint_angles[i]
            self.cmd.motor_cmd[i].dq = q_dot[i]
            self.cmd.motor_cmd[i].kp = kp
            self.cmd.motor_cmd[i].kd = kd
            self.cmd.motor_cmd[i].tau = torque[i]

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_pub.Write(self.cmd)
        time.sleep(self.step_size)

    def update_motor_commands(self):
        """
        Continuously update motor commands in a separate thread.
        """
        while True:
            self.low_cmd_pub.Write(self.cmd)
            time.sleep(self.step_size)

    def move_to_initial_position(self, current_joint_angles = STAND_DOWN_JOINT_POS , target_joint_angles = STAND_UP_JOINT_POS):
        """
        Smoothly transition to the initial standing position and maintain it.
        """
        print("Transitioning to initial position...")
        print("Press Ctrl+ C to enter Trot Gait ...")
        running_time = 0.0
        try:
            while True:
                # Check if the robot is at the target body height
                running_time += self.step_size
                
                # Smoothly transition to the initial position
                phase = np.tanh(running_time / 1.2)
                for i in range(12):
                    self.cmd.motor_cmd[i].q = phase * target_joint_angles[i] + (
                        1 - phase) * current_joint_angles[i]
                    self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0  # Gradual stiffness
                    self.cmd.motor_cmd[i].kd = 3.5
                
                self.cmd.crc = self.crc.Crc(self.cmd)
                self.low_cmd_pub.Write(self.cmd)
                time.sleep(self.step_size)
                
        except KeyboardInterrupt:
            # Gracefully stop the joint update thread on exit
            pass


if __name__ == '__main__':

    ChannelFactoryInitialize(1, "lo")
    cmd = send_motor_commands()
    cmd.move_to_initial_position()