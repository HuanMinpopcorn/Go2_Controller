
from Inverse_Dynamic import InverseDynamic     
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from reference_trajectory import ReferenceTrajectory
import time

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Simulation'))
from Simulation.PhysicalSim import PhysicalSim 
import threading
import socket
import pickle 
from Simulation import config

class WalkController:
    """
    This class orchestrates walking (or stepping) by calling the methods 
    of your existing inverse-dynamics (or IK) controller object.
    It also manages the weighting matrices for different gait phases.
    """
    def __init__(self):
        """
        Args:
            inverse_dynamics: An object that implements:
               - start_joint_updates()
               - move_to_initial_position()
               - initialize()
               - get_trajectory()
               - transition_legs()
               - calculate()
               - compute_torque()
               - send_command_api() or send_command_ik()
               - plot_error_ik(), plot_error_id()
               - Attributes like `K`, `ErrorPlotting`, `joint_toque`, etc.
        """
         # "idc" stands for "inverse dynamics controller"
        print("Initializing WalkController...")
        
        # Initialize simulation
        self.swing_phase = 0
        self.step_size = 0.005  # 200Hz Time step for simulation
        self.swing_time = 0.25  # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing

        # Real-time scaling
        self.real_time_factor = 1.0  # Adjust this factor to scale time (1.0 for real-time, <1.0 for slower, >1.0 for faster)
        self.step_size /= self.real_time_factor
       

        # Robot parameters
        self.body_height = 0.225 
        self.swing_height = 0.075
        # self.swing_height = 0.0

        # but rotation can be inserted at the same time with x and y
        self.velocity = {
            'x': 0.0,  # Forward velocity
            'y': 0.0,   # Lateral velocity
            'theta': 0.0  # Rotational velocity
        }

        # intialize the swing legs and contact legs
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
     

        self.idc = InverseDynamic(self.swing_legs, self.contact_legs)
        time.sleep(1)
        self.idc.cmd.move_to_initial_position()
        init_body,init_sw,init_contact,init_body_vel, init_sw_vel = self.idc.initialize()
         # 1. Prepare and initialize
        self.ref_traj = ReferenceTrajectory(init_body,init_sw,init_contact,init_body_vel, init_sw_vel,
                                        self.velocity,self.swing_height,self.swing_time,
                                        self.step_size,self.K, 
                                        self.swing_phase,self.swing_legs,self.contact_legs)


      

    
    def walk(self, controller="ID", running_time=1000):
        """
        Main entry point to perform walking using Inverse Dynamics or Inverse Kinematics.

        Args:
            controller (str): Either "ID" (inverse dynamics) or "IK" (inverse kinematics).
            running_time (int): Number of control steps.
        """

        print(f"Starting walk with controller: {controller} for {running_time} steps")
        # 2. Initialize the trajectory generator
        x_b, x_sw, x_b_dot, x_sw_dot = self.ref_traj.get_trajectory()
        # 3. Run the main loop
        if controller == "IK":
            for i in tqdm(range(running_time)):
                index = (i + 1) % self.K
                if index == 0:
                    self.ref_traj.transition_legs()
                    self.idc.transition_legs()
                    x_b, x_sw, x_b_dot, x_sw_dot= self.ref_traj.get_trajectory()
                    
                # Compute inverse kinematics + update commands
                self.idc.calculate(x_sw, x_b, x_sw_dot, x_b_dot, index)
                self.idc.send_command_ik()

            # Plot any IK-related errors
            self.idc.plot_error_ik()
            # self.idc.plot_error_id()
            plt.show()
        else:  # Use Inverse Dynamics
            for i in tqdm(range(running_time)):
                self.idc.ErrorPlotting.torque_sensor_data.append(self.idc.joint_toque)
                
                index = (i + 1) % self.K
                if index == 0:
                    self.ref_traj.transition_legs()
                    self.idc.transition_legs()
                    x_b, x_sw, x_b_dot, x_sw_dot = self.ref_traj.get_trajectory()
            
                # Calculate IK/ID references
                self.qd, self.dqd, self.ddqd, self.ddqd_cmd= self.idc.calculate(x_sw, x_b, x_sw_dot, x_b_dot, index)

                self.idc.compute_torque()
                self.idc.send_command_api()
                # self.idc.send_command_ik()
            # Plot ID and IK errors (if needed)
            self.idc.plot_error_ik()
            self.idc.plot_error_id()
            plt.show()



if __name__ == "__main__":
    
    
    ChannelFactoryInitialize(1, "lo")
    # Initialize simulation
    # sim = PhysicalSim()
    wc = WalkController()
    wc.walk(controller="IK", running_time=4000)
    

