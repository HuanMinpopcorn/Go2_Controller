
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
import Simulation.config as config
from reference_trajectory import ReferenceTrajectory
import time

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
        # walk_phase = ["double_standing", "transition", "swing"] 
        self.walk_phase = "double_standing" # intial phase
        self.swing_phase = 0

        self.step_size = config.SIMULATE_DT * 10
        self.swing_time =  0.25/2  # 0.25 # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing
       

        # Robot parameters
        self.body_height = 0.225 
        self.swing_height = 0.075
        # self.velocity = 0.02 # Forward velocity
        self.velocity = 0.005 # Forward velocity

        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
     

        self.idc = InverseDynamic(self.swing_legs, self.contact_legs)


      

    
    def walk(self, controller="ID", running_time=1000):
        """
        Main entry point to perform walking using Inverse Dynamics or Inverse Kinematics.

        Args:
            controller (str): Either "ID" (inverse dynamics) or "IK" (inverse kinematics).
            running_time (int): Number of control steps.
        """
        # 1. Prepare and initialize
        # self.idc.start_joint_updates()
        # time.sleep(5)
        self.idc.cmd.move_to_initial_position()
        init_body,init_sw,init_contact,init_body_vel, init_sw_vel = self.idc.initialize()
        ref_traj = ReferenceTrajectory(init_body,init_sw,init_contact,init_body_vel, init_sw_vel,
                                        self.velocity,self.swing_height,self.swing_time,
                                        self.step_size,self.K, 
                                        self.swing_phase,self.walk_phase,self.swing_legs,self.contact_legs)

        # 2. Initialize the trajectory generator
        x_b, x_sw, x_b_dot, x_sw_dot = ref_traj.get_trajectory(self.walk_phase)

        # 3. Run the main loop
        if controller == "IK":
            for i in tqdm(range(running_time)):
                index = (i + 1) % self.K
                if index == 0:
                    ref_traj.transition_legs()
                    self.idc.transition_legs()
                    x_b, x_sw, x_b_dot, x_sw_dot= ref_traj.get_trajectory(self.walk_phase)
                    
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
                    ref_traj.transition_legs()
                    self.idc.transition_legs()
                    x_b, x_sw, x_b_dot, x_sw_dot = ref_traj.get_trajectory(self.walk_phase)
            
                # Calculate IK/ID references
                self.idc.calculate(x_sw, x_b, x_sw_dot, x_b_dot, index)
                # Now compute ID torque and send to robot
                self.idc.compute_torque()
                self.idc.send_command_api()

            # Plot ID and IK errors (if needed)
            self.idc.plot_error_ik()
            self.idc.plot_error_id()
            plt.show()



if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    wc = WalkController()
    wc.walk(controller="ID", running_time=500)
