import numpy as np
from scipy import sparse
import time

from Inverse_Dynamic import InverseDynamic
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import osqp
import mujoco
import matplotlib.pyplot as plt
from tqdm import tqdm  
import Simulation.config as config

class walk():
    def __init__(self):
        self.step_size = config.SIMULATE_DT
        self.swing_time =  0.25 # 0.25 # Duration of swing phase
        self.K = int(self.swing_time / self.step_size)  # Number of steps for swing

        self.num_actuated_joints = self.model.nv - 6

        # Robot parameters
        self.body_height = 0.225 
        self.swing_height = 0.075
        # self.swing_height = 0.0
        self.velocity = 0  # Forward velocity

        self.world_frame_name = "world"
        self.body_frame_name = "base_link"
        self.leg = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FR_foot", "RL_foot"]
  
        self.walk_phase = ["standing","transition", "swing"]
    def walk_state_machine(self):
        for phase in self.walk_phase:
            if phase == "double support":
                self.swing_leg = []
                self.contact_legs = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                self.double_support()
            elif phase == "transition":
                self.transition()
            elif phase == "swing":
                self.swing()
            else:
                raise ValueError("Invalid phase")
    def walk_main(self):
        # Initialize the model and data
        # Set initial joint positions and velocities
        qpos_init = np.zeros(self.model.nq)
        qvel_init = np.zeros(self.model.nv)
        qpos_init[self.motors] = -np.pi / 2
        qvel_init[self.motor_vel] = 0.0

        # Set the initial state
        mujoco.mj_set_state(self.model, self.data, qpos_init, qvel_init)

        # Define the time step and total simulation time
        dt = 0.01
        total_time = 10.0
        num_steps = int(total_time / dt)

        # Initialize arrays to store joint positions and velocities for plotting
        joint_positions = np.zeros((num_steps, self.num_motors))
        joint_velocities = np.zeros((num_steps, self.num_motors))

        # Run the simulation loop
        for step in tqdm(range(num_steps)):
            mujoco.mj_step(self.model, self.data)

            # Store joint positions and velocities
            joint_positions[step] = mujoco.mj_get_state(self.model, self.data)[0][self.motors]
            joint_velocities[step] = mujoco.mj_get_state(self.model, self.data)[1][self.motor_vel]

if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
   
    inv_dyn = InverseDynamic()
    inv_dyn.ID_main()
