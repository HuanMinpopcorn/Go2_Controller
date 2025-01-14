import mujoco
import numpy as np
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from read_JointState import read_JointState
from read_TaskSpace import read_TaskSpace
from Simulation import config
from Forward_Kinematics import ForwardKinematic 

ChannelFactoryInitialize(1, "lo")
fk = ForwardKinematic()
fk.start_joint_updates()

class CentroidDynamic:
    def __init__(self, xml_path=config.ROBOT_SCENE):
        """
        Initializes the MuJoCo model and data.

        Parameters:
            xml_path (str): Path to the MuJoCo XML file for the robot.
        """
        self.model = fk.model
        self.data = fk.data
    
    def extract_inertia_matrix(self, cinert):
        """Extract the 3x3 inertia matrix from the cinert vector."""
        I = np.zeros((3, 3))
        # Upper triangular elements
        I[0, 0] = cinert[0]  # Ixx
        I[0, 1] = cinert[1]  # Ixy
        I[0, 2] = cinert[2]  # Ixz
        I[1, 1] = cinert[3]  # Iyy
        I[1, 2] = cinert[4]  # Iyz
        I[2, 2] = cinert[5]  # Izz
        # Symmetrize
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        return I

    def calculate_centroidal_moment_of_inertia(self):
        """Compute the centroidal moment of inertia (CMI)."""
        # System center of mass
        system_com = self.data.subtree_com[0]  # CoM of the entire system

        # Initialize the CMI
        I_cmi = np.zeros((3, 3))

        # Loop through each body
        for body_id in range(self.model.nbody):
            cinert = self.data.cinert[body_id]
            mass = cinert[9]
            # if mass == 0:
            #     com_offset = np.zeros(3)
            # else:
            #     com_offset = cinert[6:9] / mass  # Body CoM offset
            body_pos = self.data.xipos[body_id]
            body_inertia = self.extract_inertia_matrix(cinert)

            # Compute offset from system CoM
            r = body_pos - system_com

            # Parallel axis theorem
            r_outer = np.outer(r, r)
            shifted_inertia = body_inertia + mass * (np.dot(r, r) * np.eye(3) - r_outer)

            # Add to total CMI
            I_cmi += shifted_inertia

        return I_cmi

    def centroid_parameters(self):
        # Center of Mass (CoM) position
        com = self.data.subtree_com[0]  # Root subtree CoM

        # Total Mass
        total_mass = sum(self.model.body_mass)

        # Velocity of CoM 
        com_velocity = self.data.cdof[0]  # Extract the velocity of the CoM

        # Moment of Inertia (Icom) and Centroidal Momentum
        I_com = self.calculate_centroidal_moment_of_inertia()
        h_com = self.data.cvel[0]  # Centroidal momentum (linear and angular)

        return {
            "com_position": com,
            "total_mass": total_mass,
            "com_velocity": com_velocity,
            "I_com" : I_com, 
            "h_com": h_com,
        }



# Initialize CentroidDynamic
cd = CentroidDynamic()
params = cd.centroid_parameters()

# Print parameters
print("Center of Mass (Position):", params["com_position"])
print("Total Mass:", params["total_mass"])
print("CoM Velocity (Linear):", params["com_velocity"])
print("Moment of Inertia at CoM (I_com):", params["I_com"])
print("Centroidal Momentum (h_com):", params["h_com"])