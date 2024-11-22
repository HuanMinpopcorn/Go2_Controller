import numpy as np
from scipy import sparse
import config 
from forward_kinematics import ForwardKinematic as FK
from Inverse_Kinematics import InverseKinematic as IK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import osqp
import threading

"""
This class is responsible for computing the inverse dynamics of the robot.
* input: Mujoco simulation data, forward kinematics data, inverse kinematics data
* output: joint torques

The inverse dynamics problem using OSQP library is formulated as follows:
minimize: Fc^T W_Fc Fc + ddxc^T W_ddxc ddxc + ddq^T W_ddq ddq   (1)
subject to: A_ddq ddq + B_ddxc ddxc + C_Fc Fc = tau            (2)
            ddq_min <= ddq <= ddq_max                           (3)
            Fc_min <= Fc <= Fc_max                              (4)
where: 
"""
ChannelFactoryInitialize(1, "lo")
IK.start_joint_updates()
IK.cmd.move_to_initial_position()
IK.initialize()



# Define problem data 
F_dim = 3 * len(IK.contact_legs) # number of contact legs
ddxc_dim = 3 * len(IK.contact_legs) # number of contact legs
ddq_dim = len(IK.data.qacc) # number of joint accelerations


WF = sparse.csc_matrix(5 * np.eye(F_dim)) # weight matrix for contact forces
Wc = sparse.csc_matrix(5* np.eye(ddxc_dim)) # weight matrix for contact accelerations
Wddq = sparse.csc_matrix(5 * np.eye(ddq_dim)) # weight matrix for joint accelerations

# eqaulity constraints 
M = sparse.csc_matrix(IK.data.qM) # mass matrix
B = sparse.csc_matrix(IK.data.qfrc_bias) # bias vector
tau = IK.data.qfrc_passive + IK.data.qfrc_actuator + IK.data.qfrc_applied
Jc = IK.get_required_jacobian[]

# class InverseDynamic():
#     def __init__(self, initial_conditions, parameters):

        
#         self.prob = osqp.OSQP()
#         # cost function variable
#         self.WF = sparse.csc_matrix()
#         self.Wc = sparse.csc_matrix()
#         self.Wddq = sparse.csc_matrix()
#         # decision variables 
#         self.Fc

#         self.initial_conditions = initial_conditions
#         self.parameters = parameters
#         self.state = self.initialize_state()
#     def format_optimization(self):
#         # set up the constraints
#         # Holonomic constraints

#         # self.constraints = Constraints(self.parameters)
