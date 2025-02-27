import numpy as np
from scipy import sparse
import time

from Control.Inverse_Kinematics import InverseKinematic

import osqp
import mujoco
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import Lock

class InverseDynamic(InverseKinematic):
    def __init__(self,swing_legs, contact_legs):
        super().__init__(swing_legs, contact_legs)

        self.mu = 0.6  # Friction coefficient
        self.F_z_max = 50  # Max contact force
        self.tau_min = -50.0  # Torque limits
        self.tau_max = 50.0
        
        self.WF = []
        self.Wc = []
        self.Wddq = []
        self.ddq_cmd = np.zeros((self.model.nv - 6, 1))  # Initialize ddq_cmd
    
        # Solver initialization
        self.prob = osqp.OSQP()

        # Placeholder for problem matrices and bounds
        self.A = None  # Combined constraint matrix
        self.l = None  # Lower bounds
        self.u = None  # Upper bounds
        self.P = None  # Hessian matrix
        self.q = None  # Gradient vector

        

        
        
        
    def initialize_Weight_Matrix(self):
     
       
        # ncon = self.data.ncon
        
        
        self.lock.acquire()
        ncon = self.ncon
        self.Jc_id = self.Jc.copy()
        self.Jc_dot_id = self.Jc_dot.copy()
        self.lock.release()
        # print(f"Contact Jacobian for contact {self.ncon}:\n{self.Jc.shape}")
        if self.Jc_id.shape[0] != 3 * ncon or self.Jc_dot_id.shape[0] != 3 * ncon:
            raise ValueError(f"Shape mismatch: Jc shape {self.Jc_id.shape[0]}, Jc_dot shape {self.Jc_dot_id.shape[0]}, expected {3 * ncon}")


        self.F_dim = 3 * ncon  # Number of contact forces
        self.ddxc_dim = 3 * ncon  # Contact accelerations
        self.ddq_dim = self.model.nv  # Joint accelerations (DOFs) 
        self.number_of_contacts = ncon

        # Higher weight for x and y directions
        WF = np.eye(self.F_dim) * 1
        for i in range(self.number_of_contacts):
            WF[3 * i, 3 * i] = 10  # x direction
            WF[3 * i + 1, 3 * i + 1] = 10  # y direction
        self.WF = sparse.csc_matrix(WF)
        Wc = np.eye(self.ddxc_dim) * 1
        for i in range(self.number_of_contacts):
            Wc[3 * i, 3 * i] = 100  # x direction
            Wc[3 * i + 1, 3 * i + 1] = 100  # y direction
        self.Wc = sparse.csc_matrix(Wc)
        self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim) * 10)


    
    def whole_body_dynamics_constraint(self):
        """
        Formulate the whole-body dynamics constraints.
        """
        # print("Starting whole-body dynamics constraint...")

        # Compute mass matrix and bias forces
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms

        
        # self.tau = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - self.data.qfrc_inverse.reshape(-1, 1))
        tau_cmd = np.vstack((np.zeros((6, 1)), self.tau.reshape(-1, 1)))    
        # J_contact  
        A1 = sparse.csc_matrix(- self.Jc_id .T)  # Transposed contact Jacobian
        A2 = sparse.csc_matrix(( self.Jc_id .shape[1],  self.Jc_id .shape[0]))  # Placeholder
        A3 = sparse.csc_matrix(M)  # Mass matrix in sparse format
        A_matrix = sparse.hstack([A1, A2, A3])  # Constraint matrix
        dynamics_u = tau_cmd - B - M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd))  # Equality constraint RHS
        dynamics_l = dynamics_u  # Equality constraint bounds
        # print("Whole-body dynamics constraint formulated.")
        return A_matrix, dynamics_l, dynamics_u

    def kinematic_constraints(self):
        """
        Formulate the kinematic constraints.
        """
   
        A1 = sparse.csc_matrix(np.zeros((self.F_dim, self.F_dim)))  # Zero matrix
        A2 = sparse.csc_matrix(np.eye(self.ddxc_dim))  # Identity matrix
        A3 = - self.Jc_id.copy()   # Negative Jacobian
        # print(f"A1: {A1.shape}, A2: {A2.shape}, A3: {A3.shape}")
        A_matrix = sparse.hstack([A1, A2, A3],format='csc')  # Constraint matrix

        l =  self.Jc_id @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + self.Jc_dot_id @ self.data.qvel.copy().reshape(-1,1)  # Lower bound
        u = l
        return A_matrix, l, u

    def reaction_force_constraints(self):
        """
        Formulate the reaction force constraints.
        """
        # Friction cone
        S_single = np.array([
            [1,  0, -self.mu],
            [-1, 0, -self.mu],
            [0,  1, -self.mu],
            [0, -1, -self.mu]
        ])
        if self.number_of_contacts == 0:
            A_react = sparse.csc_matrix((self.F_dim + self.ddxc_dim + self.ddq_dim, self.F_dim + self.ddxc_dim + self.ddq_dim))
            l_react = np.zeros((self.F_dim + self.ddxc_dim + self.ddq_dim, 1))
            u_react = np.zeros((self.F_dim + self.ddxc_dim + self.ddq_dim, 1))
        else:
            S = sparse.block_diag([S_single] * self.number_of_contacts)
            F_r_max = (np.ones(S.shape[0]) * self.F_z_max).reshape(-1, 1)
            A1 = sparse.csc_matrix(S)
            A2 = sparse.csc_matrix((S.shape[0], 3 * self.number_of_contacts))
            A3 = sparse.csc_matrix((S.shape[0], self.ddq_dim))
            A_react = sparse.hstack([A1, A2, A3], format='csc')
            l_react = -np.inf * np.ones(S.shape[0]).reshape(-1, 1)
            u_react = F_r_max
        return A_react, l_react, u_react
    
    def compute_contact_force_direction_constraints(self):
        Fz_single = np.array([
            [0, 0,  1],     # -Fz ≤ 0 → Fz ≥ 0
        ])
        if self.number_of_contacts == 0:
            A_matrix = sparse.csc_matrix((self.F_dim + self.ddxc_dim + self.ddq_dim, self.F_dim + self.ddxc_dim + self.ddq_dim))
            l = np.zeros((self.F_dim + self.ddxc_dim + self.ddq_dim, 1))
            u = np.zeros((self.F_dim + self.ddxc_dim + self.ddq_dim, 1))
        else:
            Fz = sparse.block_diag([Fz_single] * self.number_of_contacts)

            A1 = sparse.csc_matrix(Fz)
            A2 = sparse.csc_matrix((Fz.shape[0], self.ddxc_dim))
            A3 = sparse.csc_matrix((Fz.shape[0], self.ddq_dim))
            A_matrix = sparse.hstack([A1, A2, A3], format='csc')
            l = np.zeros(Fz.shape[0]).reshape(-1, 1)
            u = np.ones(Fz.shape[0]).reshape(-1, 1) * np.inf
        return A_matrix, l, u

    def setup_cost_function(self):
        """
        Define the cost function.
        """
        # print(f"{self.WF.shape}, {self.Wc.shape}, {self.Wddq.shape}")
        self.P = sparse.block_diag([self.WF, self.Wc, self.Wddq], format='csc')
        self.q = np.zeros(self.P.shape[0])

    def combine_constraints(self, ):
        """
        Combine all constraints into a single matrix.
        """
        A_dyn, l_dyn, u_dyn = self.whole_body_dynamics_constraint()
        A_react, l_react, u_react = self.reaction_force_constraints()
        A_contact, l_contact, u_contact = self.kinematic_constraints()
        A_contact_force_dir, l_contact_force_dir, u_contact_force_dir = self.compute_contact_force_direction_constraints()

        self.A = sparse.vstack([A_dyn, A_react, A_contact, A_contact_force_dir])
        self.l = np.vstack([l_dyn, l_react, l_contact, l_contact_force_dir])
        self.u = np.vstack([u_dyn, u_react, u_contact, u_contact_force_dir])


    def _QP_solve(self):
        """
        Solve the optimization problem usinga OSQP.
        """
        self.initialize_Weight_Matrix()
        self.combine_constraints()
        self.setup_cost_function()
        self.prob = osqp.OSQP()
        # print(f"P shape: {self.P.shape}, q shape: {self.q.shape}, A shape: {self.A.shape}, l shape: {self.l.shape}, u shape: {self.u.shape}")
        self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, max_iter=100,
                        verbose=False)
        # self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
        #                 verbose=False)
        result = self.prob.solve()

        # Extract accelerations and forces      
        Fc_sol = result.x[:self.F_dim]
        ddxc_sol = result.x[self.F_dim:self.F_dim + self.ddxc_dim]
        ddq_sol = result.x[-self.ddq_dim:]
        # print(f"ddq_sol: {ddq_sol.shape}, Fc_sol: {Fc_sol.shape}, ddxc_sol: {ddxc_sol.shape}")
        self.ErrorPlotting.Fc_data.append(Fc_sol)
        self.ErrorPlotting.ddxc_data.append(ddxc_sol)
        self.ErrorPlotting.ddq_diff_data.append(ddq_sol)
        return Fc_sol.reshape(-1, 1), ddxc_sol.reshape(-1, 1), ddq_sol.reshape(-1, 1)

    def compute_torque(self):
        """
        Compute joint torques using the inverse dynamics solution.
        """
        # print("Computing torques...")
        Fc_sol, ddxc_sol, ddq_sol = self._QP_solve()
        Fc_sol = Fc_sol.astype(np.float32)
        ddxc_sol = ddxc_sol.astype(np.float32)
        ddq_sol = ddq_sol.astype(np.float32)
 
        # Compute mass matrix and bias forces
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        
        # print(f"Fc_sol shape: {Fc_sol.shape}, ddxc_sol shape: {ddxc_sol.shape}, ddq_sol shape: {ddq_sol.shape}, M shape: {M.shape}, B shape: {B.shape}, S shape: {np.linalg.pinv(S.T).shape}, Jc shape: { self.Jc_id .shape}")
        self.tau = np.linalg.pinv(S.T) @ (M @ (np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + ddq_sol) + B - self.Jc_id.T @ Fc_sol)
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
        self.ErrorPlotting.tau_data_id.append(self.tau)
        # print(f"tau shape: {self.tau.T}")
        
    def send_command_api(self):
        """
        Send the computed torques to the robot.
        """
        # print("Sending command API...")
        # ks = 1000 # spring constant
        # dq_m = self.qd.reshape(-1,1) + self.tau.reshape(-1,1) / ks

        # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(dq_m), self.change_q_order(self.dqd), self.change_q_order(self.tau))
        self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd.copy()), self.change_q_order(self.dqd.copy()), self.change_q_order(self.tau.copy()))
        

        


 
    def plot_error_id(self):
        

        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.q_desired_data, 
                                        self.ErrorPlotting.q_current_data,
                                        self.ErrorPlotting.q_error_data,
                                        "qd_send")
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.dq_desired_data, 
                                        self.ErrorPlotting.dq_current_data,
                                        self.ErrorPlotting.dq_error_data,
                                        "dqd_send")
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.ddq_desired_data,
                                        self.ErrorPlotting.ddq_current_data,
                                        self.ErrorPlotting.ddq_error_data,
                                        "ddqd_send")
     
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_data, 
                                                         self.ErrorPlotting.x2_data,
                                                         self.ErrorPlotting.dx_b_data, 
                                                         "Body")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_data, 
                                                         self.ErrorPlotting.x3_data, 
                                                         self.ErrorPlotting.dx_sw_data, 
                                                         "Swing")
        
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_dot_data, 
                 self.ErrorPlotting.x2_dot_data,
                 self.ErrorPlotting.dx_b_dot_data, 
                 "Body Velocity.")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_dot_data, 
                 self.ErrorPlotting.x3_dot_data, 
                 self.ErrorPlotting.dx_sw_dot_data, 
                 "Swing_Velocity .")
        
        self.ErrorPlotting.plot_torque(self.ErrorPlotting.tau_data_ik, self.ErrorPlotting.tau_data_id, "joint toque Compare")
        # self.ErrorPlotting.plot_contact_acceleration(self.ErrorPlotting.ddxc_data, "contact foot acceleration")
        # self.ErrorPlotting.plot_contact_force(self.ErrorPlotting.Fc_data, "Fc")
        self.ErrorPlotting.plot_full_body_state(self.ErrorPlotting.ddq_diff_data, "ddq error")
   
        # self.ErrorPlotting.plot_torque(self.ErrorPlotting.torque_sensor_data,"joint toque sensor")
        plt.show()



# if __name__ == "__main__":
#     ChannelFactoryInitialize(1, "lo")
   
#     inv_dyn = InverseDynamic()
#     inv_dyn.ID_main()
