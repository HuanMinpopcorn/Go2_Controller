import numpy as np
from scipy import sparse
import time

from Inverse_Kinematics import InverseKinematic

import osqp
import mujoco
import matplotlib.pyplot as plt
from tqdm import tqdm

class InverseDynamic(InverseKinematic):
    def __init__(self,swing_legs, contact_legs):
        super().__init__(swing_legs, contact_legs)

        self.mu = 0.6  # Friction coefficient
        self.F_z_max = 250.0  # Max contact force
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
        self.start_joint_updates()
        
    def contact_jacobian(self):
        nv = self.model.nv
        # ncon = self.data.ncon
        ncon = len(self.contact_legs)
       

        self.F_dim = 3 * ncon  # Number of contact forces
        self.ddxc_dim = 3 * ncon  # Contact accelerations
        self.ddq_dim = self.model.nv  # Joint accelerations (DOFs) 
        self.number_of_contacts = ncon
        # print(f"number_of_contacts: {self.number_of_contacts}")

     
        # Preallocate full contact Jacobian: 3 rows per contact
        Jc = np.zeros((3 * ncon, nv))
        Jc_dot = np.zeros((3 * ncon, nv))
        # Determine the number of contacts
        if ncon == 0:
            print("No contact points detected.")
            return np.zeros((0, nv)), np.zeros((0, nv))
        elif ncon == 2:
            self.state_machine("swing")
            Jc = self.J1
            Jc_dot = self.J1_dot
        elif ncon == 4:
            self.state_machine("transition")
            Jc = np.vstack((self.J1, self.J3))
            Jc_dot = np.vstack((self.J1_dot, self.J3_dot))

        return Jc, Jc_dot

    
    def whole_body_dynamics_constraint(self):
        """
        Formulate the whole-body dynamics constraints.
        """
        # print("Starting whole-body dynamics constraint...")

        # Compute mass matrix and bias forces
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        # S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))

        self.Jc, self.Jc_dot = self.contact_jacobian()
      
        
        # self.tau = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - self.data.qfrc_inverse.reshape(-1, 1))
        tau_cmd = np.vstack((np.zeros((6, 1)), self.tau.reshape(-1, 1)))    
        # J_contact  
        A1 = sparse.csc_matrix(-self.Jc.T)  # Transposed contact Jacobian
        A2 = sparse.csc_matrix((self.Jc.shape[1], self.Jc.shape[0]))  # Placeholder
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
        # print("Formulating kinematic constraints...")

        A1 = sparse.csc_matrix(np.zeros((self.F_dim, self.F_dim)))  # Zero matrix
        A2 = sparse.csc_matrix(np.eye(self.ddxc_dim))  # Identity matrix
        A3 = -self.Jc  # Negative Jacobian
        A_matrix = sparse.hstack([A1, A2, A3],format='csc')  # Constraint matrix
        # print(f"J_contact: {self.J_contact.shape}, ddq_dik: {self.ddq_dik.shape}, dJ_contact: {self.dJ_contact.shape}, qvel: {self.qvel.shape}")
        l = self.Jc @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + self.Jc_dot @ self.data.qvel.copy().reshape(-1,1)  # Lower bound
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


    def solve(self):
        """
        Solve the optimization problem usinga OSQP.
        """
        
        self.combine_constraints()
        self.setup_cost_function()


        self.prob = osqp.OSQP()
        # print(f"P shape: {self.P.shape}, q shape: {self.q.shape}, A shape: {self.A.shape}, l shape: {self.l.shape}, u shape: {self.u.shape}")
        self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
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
        Fc_sol, ddxc_sol, ddq_sol = self.solve()
        Fc_sol = Fc_sol.astype(np.float32)
        ddxc_sol = ddxc_sol.astype(np.float32)
        ddq_sol = ddq_sol.astype(np.float32)
        if ddq_sol is None:
            print("Warning: ddq_sol is None, skipping torque computation.")
            return
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        self.ErrorPlotting.tau_data.append(self.tau)
        # print(f"Fc_sol shape: {Fc_sol.shape}, ddxc_sol shape: {ddxc_sol.shape}, ddq_sol shape: {ddq_sol.shape}, M shape: {M.shape}, B shape: {B.shape}, S shape: {S.shape}")
        self.tau = np.linalg.pinv(S.T) @ (M @ (np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + ddq_sol) + B - self.Jc.T @ Fc_sol)
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
        
    def send_command_api(self):
        """
        Send the computed torques to the robot.
        """
        # print("Sending command API...")
        ks = 100
        dq_m = self.qd.reshape(-1,1) + self.tau.reshape(-1,1) / ks
        # print(dq_m.shape)
        # self.qd = np.zeros((self.model.nv, 1))
        # self.dqd = np.zeros((self.model.nv, 1))
        # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(dq_m), self.change_q_order(self.dqd), self.change_q_order(self.tau))
        self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.change_q_order(self.tau))

        
        # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.change_q_order(self.tau))
    def send_command_ik(self):
        """
        Send the computed torques to the robot.
        """
        # print("Sending command API...")
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        self.tau = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - self.data.qfrc_inverse.reshape(-1, 1))
        # self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.change_q_order(self.tau))
        self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd))


 
    def plot_error_id(self):
        
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_dot_data, 
                 self.ErrorPlotting.x2_dot_data,
                 self.ErrorPlotting.dx_b_dot_data, 
                 "Body Velocity.")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_dot_data, 
                 self.ErrorPlotting.x3_dot_data, 
                 self.ErrorPlotting.dx_sw_dot_data, 
                 "Swing_Velocity .")
        self.ErrorPlotting.plot_torque(self.ErrorPlotting.tau_data,"joint toque")
        self.ErrorPlotting.plot_contact_acceleration(self.ErrorPlotting.ddxc_data, "contact foot acceleration")
        self.ErrorPlotting.plot_contact_force(self.ErrorPlotting.Fc_data, "Fc")
        self.ErrorPlotting.plot_full_body_state(self.ErrorPlotting.ddq_diff_data, "ddq error")
   
        self.ErrorPlotting.plot_torque(self.ErrorPlotting.torque_sensor_data,"joint toque sensor")
        plt.show()

    def state_machine(self, phase):
        if phase == "double_standing":
            self.WF = sparse.csc_matrix(np.diag([5, 5, 0.5, 1, 1, 0.01]) * 100)
            self.Wc = sparse.csc_matrix(np.diag([10**-3, 10**-3, 10**-3, 10**3, 10**3, 10**3]))
            self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim) * 100)
        elif phase == "transition":
            self.WF = sparse.csc_matrix(np.diag([1, 1, 0.5, 1, 1, 0.01,1, 1, 0.5, 1, 1, 0.01]))
            self.Wc = sparse.csc_matrix(np.diag([10**-3, 10**-3, 10**-3, 10**3, 10**3, 10**3, 10**-3, 10**-3, 10**-3, 10**3, 10**3, 10**3]))
            self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim) * 100)
        
        elif phase == "swing":
            # self.WF = sparse.csc_matrix(np.diag([5, 5, 1, 5, 5, 1]))
            # self.Wc = sparse.csc_matrix(np.diag([10**3, 10**3, 10**3, 10**3, 10**3, 10**3]))
            # self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim) * 50)
            self.WF = sparse.csc_matrix(np.diag([1, 1, .5, 1, 1, 1])* 100)
            self.Wc = sparse.csc_matrix(np.diag([1,1,1,1,1,1]) * 10**3)
            self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim) * 10)
        else:
            raise ValueError("Invalid phase")
# if __name__ == "__main__":
#     ChannelFactoryInitialize(1, "lo")
   
#     inv_dyn = InverseDynamic()
#     inv_dyn.ID_main()
