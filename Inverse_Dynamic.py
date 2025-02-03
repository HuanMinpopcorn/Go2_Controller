import numpy as np
from scipy import sparse
from Simulation import config
from Forward_Kinematics import ForwardKinematic
from Inverse_Kinematics import InverseKinematic
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import osqp
import mujoco


class InverseDynamic(InverseKinematic):
    def __init__(self):
        super().__init__()

        self.mu = 0.6  # Friction coefficient
        self.F_z_max = 250.0  # Max contact force
        self.tau_min = -50.0  # Torque limits
        self.tau_max = 50.0

        self.F_dim = 3 * len(self.contact_legs)  # Number of contact forces
        self.ddxc_dim = 3 * len(self.contact_legs)  # Contact accelerations
        self.ddq_dim = len(self.model.nv - 6)  # Joint accelerations (DOFs)

        # Weight matrices for cost function
        self.WF = sparse.csc_matrix(np.eye(self.F_dim))  # Weight for contact forces
        self.Wc = sparse.csc_matrix(np.eye(self.ddxc_dim))  # Weight for contact accelerations
        self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim))  # Weight for joint accelerations

        # Solver initialization
        self.prob = osqp.OSQP()

        # Placeholder for problem matrices and bounds
        self.A = None  # Combined constraint matrix
        self.l = None  # Lower bounds
        self.u = None  # Upper bounds
        self.P = None  # Hessian matrix
        self.q = None  # Gradient vector

        self.send_command_api()

    def whole_body_dynamics_constraint(self):
        """
        Formulate the whole-body dynamics constraints.
        """
        print("Starting whole-body dynamics constraint...")

        # Compute mass matrix and bias forces
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        tau_cmd = np.linalg.pinv(S.T) @ (M @ np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + B - self.data.qfrc_inverse.reshape(-1, 1))

        # J_contact  
        A1 = sparse.csc_matrix(-self.J1.T)  # Transposed contact Jacobian
        A2 = sparse.csc_matrix((self.J1.shape[1], self.J1.shape[0]))  # Placeholder
        A3 = sparse.csc_matrix(M)  # Mass matrix in sparse format
        A_matrix = sparse.hstack([A1, A2, A3])  # Constraint matrix
        dynamics_u = tau_cmd - B - M @ self.ddq_cmd  # Equality constraint RHS
        dynamics_l = dynamics_u  # Equality constraint bounds
        print("Whole-body dynamics constraint formulated.")
        return A_matrix, dynamics_l, dynamics_u

    def kinematic_constraints(self):
        """
        Formulate the kinematic constraints.
        """
        print("Formulating kinematic constraints...")

        A1 = sparse.csc_matrix((self.F_dim, self.F_dim))  # Placeholder
        A2 = sparse.csc_matrix(np.eye(self.ddxc_dim))  # Identity matrix
        A3 = -self.J1  # Negative Jacobian
        A_matrix = sparse.hstack([A1, A2, A3],format='csc')  # Constraint matrix
        # print(f"J_contact: {self.J_contact.shape}, ddq_dik: {self.ddq_dik.shape}, dJ_contact: {self.dJ_contact.shape}, qvel: {self.qvel.shape}")
        l = self.J1 @ self.ddq_cmd + self.J1_dot @ self.qvel.reshape(-1,1)  # Lower bound
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
        S = sparse.block_diag([S_single] * len(self.contact_legs))
        F_r_max = (np.ones(S.shape[0]) * self.F_z_max).reshape(-1, 1)
        A1 = sparse.csc_matrix(S)
        A2 = sparse.csc_matrix((S.shape[0], self.ddxc_dim))
        A3 = sparse.csc_matrix((S.shape[0], self.ddq_dim))
        A_react = sparse.hstack([A1, A2, A3],format='csc')
        l_react = -np.inf * np.ones(S.shape[0]).reshape(-1, 1)
        u_react = F_r_max
        return A_react, l_react, u_react
    
    def compute_contact_force_direction_constraints(self):
        Fz_single = np.array([
            [0, 0,  1],     # -Fz ≤ 0 → Fz ≥ 0
        ])
        Fz = sparse.block_diag([Fz_single] * len(self.contact_legs))

        A1 = sparse.csc_matrix(Fz)
        A2 = sparse.csc_matrix((Fz.shape[0], self.ddxc_dim))
        A3 = sparse.csc_matrix((Fz.shape[0], self.ddq_dim))
        A_matrix = sparse.hstack([A1, A2, A3],format='csc')
        l = np.zeros(Fz.shape[0]).reshape(-1, 1)
        u = np.ones(Fz.shape[0]).reshape(-1, 1) * np.inf
        return A_matrix, l, u

    def setup_cost_function(self):
        """
        Define the cost function.
        """
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
        #
    def solve(self):
        """
        Solve the optimization problem usinga OSQP.
        """
        self.setup_cost_function()
        self.combine_constraints()

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
        return  Fc_sol, ddxc_sol, ddq_sol

    def compute_torque(self):
        """
        Compute joint torques using the inverse dynamics solution.
        """
        print("Computing torques...")
        Fc_sol, ddxc_sol, ddq_sol = self.solve()
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms
        S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        
        self.tau = np.linalg.pinv(S.T) @ (M @ (np.vstack((np.zeros((6, 1)), self.ddq_cmd)) + ddq_sol) + B - self.J1.T @ Fc_sol)
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
        
    def send_command_api(self):
        """
        Send the computed torques to the robot.
        """
        print("Sending command API...")
        self.compute_torque()
        self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.change_q_order(tau))

from Send_motor_cmd import send_motor_commands
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
   
    inv_dyn = InverseDynamic()
    # inv_dyn.send_command_api()
    # cmd.move_to_initial_position()
    # inv_dyn.start_joint_updates()

    # inv_dyn.compute_torque()
    # inv_dyn.calculate()

