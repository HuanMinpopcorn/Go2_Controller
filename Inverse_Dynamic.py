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

        # Problem dimensions
        self.F_dim = 3 * len(self.contact_legs)  # Number of contact forces
        self.ddxc_dim = 3 * len(self.contact_legs)  # Contact accelerations
        self.ddq_dim = len(self.data.qacc)  # Joint accelerations (DOFs)

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

    def whole_body_dynamics_constraint(self):
        """
        Formulate the whole-body dynamics constraints.
        """
        print("Starting whole-body dynamics constraint...")

        # Compute mass matrix and bias forces
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1, 1)  # Nonlinear terms

        # Contact Jacobian
        self.required_jacobian = self.get_required_jacobian()
        Jc = sparse.csc_matrix(self.required_jacobian["contact_leg"])  # Jacobian

        tau_cmd = np.vstack([np.zeros((6, 1)), self.tau])  # Commanded torques
        ddq_cmd = np.vstack([np.zeros((6, 1)), self.ddqd])  # Desired accelerations

        # Dynamics constraints
        A1 = -Jc.T  # Transposed contact Jacobian
        A2 = sparse.csc_matrix((Jc.shape[1], Jc.shape[0]))  # Placeholder
        A3 = sparse.csc_matrix(M)  # Mass matrix in sparse format
        A_matrix = sparse.hstack([A1, A2, A3])  # Constraint matrix
        dynamics_u = tau_cmd - B - M @ ddq_cmd  # Equality constraint RHS
        dynamics_l = dynamics_u  # Equality constraint bounds

        print("Whole-body dynamics constraint formulated.")
        return A_matrix, dynamics_l, dynamics_u

    def kinematic_constraints(self):
        """
        Formulate the kinematic constraints.
        """
        print("Formulating kinematic constraints...")
        self.required_jacobian = self.get_required_jacobian()
        Jc = sparse.csc_matrix(self.required_jacobian["contact_leg"])

        ddq_cmd = np.vstack([np.zeros((6, 1)), self.ddqd])
        Jc_dot = np.vstack([self.get_jacobian_dot(leg)["J_pos"] for leg in self.contact_legs])

        A1 = sparse.csc_matrix((Jc.shape[0], Jc.shape[0]))  # Placeholder
        A2 = sparse.eye(Jc.shape[0])  # Identity matrix
        A3 = -Jc  # Negative Jacobian
        A_matrix = sparse.hstack([A1, A2, A3])  # Constraint matrix
        l = Jc @ ddq_cmd + Jc_dot @ self.dq  # Lower bound
        u = l  # Upper bound for equality constraint

        print("Kinematic constraints formulated.")
        return A_matrix, l, u

    def reaction_force_constraints(self):
        """
        Formulate the reaction force constraints.
        """
        print("Formulating reaction force constraints...")
        mu = 0.7  # Friction coefficient
        num_contacts = len(self.contact_legs)  # Number of contact points
        F_z_max = 100  # Max vertical force

        # Friction cone constraints
        S_single = np.array([
            [1,  0, -mu],
            [-1, 0, -mu],
            [0,  1, -mu],
            [0, -1, -mu]
        ])

        S = sparse.block_diag([S_single] * num_contacts)
        F_r_max = np.tile([F_z_max] * 4, num_contacts)

        A1 = S
        A2 = sparse.csc_matrix((S.shape[0], self.Wc.shape[1]))
        A3 = sparse.csc_matrix((S.shape[0], self.Wddq.shape[1]))
        A_matrix = sparse.hstack([A1, A2, A3])

        l = -np.inf * np.ones(S.shape[0])  # No lower bound
        u = F_r_max  # Upper bound

        print("Reaction force constraints formulated.")
        return A_matrix, l, u

    def setup_cost_function(self):
        """
        Define the cost function for the optimization problem.
        """
        print("Setting up cost function...")
        P_F = sparse.block_diag([self.WF, self.Wc, self.Wddq])
        q_F = np.zeros(self.F_dim + self.ddxc_dim + self.ddq_dim)
        self.P = P_F
        self.q = q_F
        print("Cost function set.")

    def solve(self):
        """
        Solve the optimization problem using OSQP.
        """
        self.setup_cost_function()
        self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
        result = self.prob.solve()

        if result.info.status != 'solved':
            raise ValueError("OSQP failed to find a solution.")
        return result.x[:self.ddq_dim], result.x[self.ddq_dim:]

    def compute_torque(self):
        """
        Compute joint torques using the inverse dynamics solution.
        """
        print("Computing torques...")
        ddq_sol, Fc_sol = self.solve()
        M = sparse.csc_matrix(self.data.qM)
        B = self.data.qfrc_bias
        Jc = sparse.csc_matrix(self.J1)
        tau = M @ ddq_sol + B - Jc.T @ Fc_sol
        return tau

