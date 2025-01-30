import numpy as np
from scipy import sparse
import mujoco
import osqp
import Inverse_Kinematics as ik 
import Forward_Kinematics as fk


class WholeBodyController:
    def __init__(self, model, data, contact_bodies, mu=0.7, F_z_max=100, tau_limits=(-100, 100)):
        """
        Initialize the Whole Body Controller.

        Args:
            model: MuJoCo model instance.
            data: MuJoCo data instance.
            contact_bodies: List of contact body IDs.
            mu: Friction coefficient.
            F_z_max: Maximum vertical contact force.
            tau_limits: Tuple (tau_min, tau_max) for joint torque limits.
        """
        self.model = model
        self.data = data
        self.contact_bodies = contact_bodies
        self.mu = mu
        self.F_z_max = F_z_max
        self.tau_min, self.tau_max = tau_limits

        # Dimensions
        self.num_contacts = len(contact_bodies)
        self.F_dim = 3 * self.num_contacts  # Contact forces
        self.ddxc_dim = 3 * self.num_contacts  # Contact accelerations
        self.ddq_dim = model.nv  # Joint accelerations

        # Weight matrices for cost function
        self.WF = sparse.csc_matrix(np.eye(self.F_dim))  # Weight for contact forces
        self.Wc = sparse.csc_matrix(np.eye(self.ddxc_dim))  # Weight for contact accelerations
        self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim))  # Weight for joint accelerations

        # Optimization solver
        self.prob = osqp.OSQP()

        # Placeholder for problem matrices
        self.A = None
        self.l = None
        self.u = None
        self.P = None
        self.q = None

    def compute_mass_matrix_and_bias(self):
        """
        Compute the mass matrix and bias forces (b + g).

        Returns:
            M: Sparse mass matrix.
            bias_forces: Bias forces including Coriolis and gravity.
        """
        # Mass matrix
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M_sparse = sparse.csc_matrix(M)

        # Bias forces
        bias_forces = self.data.qfrc_bias.reshape(-1, 1)
        return M_sparse, bias_forces

    def compute_contact_jacobian(self):
        """
        Compute the contact Jacobian for all contact points.

        Returns:
            Jc: Sparse contact Jacobian matrix.
        """
        Jc = np.zeros((3 * self.num_contacts, self.model.nv))
        for i, contact_id in enumerate(self.contact_bodies):
            mujoco.mj_jacBody(self.model, self.data, Jc[i*3:(i+1)*3, :], None, contact_id)
        return sparse.csc_matrix(Jc)

    def compute_kinematics_constraints(self, Jc):
        """
        Formulate kinematic constraints.

        Args:
            Jc: Contact Jacobian matrix.

        Returns:
            A_kin, l_kin, u_kin: Kinematic constraint matrix and bounds.
        """
        # Numerically compute Jacobian derivative
        dt = self.model.opt.timestep
        Jc_dot = (Jc - getattr(self, "prev_Jc", Jc)) / dt
        self.prev_Jc = Jc

        # Placeholder for desired joint accelerations
        ddq_cmd = np.zeros((self.ddq_dim, 1))
        dq = self.data.qvel  # Joint velocities

        # Constraint formulation
        A1 = sparse.csc_matrix((Jc.shape[0], Jc.shape[0]))  # Placeholder
        A2 = sparse.eye(Jc.shape[0])  # Identity matrix
        A3 = -Jc  # Negative Jacobian
        A_kin = sparse.hstack([A1, A2, A3])
        l_kin = Jc @ ddq_cmd + Jc_dot @ dq
        u_kin = l_kin  # Equality constraint
        return A_kin, l_kin, u_kin

    def compute_dynamics_constraints(self, M, bias_forces, Jc):
        """
        Formulate whole-body dynamics constraints.

        Args:
            M: Mass matrix.
            bias_forces: Bias forces.
            Jc: Contact Jacobian matrix.

        Returns:
            A_dyn, l_dyn, u_dyn: Dynamics constraint matrix and bounds.
        """
        tau_cmd = np.zeros((self.ddq_dim, 1))  # Placeholder for joint torques
        A_dyn = sparse.hstack([-Jc.T, sparse.csc_matrix(M)])
        l_dyn = tau_cmd - bias_forces
        u_dyn = l_dyn
        return A_dyn, l_dyn, u_dyn

    def compute_reaction_force_constraints(self):
        """
        Formulate reaction force constraints (friction cone).

        Returns:
            A_react, l_react, u_react: Reaction force constraint matrix and bounds.
        """
        # Friction cone
        S_single = np.array([
            [1,  0, -self.mu],
            [-1, 0, -self.mu],
            [0,  1, -self.mu],
            [0, -1, -self.mu]
        ])
        S = sparse.block_diag([S_single] * self.num_contacts)
        F_r_max = np.tile([self.F_z_max] * 4, self.num_contacts)

        A_react = sparse.hstack([S, sparse.csc_matrix((S.shape[0], self.Wc.shape[1])), sparse.csc_matrix((S.shape[0], self.Wddq.shape[1]))])
        l_react = -np.inf * np.ones(S.shape[0])
        u_react = F_r_max
        return A_react, l_react, u_react

    def compute_torque_constraints(self):
        """
        Formulate joint torque constraints.

        Returns:
            A_tau, l_tau, u_tau: Torque constraint matrix and bounds.
        """
        A_tau = sparse.hstack([
            sparse.csc_matrix((self.ddq_dim, self.F_dim)),  # No influence of forces
            sparse.csc_matrix((self.ddq_dim, self.ddxc_dim)),  # No influence of accelerations
            sparse.eye(self.ddq_dim)  # Identity matrix for joint torques
        ])
        l_tau = np.full(self.ddq_dim, self.tau_min)
        u_tau = np.full(self.ddq_dim, self.tau_max)
        return A_tau, l_tau, u_tau

    def setup_cost_function(self):
        """
        Define the cost function.
        """
        P_F = sparse.block_diag([self.WF, self.Wc, self.Wddq])
        q_F = np.zeros(self.F_dim + self.ddxc_dim + self.ddq_dim)
        self.P = P_F
        self.q = q_F

    def combine_constraints(self):
        """
        Combine all constraints into a single matrix.
        """
        M, bias_forces = self.compute_mass_matrix_and_bias()
        Jc = self.compute_contact_jacobian()

        A_dyn, l_dyn, u_dyn = self.compute_dynamics_constraints(M, bias_forces, Jc)
        A_kin, l_kin, u_kin = self.compute_kinematics_constraints(Jc)
        A_react, l_react, u_react = self.compute_reaction_force_constraints()
        A_tau, l_tau, u_tau = self.compute_torque_constraints()

        self.A = sparse.vstack([A_dyn, A_kin, A_react, A_tau])
        self.l = np.hstack([l_dyn, l_kin, l_react, l_tau])
        self.u = np.hstack([u_dyn, u_kin, u_react, u_tau])

    def solve(self):
        """
        Solve the optimization problem using OSQP.
        """
        self.setup_cost_function()
        self.combine_constraints()

        self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
        result = self.prob.solve()

        if result.info.status != 'solved':
            raise ValueError(f"OSQP failed to solve the problem: {result.info.status}")

        ddq_sol = result.x[-self.ddq_dim:]
        Fc_sol = result.x[:self.F_dim]
        return ddq_sol, Fc_sol


def main(): 
 

    # calculate the qcmd 
    ik.start_joint_updates() # start forward kinematics updates
    ik.cmd.move_to_initial_position() # move to initial position
    ik.initialize() # initialize the inverse kinematics
    model = ik.model 
    data = ik.data
    contact_bodies = ik.data.contact_bodies
    q_cmd = ik.calculate_q_cmd() # give the trajectory to the robot and get desired q_cmd
    wbc = WholeBodyController(model, data, contact_bodies, q_cmd)