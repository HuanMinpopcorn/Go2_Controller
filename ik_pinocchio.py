import numpy as np
from scipy import sparse
import osqp
import pinocchio as pin
import Simulation.config as config

class WholeBodyController:
    def __init__(self, model, data, contact_frames, mu=0.7, F_z_max=100, tau_limits=(-100, 100)):
        """
        Initialize the Whole Body Controller.

        Args:
            model: Pinocchio model instance.
            data: Pinocchio data instance.
            contact_frames: List of frame IDs for contact points.
            mu: Friction coefficient.
            F_z_max: Maximum vertical contact force.
            tau_limits: Tuple (tau_min, tau_max) for joint torque limits.
        """
        self.model = model
        self.data = data
        self.contact_frames = contact_frames  # List of contact frame IDs
        self.mu = mu
        self.F_z_max = F_z_max
        self.tau_min, self.tau_max = tau_limits

        # Dimensions
        self.num_contacts = len(contact_frames)
        self.F_dim = 3 * self.num_contacts  # Contact forces
        self.ddq_dim = model.nv  # Joint accelerations

        # Weight matrices for cost function
        self.WF = sparse.csc_matrix(np.eye(self.F_dim))  # Weight for contact forces
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
        Compute the mass matrix and bias forces (Coriolis + gravity).

        Returns:
            M: Sparse mass matrix.
            bias_forces: Bias forces.
        """
        M = pin.crba(self.model, self.data, self.data.qpos)  # Compute mass matrix
        bias_forces = pin.rnea(self.model, self.data, self.data.qpos, self.data.qvel, np.zeros(self.ddq_dim))
        return sparse.csc_matrix(M), bias_forces

    def compute_contact_jacobian(self):
        """
        Compute the contact Jacobian for all contact frames.

        Returns:
            Jc: Sparse contact Jacobian matrix.
        """
        Jc = []
        for frame_id in self.contact_frames:
            J_frame = pin.computeFrameJacobian(self.model, self.data, self.data.qpos, frame_id, pin.LOCAL)
            Jc.append(J_frame[:3, :])  # Only consider linear components
        return sparse.vstack(Jc)

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
        A_dyn = sparse.hstack([-Jc.T, M])
        l_dyn = -bias_forces
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

        A_react = sparse.hstack([S, sparse.csc_matrix((S.shape[0], self.Wddq.shape[1]))])
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
            sparse.eye(self.ddq_dim)  # Identity matrix for joint torques
        ])
        l_tau = np.full(self.ddq_dim, self.tau_min)
        u_tau = np.full(self.ddq_dim, self.tau_max)
        return A_tau, l_tau, u_tau

    def setup_cost_function(self):
        """
        Define the cost function.
        """
        self.P = sparse.block_diag([self.WF, self.Wddq])
        self.q = np.zeros(self.F_dim + self.ddq_dim)

    def combine_constraints(self, M, bias_forces, Jc):
        """
        Combine all constraints into a single matrix.
        """
        A_dyn, l_dyn, u_dyn = self.compute_dynamics_constraints(M, bias_forces, Jc)
        A_react, l_react, u_react = self.compute_reaction_force_constraints()
        A_tau, l_tau, u_tau = self.compute_torque_constraints()

        self.A = sparse.vstack([A_dyn, A_react, A_tau])
        self.l = np.hstack([l_dyn, l_react, l_tau])
        self.u = np.hstack([u_dyn, u_react, u_tau])

    def solve(self):
        """
        Solve the optimization problem using OSQP.
        """
        M, bias_forces = self.compute_mass_matrix_and_bias()
        Jc = self.compute_contact_jacobian()

        self.setup_cost_function()
        self.combine_constraints(M, bias_forces, Jc)

        self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
        result = self.prob.solve()

        if result.info.status != 'solved':
            raise ValueError(f"OSQP failed to solve the problem: {result.info.status}")

        ddq_sol = result.x[-self.ddq_dim:]
        Fc_sol = result.x[:self.F_dim]
        return ddq_sol, Fc_sol


# Example Usage
if __name__ == "__main__":
    # Load the URDF model
    model = pin.buildModelFromUrdf(config.ROBOT_SCENE) # Load the URDF model
    data = model.createData()

    

    # Initialize the controller
    controller = WholeBodyController(model, data, contact_frames)

    # Solve the optimization problem
    ddq_sol, Fc_sol = controller.solve()
    print("Joint accelerations:", ddq_sol)
    print("Contact forces:", Fc_sol)