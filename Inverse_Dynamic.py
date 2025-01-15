import numpy as np
from scipy import sparse
from Simulation import config 
from Forward_Kinematics import ForwardKinematic 
from Inverse_Kinematics import InverseKinematic 
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import osqp
import threading
import mujoco
# from mujoco_py import functions

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


class InverseDynamic(InverseKinematic):
    def __init__(self):
        super().__init__()

        # Problem dimensions
        F_dim = 3 * len(self.contact_legs)  # Number of contact forces
        ddxc_dim = 3 * len(self.contact_legs)  # Number of contact accelerations
        ddq_dim = len(self.data.qacc)  # Joint accelerations (degrees of freedom)

        # Weight matrices for cost function
        self.WF = sparse.csc_matrix(np.eye(F_dim))  # Weight for contact forces
        self.Wc = sparse.csc_matrix(np.eye(ddxc_dim))  # Weight for contact accelerations
        self.Wddq = sparse.csc_matrix(np.eye(ddq_dim))  # Weight for joint accelerations

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
        print("Starting whole body dynamics constraint...")
        # Compute mass matrix and bias forces
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # Mass matrix
        B = self.data.qfrc_bias.reshape(-1,1)  # Nonlinear terms (bias forces, gravity)

        # Contact Jacobian
        self.required_jacobian = self.get_required_jacobian()
        Jc = sparse.csc_matrix(self.required_jacobian["contact_leg"])  # Ensure J1 is correct
        # Fc = self.data.efc_force  # External contact forces
        # print("Debug: Jc shape", Jc.shape)
        # print("Debug: Fc shape", Fc)
        # print("Debug: Fc shape", Fc.shape)
        tau_cmd = np.vstack([np.zeros((6, 1)), self.tau])  # Commanded torques

        # # Optional: Validate using mj_inverse
        # mujoco.mj_inverse(self.model, self.data)
        # ddq_mj_inverse = self.data.qacc # real ddq from sensor data
        ddq_cmd = np.vstack([np.zeros((6, 1)), self.ddqd]) # desired ddq from IK
        # print("Debug: ddq (Mujoco) vs. ddq_des", ddq_mj_inverse, self.ddqd)

        # Dynamics constraints
        A1 = -Jc.T # Adjust matrix dimensions as needed
        A2 = sparse.csc_matrix((Jc.shape[1], Jc.shape[0]))  # Placeholder
        A3 = sparse.csc_matrix(M)  # Mass matrix in sparse format
        A_matrix = sparse.hstack([A1, A2, A3])  # Constraint matrix
        dynamics_u = tau_cmd - B - M @ ddq_cmd  # Equality constraint RHS
        dynamics_l = dynamics_u  # Equality constraint bounds
        # print(tau_cmd.shape, B.shape, M.shape, ddq_cmd.shape)
        # print(A_matrix.shape, dynamics_l.shape, dynamics_u.shape)
        return A_matrix, dynamics_l, dynamics_u

    def kinematic_constraints(self):
        """
        Formulate the kinematic constraints.
        """
        # Placeholder for kinematic constraints
        # ddxc = Jc * ddq + dJc * dq
        # Jc = sparse.csc_matrix(self.J1)
        # ddq = self.ddqd
        # dq = self.dq
        # A1 = 
        # A2 =
        # A3 =
        # A_matrix = sparse.hstack([A1, A2, A3])
        # l =
        # u =
        # return A_matrix, l, u
        pass
    def reaction_force_constraints(self):
        """
        Formulate the reaction force constraints.
        """
        # Placeholder for reaction force constraints
        # Fc = self.Fc
        # A1 = 
        # A2 =
        # A3 =
        # A_matrix = sparse.hstack([A1, A2, A3])
        # l =
        # u =
        # return A_matrix, l, u
        pass
    def acceleration_constraints(self):
        """
        Formulate the acceleration constraints.
        """
        # Placeholder for acceleration constraints
        # ddq = self.ddqd
        # A1 = 
        # A2 =
        # A3 =
        # A_matrix = sparse.hstack([A1, A2, A3])
        # l =
        # u =
        # return A_matrix, l, u
        pass
    def torque_constraints(self):
        """
        Formulate the torque constraints.
        """
        # Placeholder for torque constraints
        # tau = self.tau
        # A1 = 
        # A2 =
        # A3 =
        # A_matrix = sparse.hstack([A1, A2, A3])
        # l =
        # u =
        # return A_matrix, l, u
        pass
    
        

    def setup_constraints(self):
        """
        Combine constraints for the optimization problem.
        """
        dynamics_matrix, dynamics_rhs = self.whole_body_dynamics()

        # Placeholder for other constraints (e.g., contact wrenches, torque limits)
        # ...

        # Set A matrix, bounds l and u
        self.A = dynamics_matrix  # Expand with other constraints
        self.l = dynamics_rhs
        self.u = dynamics_rhs  # For equality constraints, l = u

    def setup_cost_function(self):
        """
        Define the cost function for the optimization problem.
        """
        # Combine weights for the quadratic cost function
        F_dim = self.WF.shape[0]
        ddxc_dim = self.Wc.shape[0]
        ddq_dim = self.Wddq.shape[0]

        P_F = sparse.block_diag([self.WF, self.Wc, self.Wddq])  # Cost Hessian
        q_F = np.zeros(F_dim + ddxc_dim + ddq_dim)  # Linear cost vector

        self.P = P_F
        self.q = q_F

    def solve(self):
        """
        Solve the optimization problem using OSQP.
        """
        self.setup_constraints()
        self.setup_cost_function()

        self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
        result = self.prob.solve()

        if result.info.status != 'solved':
            raise ValueError("OSQP failed to find a solution.")

        solution = result.x
        return solution[:len(self.data.qacc)], solution[len(self.data.qacc):]

    def compute_torque(self):
        """
        Compute joint torques using the inverse dynamics solution.
        """
        ddq_sol, Fc_sol = self.solve()

        # Using M * ddq + B - Jc.T * Fc = tau
        M = sparse.csc_matrix(self.data.qM)
        B = self.data.qfrc_bias
        Jc = sparse.csc_matrix(self.J1)

        tau = M @ ddq_sol + B - Jc.T @ Fc_sol
        return tau
    def ID_calculate(self):
        """
        Calculate the joint torques using inverse dynamics.
        """
        

        # Initialize the loop variables
        i = 0
        trail = 0
        running_time = 0
        
        print("Starting ID Trot Gait...")

        x_sw = self.compute_desired_swing_leg_trajectory()
        leg_pair_in_swing = True
        while True:
            i = (i + 1) % self.K  # Loop over the swing cycle duration
            print("Starting ID Trot Gait...")
            if i == 0:
                # over one cycle
                self.transition_legs()
                x_sw = self.compute_desired_swing_leg_trajectory()
                # print("Transitioning legs...")
            
            x_b = self.compute_desired_body_state()             # update the body state for the next cycle
            joint_angles = self.joint_state_reader.joint_angles # get the current joint angles
            
           
            self.required_state = self.get_required_state()          # get the required state for IK
            self.x1 = self.required_state["contact_leg"]
            self.x2 = self.required_state["body"]
            self.x3 = self.required_state["swing_leg"]
            # x1 : contact leg positions, x2 : body state, x3 : swing leg positions

            self.required_jacobian = self.get_required_jacobian()
            self.J1 = self.required_jacobian["contact_leg"]
            self.J2 = self.required_jacobian["body"]
            self.J3 = self.required_jacobian["swing_leg"]
            # J1 : contact leg Jacobian, J2 : body Jacobian, J3 : swing leg Jacobian

            # calculate the desired joint angles and joint velocities
            self.qd, self.dqd = self.InverseKinematic_formulation(self.J1, self.J2, self.J3, 
                                                                self.x1, self.x2, self.x3, 
                                                                self.kb, self.kc, self.ks, 
                                                                x_b, x_sw, 
                                                                i, joint_angles)
            # calculate ctrl output through PD controller 
            dq_error = self.kp * (self.change_q_order(self.qd) - self.data.sensordata[:12])
            dq_dot = self.kd * (self.change_q_order(self.dqd) - self.data.sensordata[12:24])
            
            
            self.ErrorPlotting.dq_error_data.append(dq_error)
            self.ErrorPlotting.dq_dot_data.append(dq_dot)
            self.ErrorPlotting.output_data.append(dq_error + dq_dot)
           
            self.ddqd = dq_error + dq_dot # desired joint acceleration
            self.q = self.change_q_order(self.data.sensordata[:12]) # current joint angles in kinematic order
            self.dq = self.change_q_order(self.data.sensordata[12:24]) # current joint velocities in kinematic order

            # TODO update the toque output for the Inverse Dynamics
            self.tau = self.compute_torque()
            
            # send qd and dqd to the API 
            self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.qd), self.change_q_order(self.dqd), self.tau)

            # data storage for plotting
            self.ErrorPlotting.q_desired_data.append(self.change_q_order(self.qd))
            self.ErrorPlotting.q_current_data.append(self.change_q_order(self.q ))
            # update running steps 
            trail += 1
            if trail > 5000: # 5000 steps 
                break
        # call the error plotting class for plotting the data
        self.ErrorPlotting.plot_api_value(self.ErrorPlotting.dq_error_data, self.ErrorPlotting.dq_dot_data, self.ErrorPlotting.output_data)
        self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q3_dot_data, "q3_dot")
        self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q2_dot_data, "q2_dot")
        self.ErrorPlotting.plot_q_dot(self.ErrorPlotting.q1_dot_data , "q1_dot")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xb_data, self.ErrorPlotting.x2_data, self.ErrorPlotting.dx_b_data, "Body")
        self.ErrorPlotting.plot_state_error_trajectories(self.ErrorPlotting.xw_data, self.ErrorPlotting.x3_data, self.ErrorPlotting.dx_sw_data, "Swing")
        self.ErrorPlotting.plot_q_error(self.ErrorPlotting.q_desired_data, self.ErrorPlotting.q_current_data) 
        plt.show()


if __name__ == "__main__":
    # Initialize the channel factory
    ChannelFactoryInitialize(1, "lo")
    # Create an instance of the InverseDynamic class
    id = InverseDynamic()
    id.start_joint_updates()
    id.cmd.move_to_initial_position()
    id.initialize()
    id.whole_body_dynamics_constraint()
    # Calculate the joint torques using inverse dynamics
    # id.ID_calculate()
    # Close the channel factory
    # ChannelFactoryClose()