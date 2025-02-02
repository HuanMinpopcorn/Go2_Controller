import numpy as np
import multiprocessing
import threading
import time
from scipy import sparse
import osqp
import pinocchio as pin
import Simulation.config as config

from scipy.spatial.transform import Rotation as R

from Forward_Kinematics import ForwardKinematic
from Inverse_Kinematics import InverseKinematic
from read_JointState import read_JointState
from read_TaskSpace import read_TaskSpace

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from Send_motor_cmd import send_motor_commands
from error_plotting import ErrorPlotting
import tqdm
import matplotlib.pyplot as plt
class WholeBodyController:
    def __init__(self, conn):
    # def __init__(self):
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
        # Add required parameters
        self.mu = 0.6  # Friction coefficient
        self.F_z_max = 250.0  # Max contact force
        self.tau_min = -50.0  # Torque limits
        self.tau_max = 50.0
        

        # running the ik 
        joint_composite = pin.JointModelComposite(2)
        # the virtual joint is add on the body frame of the robot
        joint_composite.addJoint(pin.JointModelFreeFlyer())
        
        self.model = pin.buildModelFromUrdf(config.urdf_path, joint_composite)
        self.data = self.model.createData()

  
        self.joint_state_reader = read_JointState()
        self.task_space_reader = read_TaskSpace()
        

        self.conn = conn
   
        # print(self.model.nv) # 18 
        # print(self.model.nq) # 19


        
        self.swing_legs = ["FL_foot", "RR_foot"]
        self.contact_legs = ["FL_foot", "RR_foot"] + ["FR_foot", "RL_foot"]
        self.num_contacts = len(self.contact_legs)
        print(self.num_contacts) # 4
        self.qpos = np.zeros(self.model.nq)
        self.qvel = np.zeros(self.model.nv)
        # print(self.qpos.shape)
        self.J_contact = np.zeros((3 * self.num_contacts, self.model.nv)) 
        self.dJ_contact = np.zeros((3 * self.num_contacts, self.model.nv))

        self.F_dim = 3 * self.num_contacts  # Contact forces
        self.ddxc_dim = 3 * self.num_contacts  # Contact accelerations
        self.ddq_dim = self.model.nv  # Joint accelerations
        
        # intialize the input tau_cmd and ddq
        self.ddq_ik = np.zeros((self.model.nv - 6, 1))
        self.dq_ik = np.zeros((self.model.nv - 6, 1))
        self.q_ik = np.zeros((self.model.nv - 6, 1))
        self.tau = np.zeros((self.model.nv - 6, 1))

        self.desire_body_position = [-0.013, 0.0, 0.266]

        # self.kp = 250 + 50 + 100
        # self.kd = 10 
        self.kp = 0
        self.kd = 0

        
        # Weight matrices for cost function
        self.WF = sparse.csc_matrix(np.eye(self.F_dim) * 1)  # Weight for contact forces
        self.Wc = sparse.csc_matrix(np.eye(self.ddxc_dim) * 1000)  # Weight for contact accelerations
        self.Wddq = sparse.csc_matrix(np.eye(self.ddq_dim)* 1)  # Weight for joint accelerations

  
        # Optimization solver
        self.prob = osqp.OSQP()

        # Placeholder for problem matrices
        self.A = None
        self.l = None
        self.u = None
        self.P = None
        self.q = None

        self.counter = 0

        self.update_thread = None  # Thread for continuous updates
        self.running = False  # Control flag for the thread

        # Initialize the API
        self.cmd = send_motor_commands()
        self.ErrorPlotting = ErrorPlotting()
        self.start_joint_updates()

    def start_joint_updates(self):
        """
        Start a thread to update joint angles.
        """
        # print("ID Starting joint updates...")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update_joint_angles)
            self.thread.start()

    def update_joint_angles(self):
        """
        Continuously updates the joint angles in the background.
        """
        # TODO: Continuously update joint angles and run forward kinematics
        while self.running:
            self.robot_state = self.task_space_reader.robot_state.position
            self.robot_velocity = self.task_space_reader.robot_state.velocity
            # print(self.robot_state)
        

            self.joint_angles = self.joint_state_reader.joint_angles
            self.joint_velocity = self.joint_state_reader.joint_velocity
            self.joint_torque = self.joint_state_reader.joint_tau
            self.tau = self.joint_torque
            
            self.imu_data = self.joint_state_reader.imu_data
            self.imu_velocity = self.joint_state_reader.imu_gyroscope
            self.imu_acc = self.joint_state_reader.imu_accelerometer
            self.set_joint_angles()
            self.update_pinocchio_fk_model()
            self.update_jacobian_derivative()
            # print("Updated all joint angles in pinocchio model")

    def set_joint_angles(self):
        """
        Sets the joint angles in the MuJoCo qpos array.
        """
        # TODO: Set the joint angles and robot state in the MuJoCo data structure
        # self.qpos[:3] 
        self.qpos[:3] = self.robot_state
        self.qpos[3:7] = self.imu_data
        # Change order from wxyz to xyzw
        self.qpos[3:7] = [
        self.imu_data[1],  # x
        self.imu_data[2],  # y 
        self.imu_data[3],  # z
        self.imu_data[0]   # w
        ]

        self.qpos[7:19] = self.joint_angles
    
        self.qvel[:3] = self.robot_velocity
        self.qvel[3:6] = self.imu_velocity
        self.qvel[6:19] = self.joint_velocity

    def update_pinocchio_fk_model(self):
        """
        Update the Pinocchio model with the latest joint angles.
        """
        # run forward kinematics
        pin.forwardKinematics(self.model, self.data, self.qpos)

        # get the contact jacobian 
        for leg in self.contact_legs:
            frame_id = self.model.getFrameId(leg)
            J_frame = pin.computeFrameJacobian(self.model, self.data, self.qpos, frame_id, pin.LOCAL_WORLD_ALIGNED)
        
            rlocglob = R.from_quat(self.qpos[3:7]).as_matrix()
            J_frame[:3,:3] = J_frame[:3,:3] @ rlocglob.T
        
            self.J_contact[3 * self.contact_legs.index(leg):3 * (self.contact_legs.index(leg) + 1), :] = J_frame[:3, :]
    
    def update_jacobian_derivative(self):
        # Get current joint positions (q) and velocities (v)
        q = self.qpos  # Shape: (nq,)
        v = self.qvel  # Shape: (nv,)
        # Compute forward kinematics derivatives (required for velocity terms)
        pin.computeJointJacobiansTimeVariation(self.model,self.data,q,v)
        for leg in self.contact_legs:
            
            frame_id = self.model.getFrameId(leg)
            dJ_contact = np.zeros((3, self.model.nv))
            
            
            # Pass q and v explicitly
            dJ_contact = pin.getFrameJacobianTimeVariation(
                self.model, 
                self.data, 
                frame_id, 
                pin.LOCAL_WORLD_ALIGNED, 
            )
            
            
            self.dJ_contact[3 * self.contact_legs.index(leg):3 * (self.contact_legs.index(leg) + 1), :] = dJ_contact[:3, :]
    
   
    def compute_dynamics_constraints(self):
        """
        Formulate whole-body dynamics constraints.

        Args:
            M: Mass matrix.
            bias_forces: Bias forces.
            Jc: Contact Jacobian matrix.

        Returns:
            A_dyn, l_dyn, u_dyn: Dynamics constraint matrix and bounds.
        """
        q = self.qpos  # Shape: (nq,)
        v = self.qvel  # Shape: (nv,)
        # Compute mass matrix
        pin.crba(self.model, self.data, q)
        self.M = self.data.M.copy()
    
        pin.computeCoriolisMatrix(self.model, self.data, q , v)
        self.Coriolis = self.data.C.copy() @ v.reshape(-1,1)
        
        pin.computeGeneralizedGravity(self.model, self.data, q)
        self.G = self.data.g.copy().reshape(-1,1)

        
        self.S = np.hstack((np.zeros((self.model.nv - 6, 6)), np.eye(self.model.nv - 6)))
        tau_cmd = (self.S.T @ self.tau).reshape(-1,1)  # Commanded torques
        # Receive ddq_cmd from InverseKinematics
  


        A1 = sparse.csc_matrix(-self.J_contact.T)  # Transposed contact Jacobian
        A2 = sparse.csc_matrix((self.J_contact.shape[1], self.J_contact.shape[0]))  # Placeholder
        A3 = sparse.csc_matrix(self.M)  # Mass matrix in sparse format
        A_matrix = sparse.hstack([A1, A2, A3])  # Constraint matrix
        dynamics_u = tau_cmd - self.Coriolis - self.G - self.M @ self.ddq_dik # Equality constraint RHS
        dynamics_l = dynamics_u  # Equality constraint bounds
        # print(f"A_matrix: {A_matrix.shape}, dynamics_l: {dynamics_l.shape}, dynamics_u: {dynamics_u.shape}")
        return A_matrix, dynamics_l, dynamics_u
    
    def convert_quat_to_euler(self, quat):
        """
        Convert quaternion to Euler angles.

        Parameters:
            quat (np.ndarray): Quaternion to convert in the order [q0, q1, q2, q3].

        Returns:
            np.ndarray: Euler angles [roll, pitch, yaw] in radians.
        """
        q0, q1, q2, q3 = quat
        sin_pitch = 2 * (q0 * q2 - q3 * q1)
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        pitch = np.arcsin(sin_pitch)
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        
        roll = np.clip(roll, -np.pi/2, np.pi/2)
        pitch = np.clip(pitch, -np.pi/2, np.pi/2)
        yaw = np.clip(yaw, -np.pi, np.pi)
        
        return np.array([roll, pitch, yaw]).round(3)
    
    def receive_ddq_ik(self):
        """
        Receive the desired joint accelerations from the connection.
        """
        if self.conn.poll(timeout=1):
            ik_data = self.conn.recv()
            self.ddq_ik = np.array(ik_data["ddqd"]).reshape(-1, 1)
            self.dq_ik =  np.array(ik_data["dqd"]).reshape(-1, 1)
            self.q_ik = np.array(ik_data["qd"]).reshape(-1, 1)
            self.ddq_dik = np.vstack((np.zeros((6, 1)), np.array(ik_data["ddq_desired"]).reshape(-1, 1)))
            self.J_contact_ik = np.array(ik_data["J_contact"])
            self.kp = np.array(ik_data["kp"])
            self.kd = np.array(ik_data["kd"])

 
        else:
            print("Waiting for data...")
        
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
        Fz = sparse.block_diag([Fz_single] * self.num_contacts)

        A1 = sparse.csc_matrix(Fz)
        A2 = sparse.csc_matrix((Fz.shape[0], self.ddxc_dim))
        A3 = sparse.csc_matrix((Fz.shape[0], self.ddq_dim))
        A_matrix = sparse.hstack([A1, A2, A3],format='csc')
        l = np.zeros(Fz.shape[0]).reshape(-1, 1)
        u = np.ones(Fz.shape[0]).reshape(-1, 1) * np.inf
        return A_matrix, l, u

    def compute_contact_acceleration_constraints(self):

        A1 = sparse.csc_matrix((self.F_dim, self.F_dim))  # Placeholder
        A2 = sparse.csc_matrix(np.eye(self.ddxc_dim))  # Identity matrix
        A3 = -self.J_contact  # Negative Jacobian
        A_matrix = sparse.hstack([A1, A2, A3],format='csc')  # Constraint matrix
        # print(f"J_contact: {self.J_contact.shape}, ddq_dik: {self.ddq_dik.shape}, dJ_contact: {self.dJ_contact.shape}, qvel: {self.qvel.shape}")
        l = self.J_contact @ self.ddq_dik + self.dJ_contact @ self.qvel.reshape(-1,1)  # Lower bound
        u = l
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
        A_dyn, l_dyn, u_dyn = self.compute_dynamics_constraints()
        A_react, l_react, u_react = self.compute_reaction_force_constraints()
        A_contact, l_contact, u_contact = self.compute_contact_acceleration_constraints()
        A_contact_force_dir, l_contact_force_dir, u_contact_force_dir = self.compute_contact_force_direction_constraints()

        self.A = sparse.vstack([A_dyn, A_react, A_contact, A_contact_force_dir])
        self.l = np.vstack([l_dyn, l_react, l_contact, l_contact_force_dir])
        self.u = np.vstack([u_dyn, u_react, u_contact, u_contact_force_dir])
        # print(f"A: {self.A.shape}, l: {self.l.shape}, u: {self.u.shape}")

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
    
    def calculate_tau_cmd(self, Fc_sol, ddxc_sol, ddq_sol):
        """
        Calculate the commanded torques.
        # """
        self.tau = np.zeros((self.model.nv - 6, 1))
        self.tau =  np.linalg.pinv(self.S.T) @ (self.M @ (self.ddq_dik + ddq_sol) + self.Coriolis + self.G - self.J_contact.T @ Fc_sol) 
        # print(f"Fc_sol: {Fc_sol.T} \n, ddxc_sol: {ddxc_sol.T} \n, ddq_sol: {ddq_sol.T} \n", end="\n")
        # print(f"tau: {self.tau.T}")
    
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

        self.cmd.send_motor_commands(self.kp, self.kd, self.change_q_order(self.q_ik), self.change_q_order(self.dq_ik),self.change_q_order(self.tau))
        self.ErrorPlotting.tau_data.append(self.tau)
    def change_q_order(self, q):
        """
        Change the order of the joint angles.
        """
        return np.array(
            [
                q[3], q[4], q[5], q[0], q[1], q[2], q[9], q[10], q[11], q[6], q[7], q[8]
            ]
    )
# Add these helper functions outside the class
def run_inverse_kinematics(conn):
    ChannelFactoryInitialize(1, "lo")
    ik = InverseKinematic(conn)
    ik.main()

def run_whole_body_controller(conn):
    ChannelFactoryInitialize(1, "lo")
    
    wbc = WholeBodyController(conn)
    wbc.cmd.move_to_initial_position()

    for _ in tqdm.tqdm(range(1000), desc="Running Inverse Dynamics"):
        wbc.receive_ddq_ik()
        Fc_sol, ddxc_sol, ddq_sol = wbc.solve()
        wbc.calculate_tau_cmd(Fc_sol.reshape(-1, 1), ddxc_sol.reshape(-1, 1), ddq_sol.reshape(-1, 1))
    wbc.ErrorPlotting.plot_contact_force(wbc.ErrorPlotting.Fc_data, "Contact Force")
    wbc.ErrorPlotting.plot_contact_acceleration(wbc.ErrorPlotting.ddxc_data, "Contact Acceleration")
    wbc.ErrorPlotting.plot_full_body_state(wbc.ErrorPlotting.ddq_diff_data, "Joint Accelerations")
    wbc.ErrorPlotting.plot_torque(wbc.ErrorPlotting.tau_data, "Joint Torques")
    plt.show()
if __name__ == "__main__":
    # Create pipe
    parent_conn, child_conn = multiprocessing.Pipe()

    # Proper process setup
    writer_process = multiprocessing.Process(
        target=run_inverse_kinematics,  # New wrapper function
        args=(child_conn,)
    )
    
    reader_process = multiprocessing.Process(
        target=run_whole_body_controller,  # New wrapper function
        args=(parent_conn,)
    )

    # Start processes
    writer_process.start()
    reader_process.start()

    # Clean shutdown
    writer_process.join()
    reader_process.join()

