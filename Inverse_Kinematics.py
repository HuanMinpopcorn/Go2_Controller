import mujoco
import numpy as np
from Kinematics import Kinematics
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
class InverseKinematic(Kinematics):
    def __init__(self, xml_path, step_size=0.01, tol=1e-3):
        """
        Initializes the InverseKinematic class by loading the MuJoCo model and setting parameters.

        Parameters:
            xml_path (str): Path to the MuJoCo XML file.
            step_size (float): Step size for gradient descent updates.
            tol (float): Tolerance for the IK solution's error.
        """
        super().__init__(xml_path)  # Initialize Kinematics class
        self.step_size = step_size
        self.tol = tol

    def check_joint_limits(self, joint_angles):
        """
        Ensures joint angles are within their defined limits.

        Parameters:
            joint_angles (np.ndarray): Array of joint angles to check and constrain.

        Returns:
            np.ndarray: Joint angles constrained within the defined joint limits.
        """
        for i in range(len(joint_angles)):
            lower_limit, upper_limit = self.model.jnt_range[i]
            joint_angles[i] = np.clip(joint_angles[i], lower_limit, upper_limit)
        return joint_angles

    def calculate(self, goal, init_q, body_name):
        """
        Calculate the inverse kinematics to achieve a desired end-effector position.

        Parameters:
            goal (np.ndarray): Desired end-effector position [x, y, z].
            init_q (np.ndarray): Initial joint angles.
            body_name (str): Name of the body to perform IK on.

        Returns:
            np.ndarray: Joint angles that achieve the desired goal position.
        """
        q = init_q.copy()

        for _ in range(100):  # Max iterations
            self.set_joint_angles(q)
            self.run_fk()

            # Get current position of the body
            current_pos = self.get_body_state(body_name)["position"]
            error = goal - current_pos

            # Check if the solution is within tolerance
            if np.linalg.norm(error) < self.tol:
                print(f"IK Converged! Error: {np.linalg.norm(error)}")
                break

            # Compute the Jacobian
            jacobian = self.get_jacobian(body_name)["J_pos"]

            # Update joint angles using gradient descent: q_new = q + J.T * error
            delta_q = self.step_size * np.dot(jacobian.T, error)
            q += delta_q

            # Ensure joint angles stay within limits
            q = self.check_joint_limits(q)

        return q


# Example Usage
if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")  # Initialize communication channel
    # Initialize the InverseKinematic class with the Go2 robot XML
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    ik = InverseKinematic(ROBOT_SCENE)

    # Define the goal position and initial joint angles
    goal_position = np.array([0.3, 0.2, 0.1])  # Example target position
    initial_joint_angles = np.zeros(ik.model.nq)  # Assume all joints start at 0

    # Perform IK for the 'FL_foot' body
    result_angles = ik.calculate(goal_position, initial_joint_angles, "FL_foot")

    print(f"Calculated Joint Angles:\n{result_angles}")
