import mujoco
import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
import time



from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ 
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

from read_JointState import read_JointState
from read_TaskSpace import read_TaskSpace

class Kinematics:
    def __init__(self, xml_path):
        """
        Initializes the MuJoCo model and data.
        
        Parameters:
            xml_path (str): Path to the MuJoCo XML file for the robot.
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def set_joint_angles(self, joint_angles):
        """
        Sets the joint angles in the MuJoCo qpos array.
        
        Parameters:
            joint_angles (np.ndarray): Array of joint positions to be set.
        """
        self.data.qpos[:len(joint_angles)] = joint_angles

    def run_fk(self):
        """Runs forward kinematics to update positions and orientations."""
        mujoco.mj_forward(self.model, self.data)

    def get_body_state(self, body_name):
        """
        Retrieves the position and orientation of the specified body.
        
        Parameters:
            body_name (str): Name of the body to retrieve the state for.
        
        Returns:
            dict: Contains 3D position and 3x3 orientation matrix of the body.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # Retrieve position and orientation
        position = np.copy(self.data.xpos[body_id])  # 3D position
        orientation = np.copy(self.data.xmat[body_id].reshape(3, 3))  # 3x3 rotation matrix

        return {"position": position, "orientation": orientation}

    def get_jacobian(self, body_name):
        """
        Computes the Jacobian matrix for a given body.
        
        Parameters:
            body_name (str): Name of the body to compute the Jacobian for.
        
        Returns:
            dict: Contains the translational and rotational Jacobians.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # Prepare storage for Jacobians (size: 3x(number of DOF))
        J_pos = np.zeros((3, self.model.nv))  # Translational Jacobian
        J_rot = np.zeros((3, self.model.nv))  # Rotational Jacobian

        # Compute the Jacobian matrices
        mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, body_id)

        return {"J_pos": J_pos, "J_rot": J_rot}


    def get_kinematics(self, body_name):
        """
        Retrieves the position, orientation, and Jacobians for a given body.

        Parameters:
            body_name (str): Name of the body (e.g., 'FL_foot', 'FR_foot').

        Returns:
            dict: Contains position, orientation, translational Jacobian, and rotational Jacobian of the body.
        """
        # Retrieve body state
        state = self.get_body_state(body_name)

        # Retrieve Jacobian
        jacobian = self.get_jacobian(body_name)

        # Combine all kinematic information into a struct
        return {
            "position": state["position"],
            "orientation": state["orientation"],
            "J_pos": jacobian["J_pos"],
            "J_rot": jacobian["J_rot"]
        }


    def print_kinematics(self, body_name):
        """
        Prints the position, orientation, and Jacobians for a given body.
        
        Parameters:
            body_name (str): Name of the body (e.g., 'FL_foot', 'FR_foot').
        """
        # Retrieve body state
        state = self.get_body_state(body_name)
        print(f"\n{body_name} Position: {state['position']}")
        print(f"{body_name} Orientation:\n{state['orientation']}")

        # Retrieve and print Jacobian
        jacobian = self.get_jacobian(body_name)
        print(f"{body_name} Jacobian (Translation):\n{jacobian['J_pos']}")
        print(f"{body_name} Jacobian (Rotation):\n{jacobian['J_rot']}")


# Example Usage
if __name__ == "__main__":
    # Initialize the Kinematics class with the XML file for the Go2 robot
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    kinematics = Kinematics(ROBOT_SCENE)

    # Initialize the channel subscriber
    ChannelFactoryInitialize(1, "lo")
    # Initialize the read_TaskSpace and read_JointState classes
    joint_state = read_JointState()

    while True:
        # low_state_handler = joint_state.low_state_handler()
        joint_angles = joint_state.joint_angles
        print(joint_angles)
        # Set joint angles and run FK
        kinematics.set_joint_angles(joint_angles)
        kinematics.run_fk()
        List of feet to analyze
        feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # Compute and print kinematics for each foot
        for foot in feet:
            kinematics.print_kinematics(foot)
        
        time.sleep(1.0)