import mujoco
import mujoco.viewer
import numpy as np
import unitree_mujoco 


class Go2ForwardKinematics:
    def __init__(self, xml_path):
        """
        Initializes the MuJoCo model and data for the Unitree Go2 robot.

        Parameters:
            xml_path (str): Path to the MuJoCo XML file for the Go2 robot.
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def set_joint_angles(self, joint_angles):
        """
        Sets the joint angles in the qpos array.

        Parameters:
            joint_angles (np.ndarray): Array of joint positions to be set.
        """
        self.data.qpos[:len(joint_angles)] = joint_angles

    def run_fk(self):
        """
        Runs MuJoCo's built-in forward kinematics function.
        """
        mujoco.mj_fwdPosition(self.model, self.data)  # Compute FK

    def get_body_state(self, body_name):
        """
        Returns the position and orientation of the specified body.

        Parameters:
            body_name (str): Name of the body (e.g., 'foot_front_left').

        Returns:
            dict: Contains the 3D position and orientation matrix.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # Retrieve position and orientation
        position = self.data.xpos[body_id]  # 3D position
        orientation = self.data.xmat[body_id].reshape(3, 3)  # 3x3 rotation matrix

        return {"position": position, "orientation": orientation}

# Example Usage
if __name__ == "__main__":
    # Initialize the FK class with the XML file for Unitree Go2
    ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
    ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
    go2_fk = Go2ForwardKinematics(ROBOT_SCENE)

    # Set sample joint angles (replace with actual robot configuration)
    joint_angles = np.array([0.0, 0.5, -0.3, 0.2, -0.1, 0.4, 
                             0.3, -0.2, 0.6, -0.1, 0.1, 0.0])

    # Set the joint angles in MuJoCo's qpos array
    go2_fk.set_joint_angles(joint_angles)

    # Run the built-in FK function
    go2_fk.run_fk()

    # Retrieve the state of the front left foot
    foot_state = go2_fk.get_body_state("foot_front_left")
    print("Foot Front Left Position:", foot_state["position"])
    print("Foot Front Left Orientation:\n", foot_state["orientation"])
