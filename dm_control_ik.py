import numpy as np
from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
from dm_control.mujoco.wrapper import mjbindings
import config
from error_plotting import ErrorPlotting  # Assuming you have this module for visualization
import matplotlib.pyplot as plt
from ForwardKinematics import ForwardKinematics # Assuming you have this module for IK computation
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
mjlib = mjbindings.mjlib

class Go2LeggedRobotIK:
    """
    Inverse Kinematics Controller for a Legged Robot (e.g., quadruped) using dm_control and MuJoCo simulation.
    This class computes IK for controlling both the body and leg trajectories.
    """

    def __init__(self, xml_path, body_frame_name, swing_leg_sites, contact_leg_sites, joint_names, step_size=config.SIMULATE_DT):
        """
        Initialize the Go2LeggedRobotIK controller.

        Parameters:
        - xml_path: Path to the MuJoCo XML model file for the legged robot.
        - body_frame_name: Name of the body frame (e.g., "base_link") to control.
        - swing_leg_sites: List of site names for the swing legs (legs in motion).
        - contact_leg_sites: List of site names for the contact legs (legs maintaining contact).
        - joint_names: List of joint names for IK computation.
        - step_size: Simulation step size.
        """
        # with open(xml_path, 'r') as f:
        #     model_xml = f.read()
        self.physics = mujoco.Physics.from_xml_string(xml_path)
        self.body_frame_name = body_frame_name
        self.swing_leg_sites = swing_leg_sites
        self.contact_leg_sites = contact_leg_sites
        self.joint_names = joint_names
        self.step_size = step_size

        self.swing_time = 0.25
        self.K = int(self.swing_time / self.step_size)  # Number of steps for a swing phase
        self.body_height = 0.25
        self.swing_height = 0.075
        self.velocity = 0.0

    def initialize(self):
        """
        Initialize the robot state, joint angles, and body configuration.
        """
        self.initial_joint_angles = np.copy(self.physics.data.qpos[:len(self.joint_names)])
        self.initial_body_state = self.get_body_state(self.body_frame_name)
        self.desired_body_configuration = np.hstack(
            (self.initial_body_state['position'], self.initial_body_state['orientation'])
        )
        self.initial_swing_leg_positions = np.hstack(
            [self.get_body_state(site)["position"] for site in self.swing_leg_sites]
        )
        self.initial_contact_leg_positions = np.hstack(
            [self.get_body_state(site)["position"] for site in self.contact_leg_sites]
        )

    def get_body_state(self, body_name):
        """
        Retrieve the position and orientation of a specified body or site.
        """
        body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        position = self.physics.data.xpos[body_id]
        orientation = self.physics.data.xquat[body_id]
        return {"position": position, "orientation": orientation}

    def compute_desired_body_trajectory(self):
        """
        Compute the desired body trajectory over the swing phase.
        """
        self.desired_body_configuration[0] += self.velocity * self.step_size  # Example motion along x-axis
        return self.desired_body_configuration

    def compute_desired_swing_leg_trajectory(self):
        """
        Generate trajectories for the swing legs over the swing phase using a cubic spline.
        """
        swing_leg_positions = np.hstack(
            [self.get_body_state(site)["position"] for site in self.swing_leg_sites]
        )
        initial_positions = np.copy(swing_leg_positions)

        trajectory = []
        for i in range(self.K):
            swing_leg_positions[0] = initial_positions[0] + self.cubic_spline(i, self.K, self.velocity * self.swing_time)
            swing_leg_positions[1] = initial_positions[1]  # Y-coordinate remains constant
            swing_leg_positions[2] = initial_positions[2] + self.swing_height * np.sin(np.pi * i / self.K)
            trajectory.append(swing_leg_positions.copy())

        return np.array(trajectory)

    def cubic_spline(self, t, tf, xf):
        """
        Generate a cubic spline trajectory.
        """
        a2 = -3 * xf / tf**2
        a3 = 2 * xf / tf**3
        return a2 * t**2 + a3 * t**3

    def compute_ik_for_body_and_legs(self):
        """
        Compute the inverse kinematics to control both the body and swing legs.
        """
        x_sw_trajectory = self.compute_desired_swing_leg_trajectory()
        for i in range(self.K):
            # Update body configuration
            x_b = self.compute_desired_body_trajectory()
            body_result = ik.qpos_from_site_pose(
                physics=self.physics,
                site_name=self.body_frame_name,
                target_pos=x_b[:3],
                target_quat=x_b[3:],
                joint_names=self.joint_names,
                tol=1e-6,
                max_steps=100,
                inplace=True
            )
            if not body_result.success:
                print("Failed to find a valid body pose.")
                continue

            # Compute IK for swing legs
            for leg_index, leg_site in enumerate(self.swing_leg_sites):
                swing_leg_pos = x_sw_trajectory[i, leg_index*3:(leg_index+1)*3]
                leg_result = ik.qpos_from_site_pose(
                    physics=self.physics,
                    site_name=leg_site,
                    target_pos= swing_leg_pos,
                    joint_names=self.joint_names,
                    tol=1e-6,
                    max_steps=100,
                    inplace=True
                )
                if not leg_result.success:
                    print(f"Failed to find valid swing leg position for {leg_site}.")
                    continue

            self.physics.step()  # Step the simulation

    def transition_legs(self):
        """
        Swap the swing and contact legs for the next gait cycle.
        """
        self.swing_leg_sites, self.contact_leg_sites = self.contact_leg_sites, self.swing_leg_sites

if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    xml_path = config.ROBOT_SCENE  # Replace with actual path to your robot's XML model
    # print(xml_path)
    # xml_path = "../unitree_mujoco/unitree_robots/go2/go2.xml"
    fk = ForwardKinematics(xml_path)
    fk.start_joint_updates()
    body_frame_name = 'base_link'  # Replace with your robot's body frame name
    swing_leg_sites = ['FL_foot', 'RR_foot']  # Replace with your actual swing leg site names
    contact_leg_sites = ['FR_foot', 'RL_foot']  # Replace with your actual contact leg site names
    # joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4']  # Replace with your robot's joint names
    joint_names = []
    for i in range(fk.model.njnt):
        name = mujoco.mj_id2name(fk.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(name)
    go2_ik = Go2LeggedRobotIK(xml_path, body_frame_name, swing_leg_sites, contact_leg_sites, joint_names)
    go2_ik.initialize()
    go2_ik.compute_ik_for_body_and_legs()
