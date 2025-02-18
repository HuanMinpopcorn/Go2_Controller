import numpy as np
import matplotlib.pyplot as plt
class ReferenceTrajectory:
    """
    Manages reference (desired) trajectories for the quadruped’s body and swing legs.
    """

    def __init__(
        self,
        initial_body_configuration,
        initial_swing_leg_positions,
        initial_contact_leg_positions,
        initial_body_velocity,
        initial_swing_leg_velocity,
        velocity,
        swing_height,
        swing_time,
        step_size,
        K,
        swing_phase,
        walk_phase,
        swing_legs,
        contact_legs,
    ):
        """
        Args:
            initial_body_configuration (np.array): Initial [pos, orientation] or [x, y, z, ...] of the body.
            initial_swing_leg_positions (np.array): Initial foot positions of the swing legs.
            initial_contact_leg_positions (np.array): Initial foot positions of the contact legs.
            velocity (dict): Desired velocities {'x': float, 'y': float, 'theta': float}.
            swing_height (float): Maximum foot swing height.
            swing_time (float): Duration of one swing phase (seconds).
            step_size (float): Simulation time step.
            phase (int): Gait phase (0 or 1) to select which legs are swinging.
            body_account (int): Count of how many times the body has stepped.
        """
        self.initial_body_configuration = np.copy(initial_body_configuration)
        self.initial_swing_leg_positions = np.copy(initial_swing_leg_positions)
        self.initial_contact_leg_positions = np.copy(initial_contact_leg_positions)

        self.initial_body_velocity = np.copy(initial_body_velocity)
        self.initial_swing_leg_velocity = np.copy(initial_swing_leg_velocity)

        self.velocity = velocity
        self.swing_height = swing_height
        self.swing_time = swing_time
        self.step_size = step_size
        self.swing_phase = swing_phase
        self.swing_legs = swing_legs 
        self.contact_legs = contact_legs
        self.walk_phase = walk_phase

        # Compute the total number of trajectory points per swing phase
        self.K = K

        # Internal storage for updated initial states
        self.updated_body_configuration = np.copy(self.initial_body_configuration)
        self.updated_swing_leg_positions = np.copy(self.initial_swing_leg_positions)
        self.updated_contact_leg_positions = np.copy(self.initial_contact_leg_positions)


    def compute_desired_body_state(self):
        """
        Generate a trajectory for the body (position/orientation) over the swing phase.
        Returns:
            np.array: Trajectory array of shape [K, ...] (each row is a desired body configuration).
        """
        body_moving_trajectory = []
        # Copy the latest "initial" body config that might have been updated from the previous cycle
        initial_body = np.copy(self.updated_body_configuration)
        desired_body = np.copy(initial_body)

        for i in range(self.K):

            # Create a new copy of the initial body configuration for each step
            displacement_x = (self.velocity["x"] * i / self.K)
            displacement_y = (self.velocity["y"] * i / self.K)
            displacement_theta = (self.velocity["theta"] * i / self.K)
            # Example: only moving in X with a constant velocity over K steps
            desired_body[0] = initial_body[0] + displacement_x
            desired_body[1] = initial_body[1] + displacement_y
            desired_body[2] = initial_body[2]
            desired_body[3] = initial_body[3]
            desired_body[4] = initial_body[4]
            desired_body[5] = initial_body[5] + displacement_theta
            # Append a copy to avoid modifying all previous elements
            body_moving_trajectory.append(desired_body.copy())

        # Update the stored “initial” body configuration for next time
        self.updated_body_configuration = body_moving_trajectory[-1].copy() 

        return np.array(body_moving_trajectory)


    def compute_desired_swing_leg_trajectory(self, swing_legs):
        """
        Generate foot trajectories for the swing legs over the swing phase.
        Args:
            swing_legs (list): Names/IDs of swing-leg frames (e.g., ["FL_foot", "RR_foot"]).
            contact_legs (list): Names/IDs of contact legs.

        Returns:
            np.array: Trajectory array of shape [K, 3*len(swing_legs)] for swing-leg positions.
        """
        swing_leg_trajectory = []

        # Decide which set of initial leg positions to use depending on phase
        if self.swing_phase == 0:
            init_positions = np.copy(self.updated_swing_leg_positions)
        else:
            init_positions = np.copy(self.updated_contact_leg_positions)

        # For each step in the swing, compute foot trajectory
        for i in range(self.K):
            # Customize this foot trajectory formula as desired
            # Example: simple parabola via (1 - cos(...))**2
            # Scaling by some factor (e.g., /4) is optional if you want a smaller arc
            foot_traj_z = self.swing_height * 1/4 * (1 - np.cos(2 * np.pi * i / self.K)) ** 2 

            positions = np.copy(init_positions)
            for leg_index in range(len(swing_legs)):
                base_idx = 3 * leg_index
                # X update
                positions[base_idx + 0] = (init_positions[base_idx + 0] + (self.velocity["x"] * i / self.K) * 2)
                # Y stays the same for this example
                positions[base_idx + 1] = init_positions[base_idx + 1] + (self.velocity["y"] * i / self.K) * 2
                # Z swing
                positions[base_idx + 2] = (init_positions[base_idx + 2] + foot_traj_z)

            swing_leg_trajectory.append(positions.copy())

        # Update the “initial” positions for next cycle
        if self.swing_phase == 0:
            self.updated_swing_leg_positions = swing_leg_trajectory[-1]
        else:
            self.updated_contact_leg_positions = swing_leg_trajectory[-1]

        return np.array(swing_leg_trajectory)

    def compute_desired_body_state_velocity_trajectory(self):
        """
        Generate velocity references for the body over the swing phase.
        Args:
            initial_body_velocity (np.array): The current body velocity [vx, vy, vz, ...]

        Returns:
            np.array: Velocity trajectory of shape [K, ...].
        """
        body_velocity_trajectory = []
        desired_body_velocity = np.copy(self.initial_body_velocity)

        # Example: we only increment vx by a constant value each step
        for _ in range(self.K):
            desired_body_velocity[0] =  self.velocity["x"]
            desired_body_velocity[1] =  self.velocity["y"]
            # You can similarly update other velocity components if needed
            body_velocity_trajectory.append(desired_body_velocity.copy())
        # Update the stored “initial” body velocity for next time
        self.initial_body_velocity = body_velocity_trajectory[-1]
        return np.array(body_velocity_trajectory)

    def compute_desired_swing_leg_velocity_trajectory(self, swing_legs):
        """
        Generate velocity references for the swing legs over the swing phase.
        Args:
            swing_legs (list): List of swing-leg frames.

        Returns:
            np.array: Velocity trajectory of shape [K, 3*len(swing_legs)].
        """
        swing_leg_velocity_trajectory = []

        swing_leg_velocity = np.copy(self.initial_swing_leg_velocity)
        for i in range(self.K):
            # Example derivative for foot’s Z-motion
            foot_traj_deriv =  self.swing_height * (np.pi / self.K) * np.sin(2 * np.pi * i / self.K) * (1 - np.cos(2 * np.pi * i / self.K))

            # Add forward velocity to X
            for leg_index in range(len(swing_legs)):
                base_idx = 3 * leg_index
                swing_leg_velocity[base_idx + 0] = self.velocity["x"] * 2
                # Y remains constant
                swing_leg_velocity[base_idx + 1] = self.velocity["y"] * 2
                # Z follows partial derivative of (1 - cos(...))^2 / 4 or your chosen swing formula
                swing_leg_velocity[base_idx + 2] = foot_traj_deriv

            swing_leg_velocity_trajectory.append(swing_leg_velocity.copy())

        # Update the stored “initial” swing-leg velocity for next time
        self.initial_swing_leg_velocity = swing_leg_velocity_trajectory[-1]
        return np.array(swing_leg_velocity_trajectory)
    
    def get_trajectory(self, walking_phase):
        """
        Demonstration of how to use the reference trajectories.
        """
        # Example usage after 'initialize()':
        desired_body_positions = self.compute_desired_body_state()
        desired_swing_legs = self.compute_desired_swing_leg_trajectory(self.swing_legs)
        # Get velocity references if needed:
        desired_body_velocities = self.compute_desired_body_state_velocity_trajectory()
        desired_swing_leg_velocities = self.compute_desired_swing_leg_velocity_trajectory(self.swing_legs)

        return desired_body_positions, desired_swing_legs, desired_body_velocities, desired_swing_leg_velocities
    def transition_legs(self):
        """
        Swap the swing and contact legs for the next cycle.
        """
        self.swing_legs, self.contact_legs = self.contact_legs, self.swing_legs
        if self.swing_phase == 0:
            self.swing_phase = 1
        else:
            self.swing_phase = 0

    def plot_trajectories(self,body_positions, swing_legs, body_velocities, swing_leg_velocities):
            """
            Plot all the computed trajectories for visualization purposes.
            """
          

            
            
            # Plot desired body positions
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(body_positions[:, 0], label='X')
            plt.plot(body_positions[:, 1], label='Y')
            plt.plot(body_positions[:, 2], label='Z')
            plt.title('Desired Body Positions')
            plt.xlabel('Time Step')
            plt.ylabel('Position')
            plt.legend()

            # Plot desired swing leg positions
            plt.subplot(2, 2, 2)
            for i in range(swing_legs.shape[1] // 3):
                plt.plot(swing_legs[:, 3*i], label=f'Leg {i} X')
                plt.plot(swing_legs[:, 3*i+1], label=f'Leg {i} Y')
                plt.plot(swing_legs[:, 3*i+2], label=f'Leg {i} Z')
            plt.title('Desired Swing Leg Positions')
            plt.xlabel('Time Step')
            plt.ylabel('Position')
            plt.legend()

            # Plot desired body velocities
            plt.subplot(2, 2, 3)
            plt.plot(body_velocities[:, 0], label='Vx')
            plt.plot(body_velocities[:, 1], label='Vy')
            plt.plot(body_velocities[:, 2], label='Vz')
            plt.title('Desired Body Velocities')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity')
            plt.legend()

            # Plot desired swing leg velocities
            plt.subplot(2, 2, 4)
            for i in range(swing_leg_velocities.shape[1] // 3):
                plt.plot(swing_leg_velocities[:, 3*i], label=f'Leg {i} Vx')
                plt.plot(swing_leg_velocities[:, 3*i+1], label=f'Leg {i} Vy')
                plt.plot(swing_leg_velocities[:, 3*i+2], label=f'Leg {i} Vz')
            plt.title('Desired Swing Leg Velocities')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity')
            plt.legend()

            # Plot swing leg Z positions vs Z velocities
            plt.figure(figsize=(8, 6))
            for i in range(swing_legs.shape[1] // 3):
                plt.plot(swing_legs[:, 3*i+2], swing_leg_velocities[:, 3*i+2], label=f'Leg {i}')
            plt.title('Swing Leg Z Positions vs Z Velocities')
            plt.xlabel('Z Position')
            plt.ylabel('Z Velocity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            


if __name__ == "__main__":
    # Example initial conditions and parameters
    initial_body_configuration = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_swing_leg_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_contact_leg_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_body_velocity = np.array([0.0, 0.0, 0.0])
    initial_swing_leg_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    velocity = .05
    swing_height = 0.075
    swing_time = 0.1
    step_size = 0.001
    K = int(swing_time / step_size)
    swing_phase = 0
    walk_phase = 0
    swing_legs = ["FL_foot", "RR_foot"]
    contact_legs = ["FR_foot", "RL_foot"]

    # Create an instance of ReferenceTrajectory
    ref_traj = ReferenceTrajectory(
        initial_body_configuration,
        initial_swing_leg_positions,
        initial_contact_leg_positions,
        initial_body_velocity,
        initial_swing_leg_velocity,
        velocity,
        swing_height,
        swing_time,
        step_size,
        K,
        swing_phase,
        walk_phase,
        swing_legs,
        contact_legs,
    )
    # Number of cycles to simulate
    num_cycles = 5
    # Initialize storage for combined trajectories
    all_body_positions = []
    all_swing_legs = []
    all_body_velocities = []
    all_swing_leg_velocities = []
    all_body_positions.append(initial_body_configuration)
    all_swing_legs.append(initial_swing_leg_positions)
    all_body_velocities.append(initial_body_velocity)
    all_swing_leg_velocities.append(initial_swing_leg_velocity)

    for _ in range(num_cycles):
        # Compute and store the trajectories for the current swing phase
        body_positions, swing_legs, body_velocities, swing_leg_velocities = ref_traj.get_trajectory(walk_phase)
        all_body_positions.append(body_positions[1:])
        all_swing_legs.append(swing_legs[1:])
        all_body_velocities.append(body_velocities[1:])
        all_swing_leg_velocities.append(swing_leg_velocities[1:])

        # Transition legs for the next cycle
        ref_traj.transition_legs()

    # Combine all cycles for plotting
    all_body_positions = np.vstack(all_body_positions)
    all_swing_legs = np.vstack(all_swing_legs)
    all_body_velocities = np.vstack(all_body_velocities)
    all_swing_leg_velocities = np.vstack(all_swing_leg_velocities)

    # Plot the combined trajectories
    ref_traj.plot_trajectories(all_body_positions, all_swing_legs, all_body_velocities, all_swing_leg_velocities)
    plt.show()