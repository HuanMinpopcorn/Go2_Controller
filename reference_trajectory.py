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


        # Compute the total number of trajectory points per swing phase
        self.K = K

        # Internal storage for updated initial states
        self.updated_body_configuration = np.copy(self.initial_body_configuration)
        self.updated_swing_leg_positions = np.copy(self.initial_swing_leg_positions)
        self.updated_contact_leg_positions = np.copy(self.initial_contact_leg_positions)
    
    def foot_trajectory(self, i):
        """Calculate foot trajectory and its derivative for swing phase"""
        # Common parameters
        t = i / self.K  # Normalized time [0, 1]
        omega = 2 * np.pi / self.K  # Angular frequency
        
        # Select trajectory type (make this a class parameter if needed)
        trajectory_type = 1  # 1=parabola, 2=sinusoidal, 3=elliptical
        
        if trajectory_type == 1:
            # Parabolic trajectory (smooth start/stop)
            x = self.velocity["x"] * t * 2
            y = self.velocity["y"] * t * 2
            z = self.swing_height * (1 - np.cos(omega * i)) ** 2 / 4

            dx_dt = self.velocity["x"] * 2
            dy_dt = self.velocity["y"] * 2
            dz_dt = self.swing_height * omega * np.sin(omega * i) * (1 - np.cos(omega * i))


        elif trajectory_type == 2:
            # Sinusoidal trajectory (symmetric)
            foot_traj_z = self.swing_height * np.sin(np.pi * t)
            foot_traj_deriv = self.swing_height * np.pi * np.cos(np.pi * t) / self.K
            
        elif trajectory_type == 3:
            # Elliptical trajectory (parametric form)
            #(x,y) = (a*cos(t), b*sin(t))
            theta = np.pi * t  # Parameter from 0 to π
            # Ellipse parameters
            a_x = self.swing_time * abs(self.velocity["x"])/2  # Semi-major axis (forward motion)
            a_y = self.swing_time * abs(self.velocity["y"])/2  # Semi-major axis (forward motion)
            b = self.swing_height  # Semi-minor axis (vertical motion)
            
            # Position components (parametric equations)
            x = a_x * (1 - np.cos(theta))  # Forward displacement
            y = a_y * (1 - np.cos(theta))  # Lateral displacement
            z = b * np.sin(theta)           # Vertical displacement
            
            # Velocity components (derivatives w.r.t. time)
            dx_dt = a_x * np.sin(theta) * (np.pi )
            dy_dt = a_y * np.sin(theta) * (np.pi)
            dz_dt = b * np.cos(theta) * (np.pi )
                    
        else:
            raise ValueError("Invalid trajectory type")

        return np.array([x, y, z]), np.array([dx_dt, dy_dt, dz_dt])
        
    def rotation_matrix(self, theta):
        """
        Generate a rotation matrix for a given angle theta.
        Args:
            theta (float): Angle in radians.

        Returns:
            np.array: 3x3 rotation matrix.
        """
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float64)  
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

        for i in range(1, self.K + 1):

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

        # Initialize positions based on phase
        if self.swing_phase == 0:
            init_positions = np.copy(self.updated_swing_leg_positions)
        else:
            init_positions = np.copy(self.updated_contact_leg_positions)

        for i in range(1, self.K + 1):
            # Get swing trajectory components
            foot_displacement, _ = self.foot_trajectory(i)  # Fixed tuple unpacking
            
            # Calculate accumulated rotation
            theta = self.velocity["theta"] * i * self.step_size
            rot_mat = self.rotation_matrix(theta)
            
            # Calculate body displacement
            dx_body = foot_displacement[0]
            dy_body = foot_displacement[1]
            dz_body = foot_displacement[2]

            positions = np.copy(init_positions)
            
            for leg_index in range(len(swing_legs)):
                base_idx = 3 * leg_index

                # Get initial positions from body frame
                positions[base_idx] = init_positions[base_idx] + dx_body
                positions[base_idx + 1] = init_positions[base_idx + 1] + dy_body
                positions[base_idx + 2] = init_positions[base_idx + 2] + dz_body
                # positions[base_idx] =  dx_body
                # positions[base_idx + 1] = dy_body
                # positions[base_idx + 2] = dz_body
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
        for i in range(1, self.K + 1):
            desired_body_velocity[0] =  self.velocity["x"]
            desired_body_velocity[1] =  self.velocity["y"]
            desired_body_velocity[2] = 0.0
            desired_body_velocity[3] = 0.0
            desired_body_velocity[4] = 0.0
            desired_body_velocity[5] = self.velocity["theta"]
            # You can similarly update other velocity components if needed
            body_velocity_trajectory.append(desired_body_velocity.copy())
        # Update the stored “initial” body velocity for next time
        self.initial_body_velocity = body_velocity_trajectory[-1]
        return np.array(body_velocity_trajectory)

    def compute_desired_swing_leg_velocity_trajectory(self, swing_legs):
        """
        Generate velocity references for swing legs including rotational effects.
        Returns:
            np.array: Velocity trajectory [K, 3*len(swing_legs)] with shape (steps, 3*num_legs)
        """
        swing_leg_velocity_trajectory = []
        swing_leg_velocity = np.copy(self.initial_swing_leg_velocity)

        for i in range(1, self.K + 1):
            # Get foot trajectory derivatives
            _, foot_traj_deriv = self.foot_trajectory(i)
            
            # Calculate accumulated rotation angle
            theta = self.velocity["theta"] * i * self.step_size
            
            for leg_index in range(len(swing_legs)):
                base_idx = 3 * leg_index
                
                # # Calculate body displacement in body frame
                dx_body = foot_traj_deriv[0]
                dy_body = foot_traj_deriv[1]
                dz_body = foot_traj_deriv[2]
                swing_leg_velocity[base_idx + 0] = dx_body
                swing_leg_velocity[base_idx + 1] = dy_body
                swing_leg_velocity[base_idx + 2] = dz_body
        

            swing_leg_velocity_trajectory.append(swing_leg_velocity.copy())

        self.initial_swing_leg_velocity = swing_leg_velocity_trajectory[-1]
        return np.array(swing_leg_velocity_trajectory)
        
    def get_trajectory(self):
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
            # Plot desired body positions and orientations
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(body_positions[:, 0], label='X')
            plt.plot(body_positions[:, 1], label='Y')
            plt.plot(body_positions[:, 2], label='Z')
            plt.plot(body_positions[:, 3], label='Roll')
            plt.plot(body_positions[:, 4], label='Pitch')
            plt.plot(body_positions[:, 5], label='Yaw')
            plt.title('Desired Body Positions and Orientations')
            plt.xlabel('Time Step')
            plt.ylabel('Position/Orientation')
            plt.legend()
            plt.grid(True)

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
            plt.grid(True)

            # Plot desired body velocities
            plt.subplot(2, 2, 3)
            plt.plot(body_velocities[:, 0], label='Vx')
            plt.plot(body_velocities[:, 1], label='Vy')
            plt.plot(body_velocities[:, 2], label='Vz')
            plt.plot(body_velocities[:, 3], label='Roll Rate')
            plt.plot(body_velocities[:, 4], label='Pitch Rate')
            plt.plot(body_velocities[:, 5], label='Yaw Rate')
            plt.title('Desired Body Velocities')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid(True)

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
            plt.grid(True)

            # Plot swing leg X positions vs Z positions
            plt.figure(figsize=(8, 6))
            for i in range(swing_legs.shape[1] // 3):
                plt.plot(swing_legs[:, 3*i], swing_legs[:, 3*i+2], label=f'Leg {i}')
            plt.title('Swing Leg X Positions vs Z Positions')
            plt.xlabel('X Position')
            plt.ylabel('Z Position')
            plt.ylim([0,1])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()


if __name__ == "__main__":
    # Example initial conditions and parameters
    initial_body_configuration = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_swing_leg_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_contact_leg_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_body_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_swing_leg_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    velocity = {
            'x': 0.01,  # Forward velocity
            'y': 0.0,   # Lateral velocity
            'theta': 0.0  # Rotational velocity
        }

    swing_height = 0.075
    swing_time = 0.25/2
    step_size = 0.001
    K = int(swing_time / step_size)
    swing_phase = 0
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
        swing_legs,
        contact_legs,
    )
    print(K)
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
        body_positions, swing_legs, body_velocities, swing_leg_velocities = ref_traj.get_trajectory()
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