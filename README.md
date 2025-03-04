# Go2_Controller

## Overview
The **Go2_Controller** repository provides a controller for the Unitree Go2 quadruped robot. This project includes functionalities such as forward and inverse kinematics, low-level control commands, physical simulation, and state monitoring to facilitate robot locomotion and task execution.


## Features
- **Forward Kinematics (FK)**: Computes the robot's end-effector position and orientation based on joint configurations.
- **Inverse Kinematics (IK)**: Determines the joint parameters required for a desired end-effector pose. [WBC_Notes](Notes/WBC_calc.md)
- **Low-Level Control Commands**: Provides direct control over the robot’s actuators.
- **Physical Simulation**: Implements a simulation viewer for debugging and testing robot behaviors.
- **Joint State Monitoring**: Reads joint states from the `rt/lowstate` topic.
- **Task Space Monitoring**: Tracks task space status through the `rt/sportmodestate` topic.
- **Gait Generator** : Compute the gait wave and foot placement. [Gait_Notes](Notes/gait_calc.md)

## Directory Structure
```
Go2_Controller/
│── Control/             # Low-level control implementation
│── Gait/                # Gait generation and control algorithms
│── Interface/           # Interfaces for communication with the robot
│── Model/               # Robot kinematics and dynamics models
│── Notes/               # Documentation and development notes
│── Plots/               # Visualization and analysis scripts
│── Simulation/          # Simulated environment for testing
│── README.md            # Project documentation
│── error_plotting.py    # Scripts for error visualization
│── main.py              # Main entry script for the controller
│── reference_trajectory.py # Predefined movement trajectories
```

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Unitree_python_SDK
- Unitree_Mujoco 
- Unitree_ROS2 

### Setup
Clone the repository:
```bash
git clone https://github.com/HuanMinpopcorn/Go2_Controller.git
cd Go2_Controller
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main control script:
```bash
python main.py
```
To visualize errors:
```bash
python error_plotting.py
```
To test reference trajectories:
```bash
python reference_trajectory.py
```
## Demo Video 

[![Go2 Controller Demo](https://img.youtube.com/vi/1gU5A_v00dM/0.jpg)](https://youtu.be/1gU5A_v00dM)





## Contributing
Contributions are welcome! If you find a bug or have a feature request, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For any inquiries, please reach out via GitHub issues or [Email](minhuanjane@gmail.com).

---

