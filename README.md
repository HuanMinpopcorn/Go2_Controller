# Go2 Controller

## Features
1. Forward Kinematics (FK)
2. Inverse Kinematics (IK)
3. Low-level control command
4. PhysicalSim: Open the viewer from a class
5. read_Jointstate: read topic "rt/lowstate"
6. read_Taskspace: read topic "rt/sportmodestate"

## Identify the Go2.XML

The legs and joints are organized as follows:

- **Front Left Leg (FL)**
    - Joints: FL_hip_joint, FL_thigh_joint, FL_calf_joint
    - End-effector: FL_foot

- **Front Right Leg (FR)**
    - Joints: FR_hip_joint, FR_thigh_joint, FR_calf_joint
    - End-effector: FR_foot

- **Rear Left Leg (RL)**
    - Joints: RL_hip_joint, RL_thigh_joint, RL_calf_joint
    - End-effector: RL_foot

- **Rear Right Leg (RR)**
    - Joints: RR_hip_joint, RR_thigh_joint, RR_calf_joint
    - End-effector: RR_foot

## unitree_sdk2py_bridge API

In `unitree_sdk2py_bridge.py`, the following topics are published by Mujoco:

```
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
```

These topics provide joint states (q, dq, tau) which are not equal to qpos because the order is changed, as well as IMU data and power data. 
The high state "sportmodestate" provides IMU sensor data positions (x, y, z, dx, dy, dz).

 ** when give the command to the api, it need to switch the order. 

# Xpos == Link_index
 <<------------- Link ------------->> 
link_index: 0 , name: world
link_index: 1 , name: base_link
link_index: 2 , name: FL_hip
link_index: 3 , name: FL_thigh
link_index: 4 , name: FL_calf
link_index: 5 , name: FL_foot
link_index: 6 , name: FR_hip
link_index: 7 , name: FR_thigh
link_index: 8 , name: FR_calf
link_index: 9 , name: FR_foot
link_index: 10 , name: RL_hip
link_index: 11 , name: RL_thigh
link_index: 12 , name: RL_calf
link_index: 13 , name: RL_foot
link_index: 14 , name: RR_hip
link_index: 15 , name: RR_thigh
link_index: 16 , name: RR_calf
link_index: 17 , name: RR_foot

# Command index == sensor data index 
sensor_index: 0 , name: FR_hip_pos , dim: 1
sensor_index: 1 , name: FR_thigh_pos , dim: 1
sensor_index: 2 , name: FR_calf_pos , dim: 1
sensor_index: 3 , name: FL_hip_pos , dim: 1
sensor_index: 4 , name: FL_thigh_pos , dim: 1
sensor_index: 5 , name: FL_calf_pos , dim: 1
sensor_index: 6 , name: RR_hip_pos , dim: 1
sensor_index: 7 , name: RR_thigh_pos , dim: 1
sensor_index: 8 , name: RR_calf_pos , dim: 1
sensor_index: 9 , name: RL_hip_pos , dim: 1
sensor_index: 10 , name: RL_thigh_pos , dim: 1
sensor_index: 11 , name: RL_calf_pos , dim: 1