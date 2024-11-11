# Go2 Controller 
1. Forward Kinematics (FK)
2. Inverse Kinematics (IK)
3. Jacobian
4. Low-level control command
5. PysicalSim: Open the viewer from a class

# Identify the Go2.XML

From the XML, the legs and joints are organized as follows:

    Front Left Leg (FL)
        Joints: FL_hip_joint, FL_thigh_joint, FL_calf_joint
        End-effector: FL_foot

    Front Right Leg (FR)
        Joints: FR_hip_joint, FR_thigh_joint, FR_calf_joint
        End-effector: FR_foot

    Rear Left Leg (RL)
        Joints: RL_hip_joint, RL_thigh_joint, RL_calf_joint
        End-effector: RL_foot

    Rear Right Leg (RR)
        Joints: RR_hip_joint, RR_thigh_joint, RR_calf_joint
        End-effector: RR_foot

# unitree_sdk2py_bridge API

In `unitree_sdk2py_bridge.py`, the following topics are published by Mujoco:

```
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
```

These topics provide joint states (q, dq, tau) which are not equal to qpos because the order is changed, as well as IMU data and power data. The high state "sportmodestate" provides imu positions (x, y, z, dx, dy, dz) which are not equal to xpos.

These two topics' value is not good for kinematic calculation!!!!

# how to read the correct xpos and qpos
We need to read the xpos and qpos from simulation which is from viewer or sim or in gui -> watch -> field, index will give the correct value. 

