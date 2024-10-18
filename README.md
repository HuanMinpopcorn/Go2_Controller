# Go2 Controller 
1. FK Kinematics
2. IK Kinematics
3. Jacobian
4. Low-lever control command
5. 

# Indetify the Go2.XML 

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
In ```unitree_sdk2py_bridge.py```, it provide that mujoco published the topic 
```
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
```
which provide the joint state (q,dq,tau) and robot configuration (x,y,z,dx,dy,dz)