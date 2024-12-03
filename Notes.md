# Inverse Kinematic 
Hierarchy Pseudo-Inverse Jacobian for Inverse Kinematics

$$
\dot{\mathbf{q}} = J_1^+ \dot{\mathbf{x}}_1^d + (J_2N_1)^+(\dot{\mathbf{x}}_2^d - J_2J_1^+\dot{\mathbf{x}}_1^d) \tag{1}
$$

where: 
* $J_1$: body jacobian of swing leg 
* $J_2$: body jacobian of contact leg 
* $N_1$: Null space of $J_1$ 
* $N_2$: Null space of $J_2$

Derive process: 
if we assume there is $q_0$ exit where:
$$
\dot{q} = J_1^+ \dot{\mathbf{x}}_1^d + N_1\dot{q_0} \tag{2}
$$
mutiple $J_1$ at both side 
$$
\dot{x_1} = J_1 \dot{q} = J_1J_1^+ \dot{\mathbf{x}}_1^d + J_1N_1\dot{q_0} \tag{3}
$$
where $ J_1 J_1^+ = I $ , $J_1N_1 = 0$.

if mutiple $J_2$ at both side for equation (2) 
$$
\dot{x_2}^d = J_2 \dot{q} = J_2J_1^+ \dot{\mathbf{x}}_1^d + J_2N_1\dot{q_0} \tag{4}
$$

then get : 
$$
\dot{q_0} = (J_2N_1)^+(\dot{x_2}^d - J_2J_1^+ \dot{\mathbf{x}}_1^d) \tag{5}
$$


If we have three state need to control:

$$
\dot{\mathbf{q_3}} = J_1^+ \dot{\mathbf{x}}_1^d + (J_2N_1)^+(\dot{\mathbf{x}}_2^d - J_2J_1^+\dot{\mathbf{x}}_1^d) + (J_3N_{2|1})^+ (\dot{\mathbf{x}}_3^d - J_3J_{2|1}^+ \dot{\mathbf{x}}_2^d)  \tag{6}
$$

where: 
* $x_1$: state of contact leg 
* $x_2$: state of body frame
* $x_3$  state of swing leg 
* $J_1$: body jacobian of contact leg $\in R^{3 \times 18}$
* $J_2$: body jacobian of body frame 
* $J_3$: body jacobian of swing leg 
* $N_1$: Null space of $J_1$ = $(I - J_1^+ J_1)$
* $N_2$: Null space of $J_2$ = $(I - J_2^+ J_2)$
* $J_{2|1}$: $J_2N_1$
* $N_{2|1}$:$Null(J_{2|1})$
* $+$ : pseudo inverse operator 


As contact leg is static on the ground, $\Delta x_1 = \dot{\mathbf{x}}_1^d = \mathbf{0} $, equation (5) become to:
$$
\dot{\mathbf{q}} =(J_2N_1)^+(\dot{\mathbf{x}}_2^d) + (J_3N_{2|1})^+ (\dot{\mathbf{x}}_3^d - J_3 J_{2|1}^+ \dot{\mathbf{x}}_2^d)  \tag{7}
$$

Discretized the equation (7) to get: 
$$
\Delta{\mathbf{q}} =(J_2N_1)^+(\Delta{\mathbf{x}}_2^d) + (J_3N_{2|1})^+ (\Delta{\mathbf{x}}_3^d - J_3 J_{2|1}^+ \Delta{\mathbf{x}}_2^d)  \tag{8}
$$

then 
$$
\mathbf{q}^d = \mathbf{q}^m  + \Delta\mathbf{q} * \Delta t  \tag{9}
$$
$$
\dot{\mathbf{q}}^d =(J_2N_1)^+(\dot{\mathbf{x}}_2^d) + (J_3N_{2|1})^+ (\dot{\mathbf{x}}_3^d - J_3 J_{2|1}^+ \dot{\mathbf{x}}_2^d)  \tag{10}
$$


# Inverse Dynamic 
## DynWBC
Given joint positon, velocity and accerlation commands from the KinWBC, the DynWBC computes torques commands while considering th erobot dynamic modela nd various constaints. The optimizaiotn algorithm to compute toque commands in DynWBC is as follow: 

# Optimization Problem

## Objective Function
The optimization problem is:

$$
\min_{F_r, \ddot{x}_c, \delta_{\ddot{q}}} 
F_r^\top W_r F_r + \ddot{x}_c^\top W_c \ddot{x}_c + \delta_{\ddot{q}}^\top W_{\ddot{q}} \delta_{\ddot{q}}
$$

## Subject to:
### (1) Unilateral Force Constraint:
$$
U F_r \geq 0
$$

### (2) Reaction Force Limits:
$$
S F_r \leq F_r^{\text{max}}
$$

### (3) Kinematic Relationship:
$$
\ddot{x}_c = J_c \ddot{q} + \dot{J}_c \dot{q}
$$

### (4) Dynamic Equation:
$$
A \ddot{q} + b + g = \begin{bmatrix}
0_{6 \times 1} \\
\tau_{\text{cmd}}
\end{bmatrix} + J_c^\top F_r
$$

### (5) Acceleration Relationship:
$$
\ddot{q} = \ddot{q}^{\text{cmd}} + \delta_{\ddot{q}}
$$

### (6) Commanded Acceleration:
$$
\ddot{q}^{\text{cmd}} = \ddot{q}^d + k_d (\dot{q}^d - \dot{q}) + k_p (q^d - q)
$$

### (7) Torque Limits:
$$
\tau_{\text{min}} \leq \tau_{\text{cmd}} \leq \tau_{\text{max}}
$$

# Optimization Problem Formulation Using OSQP

## **1. Cost Function**
The cost function is:

$$
\min_{F_r, \ddot{x}_c, \delta_{\ddot{q}}} \quad F_r^\top W_r F_r + \ddot{x}_c^\top W_c \ddot{x}_c + \delta_{\ddot{q}}^\top W_{\ddot{q}} \delta_{\ddot{q}}
$$

### **Matrix \( P \)**
The Hessian matrix \( P \) is:

$$
P = \begin{bmatrix}
W_r & 0 & 0 \\
0 & W_c & 0 \\
0 & 0 & W_{\ddot{q}}
\end{bmatrix}
$$

- \( W_r \): Weight matrix for \( F_r \) (size \( n_{F_r} \times n_{F_r} \)).
- \( W_c \): Weight matrix for \( \ddot{x}_c \) (size \( n_{\ddot{x}_c} \times n_{\ddot{x}_c} \)).
- \( W_{\ddot{q}} \): Weight matrix for \( \delta_{\ddot{q}} \) (size \( n_{\delta_{\ddot{q}}} \times n_{\delta_{\ddot{q}}} \)).

### **Vector \( q \)**
The linear term in the cost function:

$$
q = 0
$$

---

## **2. Constraints**
The problem includes the following constraints:

### **(1) Unilateral Force Constraint: \( U F_r \geq 0 \)**
$$
A_1 = \begin{bmatrix} U & 0 & 0 \end{bmatrix}, \quad l_1 = 0, \quad u_1 = \infty
$$

### **(2) Force Limits: \( S F_r \leq F_r^{\max} \)**
$$
A_2 = \begin{bmatrix} S & 0 & 0 \end{bmatrix}, \quad l_2 = -\infty, \quad u_2 = F_r^{\max}
$$

### **(3) Kinematic Relationship: $ \ddot{x}_c = J_c \ddot{q} + \dot{J}_c \dot{q} $**
$$
A_3 = \begin{bmatrix} 0 & -I & J_c \end{bmatrix}, \quad l_3 = -\dot{J}_c \dot{q}, \quad u_3 = -\dot{J}_c \dot{q}
$$

### **(4) Dynamics: $ M \ddot{q} + B + g = \tau_{cmd} + J_c^\top F_r $**
$$
A_4 = \begin{bmatrix} -J_c^\top & 0 & A \end{bmatrix}, \quad l_4 = -b - g, \quad u_4 = -b - g
$$

### **(5) Commanded Acceleration: \( \ddot{q} = \ddot{q}^{cmd} + \delta_{\ddot{q}} \)**
$$
A_5 = \begin{bmatrix} 0 & 0 & I \end{bmatrix}, \quad l_5 = \ddot{q}^{cmd}, \quad u_5 = \ddot{q}^{cmd}
$$

### **(6) Torque Limits: \( \tau_{\min} \leq \tau_{cmd} \leq \tau_{\max} \)**
$$
A_6 = \begin{bmatrix} 0 & 0 & I \end{bmatrix}, \quad l_6 = \tau_{\min}, \quad u_6 = \tau_{\max}
$$

---

## **3. Combined Constraints**
All constraints are stacked to form the final constraint matrix:

$$
A = \begin{bmatrix}
U & 0 & 0 \\
S & 0 & 0 \\
0 & -I & J_c \\
-J_c^\top & 0 & A \\
0 & 0 & I \\
0 & 0 & I
\end{bmatrix}
$$

The lower and upper bounds are:

$$
l = \begin{bmatrix}
0 \\
-\infty \\
-\dot{J}_c \dot{q} \\
-b - g \\
\ddot{q}^{cmd} \\
\tau_{\min}
\end{bmatrix}, \quad
u = \begin{bmatrix}
\infty \\
F_r^{\max} \\
-\dot{J}_c \dot{q} \\
-b - g \\
\ddot{q}^{cmd} \\
\tau_{\max}
\end{bmatrix}
$$

---

## **4. OSQP Formulation**
The full problem can now be expressed as:

$$
\min_x \frac{1}{2} x^\top P x + q^\top x
$$

Subject to:

$$
l \leq A x \leq u
$$
