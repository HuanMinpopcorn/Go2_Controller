
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