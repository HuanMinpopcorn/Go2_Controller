
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
