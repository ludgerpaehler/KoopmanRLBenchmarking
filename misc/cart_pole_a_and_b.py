import sympy as sy

from sympy.abc import v, u, x, m, M, l, theta, omega, g
constant = 0
theta_double_dot = \
    (g * sy.sin(theta) -
        sy.cos(theta) * (u + m * l * omega * sy.sin(theta) / (m + M))) / \
            (l * (constant - (m * sy.cos(theta)**2) / (m + M)))

dyn = sy.Matrix([
    v,
    (u + m * l * (omega**2 * sy.sin(theta) - theta_double_dot * sy.cos(theta))) / (m+M),
    omega,
    theta_double_dot 
])

# Define the variables with respect to which you want to differentiate
state_variables_to_diff = [x, v, theta, omega]

# Compute the Jacobian matrix manually
state_jacobian = sy.Matrix([[sy.diff(dyn_i, var) for var in state_variables_to_diff] for dyn_i in dyn])

# Evaluate state jacobian at point
point = dict(
    v=0,
    theta=0,
    omega=0  
)
state_jacobian_at_point = state_jacobian.subs(point)
A = state_jacobian_at_point

# Define the variables with respect to which you want to differentiate
action_variables_to_diff = [u]

# Compute the Jacobian matrix manually
action_jacobian = sy.Matrix([[sy.diff(dyn_i, var) for var in action_variables_to_diff] for dyn_i in dyn])

# Evaluate action jacobian at point
action_jacobian_at_point = action_jacobian.subs(point)
B = action_jacobian_at_point

# Prints
print(f"dyn shape: {dyn.shape}")
print(f"state jacobian shape: {state_jacobian.shape}")
print(f"state jacobian: {state_jacobian}")
print(f"state jacobian at point: {state_jacobian_at_point}")
print(f"action jacobian shape: {action_jacobian.shape}")
print(f"acton jacobian: {action_jacobian}")
print(f"action jacobian at point: {action_jacobian_at_point}")
print(f"A: {A}")
print(f"B: {B}")
"""
dyn shape: (4, 1)
state jacobian shape: (4, 4)
state jacobian: Matrix([
    [0, 1, 0, 0],
    [0, 0, l*m*(omega**2*cos(theta) + 1.125*m*(g*sin(theta) - (l*m*omega*sin(theta)/(M + m) + u)*cos(theta))*sin(theta)*cos(theta)**2/(l*(M + m)*(-0.75*m*cos(theta)**2/(M + m) + 1)**2) + (g*sin(theta) - (l*m*omega*sin(theta)/(M + m) + u)*cos(theta))*sin(theta)/(l*(-m*cos(theta)**2/(M + m) + 1.33333333333333)) - (g*cos(theta) - l*m*omega*cos(theta)**2/(M + m) + (l*m*omega*sin(theta)/(M + m) + u)*sin(theta))*cos(theta)/(l*(-m*cos(theta)**2/(M + m) + 1.33333333333333)))/(M + m), l*m*(m*sin(theta)*cos(theta)**2/((M + m)*(-m*cos(theta)**2/(M + m) + 1.33333333333333)) + 2*omega*sin(theta))/(M + m)],
    [0, 0, 0, 1],
    [0, 0, -1.125*m*(g*sin(theta) - (l*m*omega*sin(theta)/(M + m) + u)*cos(theta))*sin(theta)*cos(theta)/(l*(M + m)*(-0.75*m*cos(theta)**2/(M + m) + 1)**2) + (g*cos(theta) - l*m*omega*cos(theta)**2/(M + m) + (l*m*omega*sin(theta)/(M + m) + u)*sin(theta))/(l*(-m*cos(theta)**2/(M + m) + 1.33333333333333)), -m*sin(theta)*cos(theta)/((M + m)*(-m*cos(theta)**2/(M + m) + 1.33333333333333))]
])
state jacobian at point: Matrix([
    [0, 1, 0, 0],
    [0, 0, -g*m/((M + m)*(-m/(M + m) + 1.33333333333333)), 0],
    [0, 0, 0, 1],
    [0, 0, g/(l*(-m/(M + m) + 1.33333333333333)), 0]
])

action jacobian shape: (4, 1)
acton jacobian: Matrix([
    [0],
    [(m*cos(theta)**2/(-m*cos(theta)**2/(M + m) + 1.33333333333333) + 1)/(M + m)],
    [0],
    [-cos(theta)/(l*(-m*cos(theta)**2/(M + m) + 1.33333333333333))]
])
action jacobian at point: Matrix([
    [0],
    [(m/(-m/(M + m) + 1.33333333333333) + 1)/(M + m)],
    [0],
    [-1/(l*(-m/(M + m) + 1.33333333333333))]
])

A: Matrix([
    [0, 1, 0, 0],
    [0, 0, -g*m/((M + m)*(-m/(M + m) + 1.33333333333333)), 0],
    [0, 0, 0, 1],
    [0, 0, g/(l*(-m/(M + m) + 1.33333333333333)), 0]
])
B: Matrix([
    [0],
    [(m/(-m/(M + m) + 1.33333333333333) + 1)/(M + m)],
    [0],
    [-1/(l*(-m/(M + m) + 1.33333333333333))]
])



A: Matrix([
    [0, 1, 0, 0],
    [0, 0, g, 0],
    [0, 0, 0, 1],
    [0, 0, -g*(M + m)/(l*m), 0]
])
B: Matrix([
    [0],
    [(-M - m + 1)/(M + m)],
    [0],
    [(M + m)/(l*m)]
])
"""