‘’‘
#Question 1:

0-stable:

$x_{k+2} + \alpha x_{k+1} - (1 + \alpha) x_k= \frac{1}{2}h(-\alpha f_{k+2} + (4 + 3\alpha)f_{k+1})$:

We can ignore the right-hand side of the equation for now, as it doesn't affect the stability of the method. The left-hand side of the equation can be written as a recurrence relation:

$x_{k+2} = -\alpha x_{k+1} + (1 + \alpha) x_k$.

The characteristic polynomial of this equation is obtained by assuming a solution of the form $x_k = z^k$ and substituting this into the equation. This gives:

$z^2 = -\alpha z + (1 + \alpha)$

$z^2 + \alpha z - (1 + \alpha) = 0$

The roots of this polynomial are the solutions to this equation, which can be found using the quadratic formula. The roots are:

$z = \frac{-\alpha \pm \sqrt{\alpha^2 + 4(1 + \alpha)}}{2}$

Solving this equation gives the roots $1$ and $-1 - \alpha$.

For zero-stability, all roots must lie within the unit circle in the complex plane, which means their absolute values must be less than or equal to 1.

From the roots, we can see that the first root, 1, is always within the unit circle. However, the second root, $-1 - \alpha$ depends on the value of $\alpha$. The range of $\alpha$ for which the method is zero-stable is $\alpha = -2$ or $\alpha = 0$, or when $\alpha$ lies in the interval $-2 < \alpha < 0$.

Consistent: 

A linear multistep method is consistent if it satisfies the local truncation error condition. The coefficients are consistent if they satisfy the following conditions:

   $\sum_{i=0}^{2} (-1)^i \binom{2}{i} \alpha^{2-i} = 0$

   $\sum_{i=0}^{2} (-1)^i \binom{2}{i} (2-i) \alpha^{2-i-1} = 0$

The solutions to the consistency conditions are $\alpha = 1$. Therefore, the method is consistent for $\alpha = 1$.

Convergent: 

A linear multistep method is convergent if it is both zero-stable and consistent. Therefore, the range of $\alpha$ for which the method is convergent is the intersection of the ranges for zero-stability and consistency. Therefore, the method is convergent for $\alpha = 1$ and when $\alpha$ lies in the interval $-2 < \alpha < 0$.

A-stabe: 

A linear multistep method is A-stable if it is stable for all $z \leq 0$, where $z = h \lambda$ and $\lambda$ is an eigenvalue of the system matrix. For the given method, this condition is satisfied if $\alpha \geq -2$. Therefore, for the given method, this condition is satisfied if $\alpha \geq -2$. 

Second order: 

A linear multistep method is of the second order if the local truncation error is of order $h^3$. This condition is satisfied if the coefficients of the method satisfy the following additional condition:

$\sum_{i=0}^{2} (-1)^i \binom{2}{i} (2-i)(2-i-1) \alpha^{2-i-2} = 0$.

The equation simplifies to $2 = 0$, which has no solution. Therefore, the method is not of the second order for any $\alpha$. This means that the local truncation error is not of order $h^3$ for any $\alpha$, and the method does not achieve second order accuracy.

’‘’
#Question 2
# (a)

import numpy as np
import matplotlib.pyplot as plt

# Define the time variable
t = np.linspace(0, 3, 1000)

# Define the exact solution functions
x1_exact = np.exp(-100*t) + 2*np.exp(-t/10)
x2_exact = -np.exp(-100*t) + np.exp(-t/10)

# Plot the exact solution
plt.figure(figsize=(10,6))
plt.plot(t, x1_exact, label='x1(t)')
plt.plot(t, x2_exact, label='x2(t)')
plt.legend()
plt.title('Exact Solution')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()

# (b)

# Define the system matrix and initial condition
A = np.array([[-33.4, 66.6], [33.3, -66.7]])
x0 = np.array([3, 0])

# Define the step size and the number of steps
h = 1/10
steps = int(3/h)

# Initialize the solution array
x = np.zeros((steps+1, 2))
x[0] = x0

# Implement the forward Euler method
for n in range(steps):
    x[n+1] = x[n] + h * A @ x[n]

# Plot the numerical solution
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0, 3, steps+1), x[:, 0], label='x1(t)')
plt.plot(np.linspace(0, 3, steps+1), x[:, 1], label='x2(t)')
plt.legend()
plt.title('Numerical Solution (Forward Euler Method)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()

# (c)

# Define the system matrix and initial condition
A = np.array([[-33.4, 66.6], [33.3, -66.7]])
x0 = np.array([3, 0])

# Define the step size and the number of steps
h = 1/20
steps = int(3/h)

# Initialize the solution array
x = np.zeros((steps+1, 2))
x[0] = x0

# Implement the forward Euler method
for n in range(steps):
    x[n+1] = x[n] + h * A @ x[n]

# Plot the numerical solution
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0, 3, steps+1), x[:, 0], label='x1(t)')
plt.plot(np.linspace(0, 3, steps+1), x[:, 1], label='x2(t)')
plt.legend()
plt.title('Numerical Solution (Forward Euler Method, h=1/20)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()

#(d)

# Define the step size and the number of steps
h = 1/40
steps = int(3/h)

# Initialize the solution array
x = np.zeros((steps+1, 2))
x[0] = x0

# Implement the forward Euler method
for n in range(steps):
    x[n+1] = x[n] + h * A @ x[n]

# Plot the numerical solution
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0, 3, steps+1), x[:, 0], label='x1(t)')
plt.plot(np.linspace(0, 3, steps+1), x[:, 1], label='x2(t)')
plt.legend()
plt.title('Numerical Solution (Forward Euler Method, h=1/40)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()

#(e)

# Define the step size and the number of steps
h = 1/80
steps = int(3/h)

# Initialize the solution array
x = np.zeros((steps+1, 2))
x[0] = x0

# Implement the forward Euler method
for n in range(steps):
    x[n+1] = x[n] + h * A @ x[n]

# Plot the numerical solution
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0, 3, steps+1), x[:, 0], label='x1(t)')
plt.plot(np.linspace(0, 3, steps+1), x[:, 1], label='x2(t)')
plt.legend()
plt.title('Numerical Solution (Forward Euler Method, h=1/80)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()

#(f)

# Define the step size and the number of steps
h = 1/160
steps = int(3/h)

# Initialize the solution array
x = np.zeros((steps+1, 2))
x[0] = x0

# Implement the forward Euler method
for n in range(steps):
    x[n+1] = x[n] + h * A @ x[n]

# Plot the numerical solution
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0, 3, steps+1), x[:, 0], label='x1(t)')
plt.plot(np.linspace(0, 3, steps+1), x[:, 1], label='x2(t)')
plt.legend()
plt.title('Numerical Solution (Forward Euler Method, h=1/160)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()

(g)
'''
Visually speaking, we can see from the graphs that as the step size decreases, the numerical solution should become more accurate and closer to the exact solution. This is because the forward Euler method is a first-order method, and its error is proportional to the step size. Therefore, by halving the step size, we should roughly halve the error at each step.To explain this in terms of the eigenvalues of A:
'''  
eigenvalues = np.linalg.eigvals(A)
print(eigenvalues)
'''
The eigenvalues of the matrix A are -0.1 and -100. These eigenvalues are negative, which indicates that the exact solution to the system of differential equations should decay over time, as we can see in the exact solution.

The forward Euler method is an explicit method, and its stability depends on the step size and the eigenvalues of the system matrix. Specifically, the method is stable if the product of the step size and the maximum absolute eigenvalue is less than 2. In this case, the maximum absolute eigenvalue is 100.

For the step sizes h = 1/10, 1/20, 1/40, 1/80, and 1/160, the product of the step size and the maximum absolute eigenvalue is 10, 5, 2.5, 1.25, and 0.625, respectively. Therefore, the method is expected to be stable for h = 1/80 and h = 1/160, and unstable for the larger step sizes.

In the graphs, we have observed that the numerical solution oscillates and grows for larger step sizes, which is a sign of instability. For smaller step sizes, the numerical solution should be closer to the exact solution and not exhibit growing oscillations, indicating that the method is stable.

This analysis is consistent with the theory of the forward Euler method and the effect of the step size on its stability. By choosing an appropriate step size, we can control the accuracy and stability of the numerical solution.
'''

#Question 3

#(a)

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
k = 0.1  # time step
h = 0.02  # mesh size
x = np.arange(-1, 1+h, h)  # spatial grid
t = np.arange(0, 1+k, k)  # time grid
r = k / h**2  # ratio of time step to mesh size squared

# Initialize the solution array
u = np.zeros((len(t), len(x)))

# Set the initial condition
u[0, :] = np.exp(-100 * x**2)

# Set the boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Forward Euler method
for n in range(len(t)-1):
    u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])

# Plot the solution at t = 1
plt.figure(figsize=(10,6))
plt.plot(x, u[-1, :], label='u(1, x)')
plt.legend()
plt.title('Solution of the Heat Equation (Forward Euler Method)')
plt.xlabel('x')
plt.ylabel('u(1, x)')
plt.grid(True)
plt.show()

#(b)

# Define the parameters
k = 0.01  # time step
h = 0.02  # mesh size
x = np.arange(-1, 1+h, h)  # spatial grid
t = np.arange(0, 1+k, k)  # time grid
r = k / h**2  # ratio of time step to mesh size squared

# Initialize the solution array
u = np.zeros((len(t), len(x)))

# Set the initial condition
u[0, :] = np.exp(-100 * x**2)

# Set the boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Forward Euler method
for n in range(len(t)-1):
    u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])

# Plot the solution at t = 1
plt.figure(figsize=(10,6))
plt.plot(x, u[-1, :], label='u(1, x)')
plt.legend()
plt.title('Solution of the Heat Equation (Forward Euler Method, k=0.01)')
plt.xlabel('x')
plt.ylabel('u(1, x)')
plt.grid(True)
plt.show()

#(c)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Define the parameters
k = 0.001  # time step
h = 0.02  # mesh size
x = np.arange(-1, 1+h, h)  # spatial grid
t = np.arange(0, 1+k, k)  # time grid
r = k / (2*h**2)  # ratio of time step to twice the mesh size squared

# Initialize the solution array
u = np.zeros((len(t), len(x)))

# Set the initial condition
u[0, :] = np.exp(-100 * x**2)

# Set the boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Define the matrix for the linear system
A_upper = -r * np.ones(len(x)-3)
A_middle = (1 + 2*r) * np.ones(len(x)-2)
A_lower = -r * np.ones(len(x)-3)
A = np.vstack((np.hstack([0, A_upper]), A_middle, np.hstack([A_lower, 0])))

# Crank-Nicolson method
for n in range(len(t)-1):
    b = u[n, 1:-1] + r * (u[n, :-2] - 2*u[n, 1:-1] + u[n, 2:])
    u[n+1, 1:-1] = solve_banded((1, 1), A, b)

# Plot the solution at t = 1
plt.figure(figsize=(10,6))
plt.plot(x, u[-1, :], label='u(1, x)')
plt.legend()
plt.title('Solution of the Heat Equation (Crank-Nicolson Method, k=0.001)')
plt.xlabel('x')
plt.ylabel('u(1, x)')
plt.grid(True)
plt.show()

#(d)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Define the parameters
k = 0.0001  # time step
h = 0.02  # mesh size
x = np.arange(-1, 1+h, h)  # spatial grid
t = np.arange(0, 1+k, k)  # time grid
r = k / (2*h**2)  # ratio of time step to twice the mesh size squared

# Initialize the solution array
u = np.zeros((len(t), len(x)))

# Set the initial condition
u[0, :] = np.exp(-100 * x**2)

# Set the boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Define the matrix for the linear system
A_upper = -r * np.ones(len(x)-3)
A_middle = (1 + 2*r) * np.ones(len(x)-2)
A_lower = -r * np.ones(len(x)-3)
A = np.vstack((np.hstack([0, A_upper]), A_middle, np.hstack([A_lower, 0])))

# Crank-Nicolson method
for n in range(len(t)-1):
    b = u[n, 1:-1] + r * (u[n, :-2] - 2*u[n, 1:-1] + u[n, 2:])
    u[n+1, 1:-1] = solve_banded((1, 1), A, b)

# Plot the solution at t = 1
plt.figure(figsize=(10,6))
plt.plot(x, u[-1, :], label='u(1, x)')
plt.legend()
plt.title('Solution of the Heat Equation (Crank-Nicolson Method, k=0.0001)')
plt.xlabel('x')
plt.ylabel('u(1, x)')
plt.grid(True)
plt.show()

'''
(e)

The graphs obtained in parts (a) - (d) represent the numerical solutions of the heat equation for different time steps `k` while keeping the mesh size `h` constant. 

In part (a), with `k = 0.1`, the solution might not be accurate due to the relatively large time step. This is because the local truncation error, which is the error made in a single time step, is proportional to `k` for the methods used. Therefore, a larger `k` results in a larger local truncation error, leading to less accurate results.

In part (b), with `k = 0.01`, the solution is more accurate than in part (a) because the local truncation error is reduced due to the smaller time step. 

In part (c), with `k = 0.001`, the solution might not be accurate and the graph might not display correctly due to the instability of the numerical method. The stability of a numerical method is related to the choice of time step `k` and mesh size `h`. For the forward Euler method, the method is stable if `k` is less than or equal to `0.5 * h^2`. In this case, `k` is too large for the chosen `h` to ensure stability, leading to incorrect results.

In part (d), with `k = 0.0001`, the solution is expected to be accurate and stable because the time step is small enough to ensure both a small local truncation error and stability of the numerical method.

In conclusion, the choice of time step `k` significantly affects the stability and accuracy of the numerical solution. A smaller `k` reduces the local truncation error, leading to more accurate results, but it must also be small enough to ensure the stability of the numerical method.
'''
