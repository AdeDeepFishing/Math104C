#problem 1
#testing

import numpy as np
from scipy.optimize import fsolve

# Define the function f(t, u) representing the differential equation
def f(t, u):
    return -3 * t * u - u ** 2

# Forward Euler method
def forward_euler(h, t0, u0, target_t):
    t, u = t0, u0
    while t <= target_t:
        u += h * f(t, u)
        t += h
    return u

# Backward Euler method
def backward_euler(h, t0, u0, target_t):
    t, u = t0, u0
    while t <= target_t:
        def g(u_new):
            return -u_new + u + h * f(t + h, u_new)

        u_new = fsolve(g, u)[0]
        u, t = u_new, t + h
    return u

# Trapezoidal method
def trapezoidal(h, t0, u0, target_t):
    t, u = t0, u0
    while t < target_t:
        # Define the auxiliary function g for the Trapezoidal method
        def g(u_new):
            return u_new - u - 0.5 * h * (f(t + h, u_new) + f(t, u))

        # Extract the single value from the array
        u_new = fsolve(g, u)[0]
        u, t = u_new, t + h
    return u

# Leapfrog method
def leapfrog(h, t0, u0, target_t):
    u1 = forward_euler(h, t0, u0, t0 + h)
    t, u_old, u = t0 + h, u0, u1
    while t <= target_t - h:
        u_new = u_old + 2 * h * f(t, u)
        u_old, u, t = u, u_new, t + h
    return u_new

h_values = [0.1, 0.05, 0.025]
methods = [forward_euler, backward_euler, trapezoidal, leapfrog]
t0, u0, target_t = 1, 0.5, 2

results = {}
for method in methods:
    method_name = method.__name__
    results[method_name] = []
    for h in h_values:
        u2 = method(h, t0, u0, target_t)
        results[method_name].append(u2)

# Compute the ratios for each method
ratios = {}
for method in methods:
    method_name = method.__name__
    rUp = (results[method_name][0] - results[method_name][1])
    rDown = (results[method_name][1] - results[method_name][2])
    r = rUp / rDown
    ratios[method_name] = r

print("Results:")
print(results)
print("\nRatios:")
print(ratios)

'''
Question 2:

(a)

We have the first-order linear ODE:

$u'(t) = -e^{-t} * u(t), u(0) = 1$

The integrating factor is:

$e^{\int e^{-t} dt}$

Which simplifies to:

$e^{-e^{-t}}$

Now, multiply the ODE by the integrating factor:

$e^{-e^{-t}}u'(t) + e^{-e^{-t}}(-e^{-t})u(t) = 0$

The left-hand side of the equation is now the derivative of the product of the integrating factor and u(t) with respect to t:

$\frac{d(u(t)e^{-e^{-t}})}{dt} = 0$

Integrate both sides with respect to t:

$u(t)e^{-e^{-t}} = C$

Now, solve for $u(t)$:

$u(t) = Ce^{e^{-t}}$

Using the initial condition $u(0) = 1$:

$u(0) = Ce^{e^0} = Ce^1$

Since $u(0) = 1$, we have:

$1 = Ce^1$

Thus, $C = \frac{1}{e}$.

The exact solution for the initial value problem in LaTeX format is:

$u(t) = \frac{1}{e}e^{e^{-t}}$
'''

#(b)
def f(t, u):
    return -np.exp(-t) * u

t0, u0, target_t = 0, 1, 1

results = {}
for method in methods:
    method_name = method.__name__
    results[method_name] = []
    for h in h_values:
        u1 = method(h, t0, u0, target_t)
        results[method_name].append(u1)

print("Numerical Results:")
print(results)


#(c)
exact_u1 = (1 / np.e) * np.exp(np.exp(-1))

ratios_r1 = {}
ratios_r2 = {}
for method in methods:
    method_name = method.__name__
    r1 = (results[method_name][0] - exact_u1) / (results[method_name][1] - exact_u1)
    r2 = (results[method_name][1] - exact_u1) / (results[method_name][2] - exact_u1)
    ratios_r1[method_name] = r1
    ratios_r2[method_name] = r2

print("\nRatios r_1:")
print(ratios_r1)
print("\nRatios r_2:")
print(ratios_r2)
