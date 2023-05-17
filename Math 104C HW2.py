'''
Problem 1:

(a)

First, we rewrite the differential equation:

$u'(t) = 1 + u(t) / t$.

This is a first-order linear differential equation, which can be written in the standard form as:

$u'(t) - (1/t)u(t) = 1$.

Now, we will solve this equation using an integrating factor. The integrating factor (IF) is given by the exponential of the integral of the coefficient of u(t), which is -1/t in this case:

$IF(t) = exp(-integral(1/t) dt) = exp(-ln(t)) = 1/t$.

Now, multiply the whole equation by the integrating factor:

$(1/t)u'(t) - (1/t^2)u(t) = 1/t$.

The left side of this equation is now the derivative of the product of u(t) and the integrating factor:

$d(u(t)/t)/dt = 1/t$.

Integrate both sides with respect to t:

$integral (d(u(t)/t)) dt = integral (1/t) dt$,

$u(t)/t = ln(t) + C$.

To find the constant C, we use the initial condition $u(1) = 2$:

$2 = ln(1) + C$,

$C = 2$.

Now, we can write the exact solution to the initial value problem:

$u(t) = t(ln(t) + 2)$.
'''


#(b)
import numpy as np

# Define the function f(t, u) from the differential equation
def f(t, u):
    return 1 + u / t

# Define the exact solution for comparison purposes
def exact_solution(t):
    return t * (np.log(t) + 2)

# Implement the classical fourth-order Runge-Kutta method
def runge_kutta_4th_order(t0, u0, k, N):
    u = u0  # Initialize u with the initial condition
    t = t0  # Initialize t with the starting value

    # Loop N times, moving forward in time by k in each iteration
    for _ in range(N):
        # Calculate the four k values used in the Runge-Kutta method
        k1 = k * f(t, u)
        k2 = k * f(t + k / 2, u + k1 / 2)
        k3 = k * f(t + k / 2, u + k2 / 2)
        k4 = k * f(t + k, u + k3)

        # Update u using the weighted average of the k values
        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # Update t by adding the step size k
        t += k

    return u

t0 = 1  # Initial value of t
u0 = 2  # Initial value of u
step_sizes = [0.2, 0.1, 0.05, 0.025]  # List of step sizes to use

# Print the header for the output table
print("k\t\te_k")

# Loop over the different step sizes
for k in step_sizes:
    N = int(1 / k)  # Calculate the number of iterations for this step size
    max_error = 0  # Initialize the maximum error to 0

    # Loop over the time steps from 0 to N
    for n in range(N + 1):
        t_n = t0 + n * k  # Calculate the value of t at this step
        u_exact = exact_solution(t_n)  # Calculate the exact value of u at t_n
        u_approx = runge_kutta_4th_order(t0, u0, k, n)  # Calculate the approximate value of u at t_n using Runge-Kutta method
        error = abs(u_exact - u_approx)  # Calculate the error between the exact and approximate values
        max_error = max(max_error, error)  # Update the maximum error if this error is larger

    # Print the step size and the maximum error for this step size
    print(f"{k}\t\t{max_error}")


#(c)

# Define the function f(t, u) from the differential equation
def f(t, u):
    return 1 + u / t

# Define the exact solution for comparison purposes
def exact_solution(t):
    return t * (np.log(t) + 2)

# Implement the second-order Runge-Kutta method
def runge_kutta_2nd_order(t0, u0, k, N):
    u = u0  # Initialize u with the initial condition
    t = t0  # Initialize t with the starting value

    # Loop N times, moving forward in time by k in each iteration
    for _ in range(N):
        # Calculate the two F values used in the second-order Runge-Kutta method
        F1 = f(t, u)
        F2 = f(t + k, u + k * F1)

        # Update u using the average of the F values
        u += k * (F1 + F2) / 2
        # Update t by adding the step size k
        t += k

    return u

t0 = 1  # Initial value of t
u0 = 2  # Initial value of u
step_sizes = [0.2, 0.1, 0.05, 0.025]  # List of step sizes to use

# Print the header for the output table
print("k\t\te_k")

# Loop over the different step sizes
for k in step_sizes:
    N = int(1 / k)  # Calculate the number of iterations for this step size
    max_error = 0  # Initialize the maximum error to 0

    # Loop over the time steps from 0 to N
    for n in range(N + 1):
        t_n = t0 + n * k  # Calculate the value of t at this step
        u_exact = exact_solution(t_n)  # Calculate the exact value of u at t_n
        u_approx = runge_kutta_2nd_order(t0, u0, k, n)  # Calculate the approximate value of u at t_n using the 2nd-order Runge-Kutta method
        error = abs(u_exact - u_approx)  # Calculate the error between the exact and approximate values
        max_error = max(max_error, error)  # Update the maximum error if this error is larger

    # Print the step size and the maximum error for this step size
    print(f"{k}\t\t{max_error}")


#problem 2
#(a)
import numpy as np
import matplotlib.pyplot as plt

def f(t, u):
    return u

def exact_solution(t):
    return np.exp(t)

def two_step_method(t0, u0, k, N):
    u = np.zeros(N+1)
    t = np.zeros(N+1)
    u[0] = u0
    t[0] = t0
    u[1] = np.exp(k)

    for n in range(N-1):
        t[n+2] = t[n+1] + k
        u[n+2] = (3/2) * u[n+1] - (1/2) * u[n] + k * ((5/4) * f(t[n+1], u[n+1]) - (3/4) * f(t[n], u[n]))

    return t, u

t0 = 0
u0 = 1
k = 0.01
N = int(1/k)

t, u = two_step_method(t0, u0, k, N)
u_exact = exact_solution(t)

plt.plot(t, u, label='Numerical Solution')
plt.plot(t, u_exact, label='Exact Solution', linestyle='dashed')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.title('Two-step Method: Numerical Solution and Exact Solution')
plt.show()

(b)
def alternative_two_step_method(t0, u0, k, N):
    u = np.zeros(N+1)
    t = np.zeros(N+1)
    u[0] = u0
    t[0] = t0
    u[1] = np.exp(k)

    for n in range(N-1):
        t[n+2] = t[n+1] + k
        u[n+2] = 3 * u[n+1] - 2 * u[n] + k * ((1/2) * f(t[n+1], u[n+1]) - (3/2) * f(t[n], u[n]))

    return t, u

t_alt, u_alt = alternative_two_step_method(t0, u0, k, N)

plt.plot(t_alt, u_alt, label='Numerical Solution')
plt.plot(t_alt, exact_solution(t_alt), label='Exact Solution', linestyle='dashed')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.title('Alternative Two-step Method: Numerical Solution and Exact Solution')
plt.show()


#(c)
t_a, u_a = two_step_method(t0, u0, k, N)
t_b, u_b = alternative_two_step_method(t0, u0, k, N)
u_exact_a = exact_solution(t_a)
u_exact_b = exact_solution(t_b)

error_a = np.max(np.abs(u_a - u_exact_a))
error_b = np.max(np.abs(u_b - u_exact_b))

plt.plot(t_a, u_a, label='Method A Numerical Solution')
plt.plot(t_b, u_b, label='Method B Numerical Solution', linestyle='dotted')
plt.plot(t_a, u_exact_a, label='Exact Solution', linestyle='dashed')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.title('Comparison of Two-step Methods: Numerical Solutions and Exact Solution')
plt.show()

print(f"Maximum error for Method A: {error_a}")
print(f"Maximum error for Method B: {error_b}")


'''
The difference observed between the two methods may be explained by zero-stability. Zero-stability is a concept that addresses the behavior of a numerical method when subject to small perturbations in the initial data. A zero-stable method will have a solution that is close to the true solution when the initial data are perturbed by a small amount.

In the context of our problem, it is likely that the first method (part a) is more stable than the second method (part b), which means that the first method is less sensitive to small perturbations in the initial data. This can result in better accuracy and a closer fit to the exact solution. The second method, being less stable, may produce a solution with larger deviations from the exact solution.
'''
