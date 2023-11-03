import math
import matplotlib.pyplot as plt

# Constants
T_0 = 0
Y_0 = 1

T_FINAL = 5
h = 0.05
h_EXACT = 0.01

# Functions
def f(t, y):
    return -5*y

def euler_step(t_i, y_i, h, f):
    return y_i + h*f(t_i, y_i)

def rk4_step(t_i, y_i, h, f):
    k1 = f(t_i, y_i)
    k2 = f(t_i + h/2, y_i + k1*h/2)
    k3 = f(t_i + h/2, y_i + k2*h/2)
    k4 = f(t_i + h, y_i + k3*h)
    return y_i + h/6*(k1 + 2*k2 + 2*k3 + k4)

def y_exact(t):
    
    return math.exp(-5*t)


# Computing exact result
t = T_0

ys_exact = []
ts_exact = []

while t < T_FINAL:
    ts_exact.append(t)
    ys_exact.append(y_exact(t))
    t += h_EXACT


# Computing approximate results
ts = [T_0]
ys = [Y_0]
ys_euler = [Y_0]

y = Y_0
y_euler = Y_0
t = T_0

while t < T_FINAL:
    # Solving with Runge-Kutta
    y = rk4_step(t, y, h, f)

    # Solving with Euler
    y_euler = euler_step(t, y_euler, h, f)

    # Increasing t
    t += h

    # Appending results
    ts.append(t)
    ys.append(y)
    ys_euler.append(y_euler)


# Plotting
plt.plot(ts, ys, color='red', marker='o', linewidth=0.0, label='RK4')
plt.plot(ts, ys_euler, color='green', marker='o', linewidth=0.0, label='Euler')
plt.plot(ts_exact, ys_exact, color='blue', label='Exact')
plt.legend()
plt.show()

