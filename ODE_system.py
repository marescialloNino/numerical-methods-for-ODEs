import numpy as np
import time
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import expm
from scipy.integrate import odeint

# Generation of A and y0
nx = 100
G = np.array([[0]])
A = (np.array([[4, -1, 0, -1],
               [-1, 4, -1, 0],
               [0, -1, 4, -1],
               [-1, 0, -1, 4]]) * (nx - 1) ** 2)
n = len(A)
lambda_ = -np.linalg.eigvals(A).max()
h_RK = -2.78 / lambda_
print("h_RK: " + str(h_RK))

# y0 and y_exact
y0 = np.ones(n)
y_ex = np.dot(expm(-0.1 * A), np.ones(n))

# Solution with odeint
tspan = np.array([0, 0.1])
funct = lambda y, A: -np.dot(A, y)
tStart = time.time()
t = np.linspace(tspan[0], tspan[1], 100)
y = odeint(funct, y0, t, args=(A,))
tEnd = time.time()
CPU_time_odeint = tEnd - tStart
num_steps_odeint = len(t)

# Error
err_odeint = np.max(np.abs(y[-1, :] - y_ex))
run1 = ["run 1", "odeint", num_steps_odeint, err_odeint, CPU_time_odeint]
print(run1)

# Crank Nicolson
h = np.array([0.001, 0.0001, 0.00001])
j = 0
y_start2 = np.zeros((n, len(h)))
y_start3 = np.zeros((n, len(h)))
CPU_time_CN = np.zeros(len(h))
num_steps_CN = np.zeros(len(h))
for hi in h:
    this_A = eye(n) + 0.5 * hi * A
    N = int(round(0.1 / hi))
    y_cn = np.zeros((n, N + 1))
    y_cn[:, 0] = y0
    tStart = time.time()
    for i in range(N):
        b = np.dot(eye(n) - 0.5 * hi * A, y_cn[:, i])
        y_cn[:, i + 1] = spsolve(this_A, b)
    tEnd = time.time()
    CPU_time_CN[j] = tEnd - tStart
    num_steps_CN[j] = N
    y_start2[:, j] = y_cn[:, 1]
    y_start3[:, j] = y_cn[:, 2]

    # Error
    err_CN = np.max(np.abs(y_cn[:, -1] - y_ex))
    run2 = ["run 2", "CN", num_steps_CN[0], err_CN[0], CPU_time_CN[0]]
    run3 = ["run 3", "CN", num_steps_CN[1], err_CN[1], CPU_time_CN[1]]
    run4 = ["run 4", "CN", num_steps_CN[2], err_CN[2], CPU_time_CN[2]]
    print(run2)
    print(run3)
    print(run4)

# BDF3
h = np.array([0.001, 0.0001, 0.00001])
j = 0
CPU_time_BDF3 = np.zeros(len(h))
num_steps_BDF3 = np.zeros(len(h))
for hi in h:
    N = round(0.1 / hi)
    this_A = eye(n) + (6 / 11) * hi * A
    y_bdf = np.zeros((n, N + 1))
    y_bdf[:, 0] = y0
    y_bdf[:, 1] = y_start2[:, j]
    y_bdf[:, 2] = y_start3[:, j]
    tStart = time.time()
    for i in range(N - 2):
        b = (18 / 11) * y_bdf[:, i + 2] - (9 / 11) * y_bdf[:, i + 1] + (2 / 11) * y_bdf[:, i]
        y_bdf[:, i + 3] = spsolve(this_A, b)
    tEnd = time.time()
    CPU_time_BDF3[j] = tEnd - tStart
    num_steps_BDF3[j] = N

    # Error
    err_BDF3 = np.max(np.abs(y_bdf[:, -1] - y_ex))
    run5 = ["run 5", "BDF3", num_steps_BDF3[0], err_BDF3[0], CPU_time_BDF3[0]]
    run6 = ["run 6", "BDF3", num_steps_BDF3[1], err_BDF3[1], CPU_time_BDF3[1]]
    run7 = ["run 7", "BDF3", num_steps_BDF3[2], err_BDF3[2], CPU_time_BDF3[2]]
    print(run5)
    print(run6)
    print(run7)

# Final table
head = ["RUN", "METHOD", "NUMBER OF STEPS", "ERROR", "CPU TIME"]
TABLE = [head, run1, run2, run3, run4, run5, run6, run7]
print(TABLE)


