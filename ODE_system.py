import numpy as np
import time
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import expm

from scipy.integrate import solve_ivp

import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy.sparse import csc_matrix


accurate = np.genfromtxt(open("exact_sol.csv"), delimiter=",", dtype=float)
print(accurate)

nx = 100

# Create the grid
# This might require a custom implementation depending on what numgrid('S', nx) does in your case
# For now, let's assume it's a simple 2D grid
x = np.linspace(0, 0, nx)
y = np.linspace(0, 0, nx)
# returns two matrices X = xy, Y = yx
X, Y = np.meshgrid(x, y)

# Compute the discrete Laplacian
A = sp.diags([-1, -1, 4, -1, -1], [-nx, -1, 0, 1, nx], shape=(nx*nx, nx*nx))
print(A)
# Scale the matrix as in the MATLAB code
A *= (nx - 1)**2

# Find the largest magnitude eigenvalue
# Using 'LM' (Largest Magnitude) mode 
lambda_, _ = spla.eigs(A, k=1, which='LM')
lambda_ = -lambda_[0].real

print("Largest magnitude eigenvalue:", lambda_)

# for stability h must be less then this limit value
h_RK = -2.78 / lambda_
print(f'step size must be less then {h_RK:9.4e} for stability ')

n = A.shape[0]  
y0 = np.ones(n)


# Define the ODE function
def funct(t, y, A):
    return -A @ y

#  Solve the ODE
#  The function scipy.integrate.solve_ivp uses the method RK45 by default,
#  similar the method used by Matlab's function ODE45 as both use the
#  Dormand-Pierce formulas with fourth-order method accuracy.
tspan = [0, 0.1]
tStart = time.time()
sol = solve_ivp(funct, tspan, y0, args=(A,), method='RK45')
tEnd = time.time()

CPU_time_ode45 = tEnd - tStart
num_steps_ode45 = len(sol.t)

# Error calculation
err_ode45 = np.max(np.abs(sol.y[:, -1] - accurate))

print(err_ode45)

# Display results
run1 = ["run 1", "rk45", num_steps_ode45, CPU_time_ode45]
print(run1)

# ----------------------------------------------------Crank-Nicolson method------------------------------------------
h = [0.001, 0.0001, 0.00001]
CPU_time_CN = []
num_steps_CN = []
err_CN = []
y_start2 = []
y_start3 = []

for hi in h:
    this_A = eye(n) + 0.5 * hi * A
    # number of steps
    N = round(0.1 / hi)
    y_cn = np.zeros((n, N + 1))
    y_cn[:, 0] = y0

    tStart = time.time()
    for i in range(N):
        b = (eye(n) - 0.5 * hi * A) @ y_cn[:, i]
        y_cn[:, i + 1], _ = spla.cg(this_A, b, tol=hi**3, maxiter=500)
    tEnd = time.time()

    CPU_time_CN.append(tEnd - tStart)
    num_steps_CN.append(N)
    y_start2.append(y_cn[:, 1])
    y_start3.append(y_cn[:, 2])
    #err_CN.append(np.max(np.abs(y_cn[:, -1] - y_ex)))

# -------------------------------------------------------BDF3 method-----------------------------------------------------
CPU_time_BDF3 = []
num_steps_BDF3 = []
err_BDF3 = []

for j, hi in enumerate(h):
    N = round(0.1 / hi)
    this_A = eye(n) + (6/11) * hi * A
    y_bdf = np.zeros((n, N + 1))
    y_bdf[:, 0] = y0
    y_bdf[:, 1] = y_start2[j]
    y_bdf[:, 2] = y_start3[j]

    tStart = time.time()
    for i in range(1, N - 2):
        b = (18/11) * y_bdf[:, i + 2] - (9/11) * y_bdf[:, i + 1] + (2/11) * y_bdf[:, i]
        y_bdf[:, i + 3], _ = spla.cg(this_A, b, tol=hi**3, maxiter=500)
    tEnd = time.time()

    CPU_time_BDF3.append(tEnd - tStart)
    num_steps_BDF3.append(N)
    #err_BDF3.append(np.max(np.abs(y_bdf[:, -1] - y_ex)))

# Prepare the final table
runs = [
    ["run 1", "ode45", num_steps_ode45, CPU_time_ode45],
    ["run 2", "crank-nicolson", num_steps_CN[0], CPU_time_CN[0]],
    ["run 1", "crank-nicolson", num_steps_CN[1], CPU_time_CN[1]],
    ["run 1", "crank-nicolson", num_steps_CN[2], CPU_time_CN[2]],
    ["run 1", "BDF3", num_steps_BDF3[0], CPU_time_BDF3[0]],
    ["run 1", "BDF3", num_steps_BDF3[1], CPU_time_BDF3[1]],
    ["run 1", "BDF3", num_steps_BDF3[2], CPU_time_BDF3[2]]
    
]

head = ["RUN", "METHOD", "NUMBER OF STEPS", "ERROR", "CPU TIME"]
TABLE = [head] + runs

# Display the table
# You can use a loop or a library like pandas for a nicer display
for row in TABLE:
    print(row)
""" 
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

 """
