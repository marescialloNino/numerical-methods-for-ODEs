import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy.sparse import eye
from scipy.integrate import solve_ivp



accurate_final_solution = np.genfromtxt(open("exact_sol.csv"), delimiter=",", dtype=float)

nx = 100

# Compute the discrete Laplacian
A = -sp.diags([1, 1, -4, 1, 1], [-nx, -1, 0, 1, nx], shape=((nx-2)**2, (nx-2)**2))
# Scale the matrix 
A *= (nx - 1)**2

#------------------------------------------------------TEST LAPLACIAN MATRIX-------------------------------------------
""" A = sp.diags([-1, -1, 4, -1, -1], [-4, -1, 0, 1, 4], shape=(4*4,4*4))
A_dense = A.toarray()
print(A_dense) """
#----------------------------------------------------------------------------------------------------------------------
# Find the largest magnitude eigenvalue
# Using 'LM' (Largest Magnitude) mode 
lambda_, _ = spla.eigs(A, k=1, which='LM')
lambda_ = lambda_[0].real

print("Largest magnitude eigenvalue:", lambda_)

# for stability h must be less then this limit value
h_RK = -2.78 / lambda_
print(f'step size must be less then {h_RK:9.4e} for stability ')

# number of rows in the matrix A
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
err_ode45 = np.max(np.abs(sol.y[:, -1] - accurate_final_solution))

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
    err_CN.append(np.max(np.abs(y_cn[:, -1] - accurate_final_solution)))

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
    err_BDF3.append(np.max(np.abs(y_bdf[:, -1] - accurate_final_solution)))

# Prepare the final table
runs = [
    ["run 1", "ode45", num_steps_ode45, CPU_time_ode45],
    ["run 2", "crank-nicolson", num_steps_CN[0],err_CN[0], CPU_time_CN[0]],
    ["run 1", "crank-nicolson", num_steps_CN[1], err_CN[1],CPU_time_CN[1]],
    ["run 1", "crank-nicolson", num_steps_CN[2],err_CN[2], CPU_time_CN[2]],
    ["run 1", "BDF3", num_steps_BDF3[0], err_BDF3[0],CPU_time_BDF3[0]],
    ["run 1", "BDF3", num_steps_BDF3[1],err_BDF3[1], CPU_time_BDF3[1]],
    ["run 1", "BDF3", num_steps_BDF3[2],err_BDF3[2], CPU_time_BDF3[2]]
    
]

head = ["RUN", "METHOD", "NUMBER OF STEPS", "ERROR", "CPU TIME"]
TABLE = [head] + runs

# Display the table
for row in TABLE:
    print(row)