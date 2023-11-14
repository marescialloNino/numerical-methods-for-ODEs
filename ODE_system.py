import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from scipy.sparse import eye
from scipy.integrate import solve_ivp



accurate_final_solution = np.genfromtxt(open("exact_sol.csv"), delimiter=",", dtype=float)

from scipy.sparse import diags, kron
from scipy.sparse.linalg import spsolve

def discrete_laplacian(n):
    """Create a discrete Laplacian for a square grid of size n x n."""
    # Create a 1D discrete Laplacian
    laplacian_1d = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))

    # Create a 2D discrete Laplacian using the Kronecker product
    I_n = diags([1], [0], shape=(n, n))
    laplacian_2d = kron(laplacian_1d, I_n) + kron(I_n, laplacian_1d)

    return laplacian_2d

nx = 100  # grid size
A = -discrete_laplacian(nx-2) * (nx - 1)**2  # scale the Laplacian









""" nx = 100
# Compute the discrete Laplacian
A = -(sp.diags([1, 1, -4, 1, 1], [-(nx-1), -1, 0, 1, (nx-1)], shape=((nx-2)**2, (nx-2)**2))*(nx - 1)**2)
# Scale the matrix 
 """

#------------------------------------------------------TEST LAPLACIAN MATRIX-------------------------------------------
""" A = sp.diags([-1, -1, 4, -1, -1], [-4, -1, 0, 1, 4], shape=(4*4,4*4))
A_dense = A.toarray()
print(A_dense) """
#----------------------------------------------------------------------------------------------------------------------
# Find the largest magnitude eigenvalue
# Using 'LM' (Largest Magnitude) mode 
# spla.eigs() returns an array of eigenvalues (ordered by largest modulus)and one of eigenvectors
lambda_, _ = spla.eigs(A, k=1, which='LM')
lambda_ = lambda_[0].real

print("Largest magnitude eigenvalue:", lambda_)


#---------------------------------------------STABILITY-------------------------------------------
# for stability h must be less then this limit value
h_max = -2.78 / lambda_
print(f'step size must be less then {h_max:9.4e} for stability ')

""" # stability characteristic equation
def R(z):
    return 1 + z + 0.5*z**2 + (1/6)*z**3 + (1/24)*z**4

# Create a grid of complex numbers
re = np.linspace(-5, 5, 500)
im = np.linspace(-5, 5, 500)
Re, Im = np.meshgrid(re, im)
Z = Re + 1j*Im

# Evaluate the stability function
R_Z = R(Z)

# Identify points in the stability region
stability_region = np.abs(R_Z) <= 1

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(Re, Im, stability_region, shading='auto',cmap='gray_r')
plt.title('Stability Region for 4th Order Runge-Kutta')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.colorbar(label='|R(z)|')
plt.axis('equal')
plt.grid()
plt.show() """

#-------------------------------------------------------------------------------------------------------------
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
run1 = ["run 1", "rk45", num_steps_ode45,err_ode45, CPU_time_ode45]
print(run1)

# ----------------------------------------------------Crank-Nicolson method------------------------------------------
h = [0.001, 0.0001, 0.00001]
CPU_time_CN = []
num_steps_CN = []
err_CN = []
y_start2 = []
y_start3 = []

for step in h:
    this_A = eye(n) + 0.5 * step * A
    # number of steps
    N = round(0.1 / step)
    y_cn = np.zeros((n, N))
    y_cn[:, 0] = y0

    tStart = time.time()
    for i in range(N-1):
        b = (eye(n) - 0.5 * step * A) @ y_cn[:, i]
        y_cn[:, i + 1], _ = spla.cg(this_A, b, tol=step**3, maxiter=500)
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

for step in h:

    N = round(0.1 / step)
    this_A = eye(n) + (6/11) * step * A
    y_bdf = np.zeros((n, N+1))

    j=0
    y_bdf[:, 0] = y0
    y_bdf[:, 1] = y_start2[j]
    y_bdf[:, 2] = y_start3[j]
    j+=1
    tStart = time.time()
    for i in range(1, N - 3):
        b = (18/11) * y_bdf[:, i + 2] - (9/11) * y_bdf[:, i + 1] + (2/11) * y_bdf[:, i]
        y_bdf[:, i + 3], _ = spla.cg(this_A, b, tol=step**3, maxiter=1000)
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