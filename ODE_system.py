import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from scipy.sparse import eye
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron




#---------------------------------DISCRETE LAPLACIAN--------------------------------------------
def discrete_laplacian(n):
    """Create a discrete Laplacian Matrix for a square grid of size n x n."""
    # Create a 1D discrete Laplacian
    laplacian_1d = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))

    # Create a 2D discrete Laplacian using the Kronecker product
    I_n = diags([1], [0], shape=(n, n))
    laplacian_2d = kron(laplacian_1d, I_n) + kron(I_n, laplacian_1d)

    return laplacian_2d

#---------------------------------------------STABILITY-------------------------------------------


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


# Define the system ODE function
def funct(t, y, A):
    return -A @ y

# ----------------------------------------------------Crank-Nicolson method------------------------------------------

def CN(t0, T, h, y0, f, A):
    N = int((T-t0)/h)
    N = N+1
    t = np.linspace(t0, T, N)
    m = np.shape(y0)[0]
    y = np.zeros([N, m])
    y[0] = y0
    thisA = eye(np.shape(A)[0])+(h/2)*A
    for i in range(N-1):
        b = (eye(np.shape(A)[0])-(h/2)*A).dot(y[i])
        y[i+1] = spla.cg(A=thisA, b=b.T, x0=y[i], tol=h**3)[0]
    return t, y

""" for step in h:
    this_A = eye(n) + 0.5 * step * A
    # number of steps
    N = round(0.1 / step)
    y_cn = np.zeros((n, N))
    y_cn[:, 0] = y0

    tStart = time.time()
    for i in range(N-1):
        b = (eye(n) - 0.5 * step * A).dot(y_cn[:, i])
        y_cn[:, i + 1], _ = spla.cg(this_A, b, tol=step**3, maxiter=1000)
    tEnd = time.time()

    CPU_time_CN.append(tEnd - tStart)
    num_steps_CN.append(N)
    y_start2.append(y_cn[:, 1])
    y_start3.append(y_cn[:, 2])
    err_CN.append(np.max(np.abs(y_cn[:, -1] - accurate_final_solution)))
 """

# -------------------------------------------------------BDF3 method-----------------------------------------------------


# I define BDF3 method for this particular function
def BDF3(t0, tN, h, y0, f, A):
    N = int((tN-t0)/h)
    N = N+1
    t = np.linspace(t0, tN, N)
    m = np.shape(y0)[0]
    y = np.zeros([N, m])
    y[0] = y0
    thisA = eye(np.shape(A)[0])+(h*6/11)*A

    solve = solve_ivp(f, [t0, t0+2*h], y0, t_eval=np.linspace(t0, t0+2*h, 3), args=(A,))
    y[:3] = solve.y.T
    for i in range(N-3):
        b = (18/11)*y[i+2]-(9/11)*y[i+1]+(2/11)*y[i]
        y[i+3] = spla.cg(A=thisA, b=b.T, x0=y[i], tol=h**3)[0]
    return t, y

""" for step in h:

    N = round(0.1 / step)
    this_A = eye(n) + (6/11) * step * A
    y_bdf = np.zeros((n, N+1))

    # set initial 3 values to solution matrix of BDF3
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
 """


def main():

    nx = 100  # grid size
    A = -discrete_laplacian(nx-2) * (nx - 1)**2  # scale the Laplacian

    # Find the largest magnitude eigenvalue
    # Using 'LM' (Largest Magnitude) mode 
    # spla.eigs() returns an array of eigenvalues (ordered by largest modulus)and one of eigenvectors
    lambda_, _ = spla.eigs(A, k=1, which='LM')
    lambda_ = -lambda_[0].real

    print("Largest magnitude eigenvalue:", lambda_)

    # for stability h must be less then this limit value
    h_max = -2.78 / lambda_
    print(f'step size must be less then {h_max:9.4e} for stability ')

    # number of rows in the matrix A
    n = A.shape[0]  
    y0 = np.ones(n)
    
    t0 = 0
    T = 0.1
    h = [0.001, 0.0001, 0.00001]
    runs = []
    accurate_final_solution = np.genfromtxt(open("exact_sol.csv"), delimiter=",", dtype=float)

    #  The function scipy.integrate.solve_ivp uses the method RK45 by default,
    #  similar the method used by Matlab's function ODE45 as both use the
    #  Dormand-Pierce formulas with fourth-order method accuracy.
    
    tStart = time.time()
    sol = solve_ivp(funct,[t0,T], y0, args=(A,), method='RK45')
    tEnd = time.time()

    CPU_time_ode45 = tEnd - tStart
    num_steps_ode45 = len(sol.t)
    err_ode45 = np.max(np.abs(sol.y[:, -1] - accurate_final_solution))

    runs.append(["rk45", num_steps_ode45, err_ode45, CPU_time_ode45]) 

    
    for step in h:
        num_steps = int((T-t0)/step)
        start = time.time()
        yCN_final = CN(t0, T, step, y0, funct, A=A)[1][-1, :]
        end = time.time()
        CPU_time = end-start
        errCN = np.linalg.norm(accurate_final_solution - yCN_final, ord=np.inf)
        runs.append(["Crank-Nicolson",num_steps,errCN,CPU_time])

        
    for step in h:
        num_steps = int((T-t0)/step)
        tic = time.time()
        yNBDF_final = BDF3(t0, T, step, y0, funct, A=A)[1][-1, :]
        toc = time.time()
        CPU_time = toc-tic
        errBDF3 = np.linalg.norm(accurate_final_solution - yNBDF_final, ord=np.inf)
        runs.append(["BDF3",num_steps,errBDF3,CPU_time])

    head = ["METHOD", "NUMBER OF STEPS", "ERROR", "CPU TIME"]
    TABLE = [head] + runs

    # Display the table
    for row in TABLE:
        print(row) 
        


if __name__ == '__main__':
	main()
