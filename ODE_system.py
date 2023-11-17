import numpy as np
import pandas as pd
import time
import scipy.sparse.linalg as spla


from scipy.sparse import eye
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron

#-----------------------------------------DISCRETE LAPLACIAN--------------------------------------------
def discrete_laplacian(n):
    """
    Create a discrete Laplacian Matrix for a square grid of size n x n
    utilizing kroenecker product.
    Returns a 2D laplacian in sparse matrix form.
    """
    # Create a 1D discrete Laplacian
    laplacian_1d = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))

    # Create a 2D discrete Laplacian using the Kronecker product
    I_n = diags([1], [0], shape=(n, n))
    laplacian_2d = kron(laplacian_1d, I_n) + kron(I_n, laplacian_1d)

    return laplacian_2d

#---------------------------------------------STABILITY-------------------------------------------

# stability characteristic equation for Runge Kutta 4
def R(z):
    return 1 + z + 0.5*z**2 + (1/6)*z**3 + (1/24)*z**4

#----------------------------------------------------------------------------------------------

# Define the system of ODEs 
def funct(t, y, A):
    return -A @ y

#--------------------------------------Crank-Nicolson method------------------------------------------

def CN(t0, T, h, y0, f, A):
    """ 
     Crank-Nicolson implementation, takes as input :
     t0 --> initial time
     T --> final time
     h --> stepsize
     y0 --> vector containing initial values
     f --> function to compute
     A --> 2D laplacian
     Returns a vector of times and a matrix containing the solutions at each timestamp.
    """
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

# -------------------------------------------------------BDF3 method-----------------------------------------------------

def BDF3(t0, tN, h, y0, f, A):
    """ 
     BDF3 implementation, takes as input :
     t0 --> initial time
     T --> final time
     h --> stepsize
     y0 --> vector containing initial values
     f --> function to compute
     A --> 2D laplacian
     Returns a vector of times and a matrix containing the solutions at each timestamp.
    """
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

def main():
    """ 
     define the IVP parameters and call the functions to solve
     the problem at requested step sizes.
    """

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
    # initial value vector
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

    for row in TABLE:
        print(row) 
        


if __name__ == '__main__':
	main()
