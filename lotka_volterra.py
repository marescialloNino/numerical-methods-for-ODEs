
import numpy as np
import matplotlib.pyplot as plt

# Defining the General Model
# X = [x,y]

def Lotka_Volterra(X,alpha,beta,gamma,delta):
   
    Xdot = np.array([alpha*X[0] - beta*X[0]*X[1], delta*X[0]*X[1] - gamma*X[1]])
    
    return Xdot



'''
Defining the Runge Kutta 4 Method for systems of ODEs, takes as parameters the function to estimate f,
a vector X0 of the initial solution (for ODEs simply put a scalar), initial time t0,
final time tf and stepsize h, the function returns a solution matrix X and a time array.
'''
def RK4(f, X0, t0, tf, h):
    
    # create a time array from t0 to tf with stepsize h
    t = np.arange(t0, tf, h)
    # number of steps
    nt = t.size
    # X vector dimension
    nx = X0.size
    # Matrix for storing the estimates at each step
    X = np.zeros((nx,nt))
    # First column of the matrix is the initial value
    X[:,0] = X0
    
    for k in range(nt-1):
        k1 = h*f(t[k], X[:,k])
        k2 = h*f(t[k] + h/2, X[:,k] + k1/2)
        k3 = h*f(t[k] + h/2, X[:,k] + k2/2)
        k4 = h*f(t[k] + h, X[:,k] + k3)
        
        dX=(k1 + 2*k2 + 2*k3 +k4)/6
        X[:,k+1] = X[:,k] + dX;  
    
    return X, t
    


# Defining the problem
t0 = 0                                
tf = 300
h = 0.001
alpha = 0.2
beta = 0.01
gamma = 0.07
delta = 0.004
x0 = 19
y0 = 22
f = lambda t,x : Lotka_Volterra(x, alpha, beta, gamma, delta)        # lotka volterra problem 
X0 = np.array([x0,y0])                                               # initial condition    

# Solution with rk4
X, t = RK4(f, X0, t0, tf, h)

# Plotting the Results

plt.subplot(1, 2, 1)
plt.plot(t, X[0,:], "r", label="Rabbits")
plt.plot(t, X[1,:], "b", label="Foxes")
plt.xlabel("Time (t)")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X[0,:], X[1,:])
plt.xlabel("Preys")
plt.ylabel("Predators")
plt.grid()

plt.show()