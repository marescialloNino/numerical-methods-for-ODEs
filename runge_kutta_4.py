from math import *
import pandas as pd
import matplotlib.pyplot as plt

'''
	Definition of Runge Kutta 4 stages method
 	This function takes as parameters:
 	f --> f(t,y)=dy/dt
	sol --> the analytical solution of the ODE
	t --> initial time t0 
	y --> initial value y0
	T --> final time T
 	h --> stepsize 
	The function returns a dataframe with 4 columns : tn, yn, y(tn), error
'''

def rk4(f, t, y, T, h, sol = None):
	
		# n: number of steps
		n = int((T-t)/h)
		# define array to store data
		results = []

		# loop to calculate the values of the parameters k at each step
		for i in range(n):
			k1 = h*f(t, y)
			k2 = h*f(t + h/2, y + k1/2)
			k3 = h*f(t + h/2, y + k2/2)
			k4 = h*f(t + h, y + k3)
			y = y + (k1 + 2*k2 + 2*k3 + k4)/6
			t += h
			# if analytical solution is given --> compare estimate ad exact solution
			if sol is not None:
				actual = sol(t)
				error = abs(actual - y)
				results.append([t, y, actual, error])
			# if not, just estimate the solution
			else:
				results.append([t, y])		

		if sol is not None:
			columns = ["t", "y", "exact", "error"]
		else:
			columns = ["t", "y"]

		return pd.DataFrame(results, columns=columns)	 

""" 
	excersise2 function depending on the ODE to solve
	returns an error dataframe and a log log plot of the final error
 	in function of the number of steps n
"""
def excersise2():

	# Define these functions depending on the problem to solve.
	f = lambda t, y: -5*y
	exact_sol= lambda t: exp(-5*t)
	t0 = 0
	y0 = 1
	T = 5
	
	# define table of stepsize, number of steps and error.
	error_columns = ["h", "n", "final_error"]
	error_df = pd.DataFrame(columns=error_columns)

	# define an array of steps, h = 2^(-k) for k=5,6,7,8,9,10.
	steps = [2**(-k) for k in range(5,11)]

	# calculate rk4 results at each given step, storing the error at final time in the error table.
	for h in steps:
		results = rk4(f, t0, y0, T, h, exact_sol) 
		final_error = results["error"].iat[-1]	
		error_df_new_row = pd.DataFrame({'h':[h], "n":[int((T-t0)/h)], "final_error":[final_error]})
		error_df = pd.concat([error_df, error_df_new_row], ignore_index=True)

	result = rk4(f, t0, y0, T, steps[0], exact_sol) 
		
	print(error_df)
	plt.loglog(error_df.n, error_df["final_error"])

	plt.show()
	plt.plot(result.t , result.exact)
	plt.plot(result.t , result.y)

	plt.show()

if __name__ == '__main__':
	excersise2()
