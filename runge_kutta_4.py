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
		# define dataframe to store data
		results = []
		#columns = ["t", "Estimate", "Exact", "Error"]
		#results_df = pd.DataFrame(columns=columns)
		#first_row = pd.DataFrame({ 't': [t], 'Estimate': [y], "Exact":[actual], "Error": [error] })
		# loop to calculate the values of the parameters k at each step
		for i in range(n):
			k1 = h*f(t, y)
			k2 = h*f(t + h/2, y + k1/2)
			k3 = h*f(t + h/2, y + k2/2)
			k4 = h*f(t + h, y + k3)
			y = y + (k1 + 2*k2 + 2*k3 + k4)/6
			t += h
			if sol is not None:
				actual = sol(t)
				error = abs(actual - y)
				results.append([t, y, actual, error])
				#df_new_row = pd.DataFrame({ 't': [t], 'Estimate': [y], "Exact":[actual], "Error": [error] })
			else:
				results.append([t, y])
				#df_new_row = pd.DataFrame({ 't': [t], 'Estimate': [y]})
			#results_df = pd.concat([results_df, df_new_row],ignore_index=True)
			

		if sol is not None:
			columns = ["t", "Estimate", "Exact", "Error"]
		else:
			columns = ["t", "Estimate"]

		return pd.DataFrame(results, columns=columns)	 

# main function depending on the ODE to solve
# returns the an error dataframe and a log log plot of the final error
# in function of the number of steps n
def main():

	# Define these functions depending on the problem.
	f = lambda t, y: -5*y
	exact_sol= lambda t: exp(-5*t)
	t0 = 0
	y0 = 1
	T = 1
	# Define 
	error_columns = ["h", "n", "Final Error"]
	error_df = pd.DataFrame(columns=error_columns)

	steps = [2**(-k) for k in range(5,11)]

	for h in steps:
		
		results = rk4(f, t0, y0, T, h, exact_sol) 
		final_error = results["Error"].iat[-1]	
		error_df_new_row = pd.DataFrame({'h':[h], "n":[int((T-t0)/h)], "Final Error":[final_error]})
		error_df = pd.concat([error_df, error_df_new_row], ignore_index=True)
		#print(results)

	print(error_df)
	plt.loglog(error_df.n, error_df["Final Error"])
	plt.show()

if __name__ == '__main__':
	main()
