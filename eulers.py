from math import *
import pandas as pd

def forward_euler(f, t, y, T, h, sol = None):
			
		# n: number of steps
		n = int((T-t)/h)

		# define array to store data
		results = []

		# loop to calculate estimates at each step
		for i in range(n):
			y = y + h*f(t,y)
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
			columns = ["t", "y", "Exact", "Error"]
		else:
			columns = ["t", "y"]

		return pd.DataFrame(results, columns=columns)

def improved_euler(f, t, y, T, h, sol = None):
			
		# n: number of steps
		n = int((T-t)/h)

		# define array to store data
		results = []

		# loop to calculate estimates at each step
		for i in range(n):
			func_value = f(t, y)
			t += h
			next_value = f(t, y + h*func_value)
			y = y + h/2*(func_value + next_value)
			# if analytical solution is given --> compare estimate ad exact solution
			if sol is not None:
				actual = sol(t)
				error = abs(actual - y)
				results.append([t, y, actual, error])
			# if not, just estimate the solution
			else:
				results.append([t, y])		

		if sol is not None:
			columns = ["t", "y", "Exact", "Error"]
		else:
			columns = ["t", "y"]

		return pd.DataFrame(results, columns=columns)	 