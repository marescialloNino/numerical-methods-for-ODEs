from math import *
import pandas as pd
from runge_kutta_4 import rk4
from eulers import forward_euler

def milne_simpson(f, f_actual, t, y, p, h):
	n = int((p-t)/h)
	time = [t+i*h for i in range(n+1)]

	y_values = [0]*(n+1)
	y_values[0] = y

	for i in range(1, 4):
		y_values[i] = rk4(f, time[i-1], y_values[i-1])

	for i in range(4, n+1):
		milne = y_values[i-4] + 4/3*h*(2*f(time[i-1], y_values[i-1]) - f(time[i-2], y_values[i-2]) + 2*f(time[i-3], y_values[i-3]))
		y_values[i] = y_values[i-2] + h/3*(f(time[i],milne) + 4*f(time[i-1], y_values[i-1]) + f(time[i-2], y_values[i-2]))

	print(f"Milne-Simpson Method")
	print("-"*70)
	print(f't\t\t|\tEstimate\t\t|\tActual')
	print("-"*70)

	for i in range(len(y_values)):
		print(f'{time[i]:.3f}\t\t|\t{y_values[i]:.9f}\t\t|\t{f_actual(time[i]):.9f}')
	print("-"*70)

def simpson(f, t, y0, y1, T, h, sol = None):
	
		# n: number of steps
		n = int((T-t)/h)
		time = [t+i*h for i in range(n+1)]
		# define array to store data
		results = []

		f0 = f(t,y0)
		 
		# loop to calculate the values of the parameters k at each step
		for i in range(n):
			y = y + (h/3)*(f(t,y) + 4*f(t+h,))
			# if analytical solution is given --> compare estimate ad exact solution
			if sol is not None:
				actual = sol(t)
				error = abs(actual - y)
				results.append([t, y, actual, error])
			# if not, just estimate the solution
			else:
				results.append([t, y])		

		if sol is not None:
			columns = ["t", "Estimate", "Exact", "Error"]
		else:
			columns = ["t", "Estimate"]

		return pd.DataFrame(results, columns=columns)	


if __name__ == '__main__':

	f = lambda t, y: -5*y + 5*t*t + 2*t

	f_actual = lambda t: t*t + exp(-5*t)/3

	t = float(input("t0: "))
	y = float(input("y0: "))
	p = float(input("Evaluation point: "))
	h = float(input("Step size: "))

	milne_simpson(f, f_actual, t, y, p, h)
