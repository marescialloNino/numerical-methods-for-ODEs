from math import *
import pandas as pd
from runge_kutta_4 import rk4
from eulers import forward_euler

def simpson_excersise(t, y0, y1, T, h ):
		
		f = lambda t, y: -5*y
		sol = lambda t: exp(-5*t)
	
		# n: number of steps
		n = int((T-t)/h)

		# define array to store data
		results = []

		# loop to calculate  t = 0, y0 = 1 
		for i in range(n):  

			tn1 = t + h

			f0 = f(t,y0)
			f1 = f(tn1, y1)

			y2 = (y0 + (h/3)*(f0 + 4*f1))/(1+(5/3)*h)
			

			actual = sol(t + 2*h)
			error = abs(actual - y2)
			results.append([t + 2*h, actual,  y2, error])
			
			
			t = tn1
			y0 = y1
			y1 = y2
		

		columns = ["t", "exact", "y", "error"]
		return pd.DataFrame(results, columns=columns)	


if __name__ == '__main__':

	t=0
	y0=1
	f = lambda t, y: -5*y
	sol = lambda t: exp(-5*t)
	h=0.05
	T=6
	
	exact_1 = sol(t + h)
	euler_1 = forward_euler(f,t,y0,t+h,h).y.iat[0]
	rk_1 = rk4(f,t,y0,t+h,h).y.iat[0]

	y1=[exact_1, rk_1,euler_1]

	result_1 = simpson_excersise(t, y0, exact_1, T, h )
	result_2 = simpson_excersise(t, y0, euler_1, T, h )
	result_3 = simpson_excersise(t, y0, rk_1, T, h )
	
	df2_extracted = result_2[['y', 'error']].rename(columns={'y': 'y_rk', 'error': 'error_rk'})
	df3_extracted = result_3[['y', 'error']].rename(columns={'y': 'y_eul', 'error': 'error_eul'})

	
	result = pd.concat([result_1, df2_extracted, df3_extracted], axis=1)

	print(result)
	


""" 	f = lambda t, y: -5*y + 5*t*t + 2*t

	f_actual = lambda t: t*t + exp(-5*t)/3

	t = float(input("t0: "))
	y = float(input("y0: "))
	p = float(input("Evaluation point: "))
	h = float(input("Step size: "))

	milne_simpson(f, f_actual, t, y, p, h) """
