from math import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



""" 
	method to solve the problem dy/dt = -5y ; y(0) = 1.
	Simpson equation can be solved explicitly as the test function is linear.
"""
def simpsons(t, y0, y1, T, h ):
		
		f = lambda t, y: -5*y
		sol = lambda t: exp(-5*t)
	
		# n: number of steps
		n = int((T-t)/h)

		# define array to store data
		results = [[t, y0], [t+h, y1]]

		# loop to calculate  t = 0, y0 = 1 
		for i in range(n-1):  

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

'''
	Define euler and runge-kutta 4 stages to return the first estimate y1
'''
def euler_step(t0, y0, h, f):
    return y0 + h*f(t0, y0)

def rk4_step(t0, y0, h, f):
    k1 = f(t0, y0)
    k2 = f(t0 + h/2, y0 + k1*h/2)
    k3 = f(t0 + h/2, y0 + k2*h/2)
    k4 = f(t0 + h, y0 + k3*h)
    return y0 + h/6*(k1 + 2*k2 + 2*k3 + k4)

if __name__ == '__main__':

	t=0
	y0=1
	f = lambda t, y: -5*y
	sol = lambda t: exp(-5*t)
	h=0.05
	T=6
	X0 = np.array([y0])
	exact_1 = sol(t + h)
	euler_1 = euler_step(t,y0,h,f)
	rk_1 = rk4_step(t,y0,h,f)
	print(euler_1)
	print(rk_1)

	result_1 = simpsons(t, y0, exact_1, T, h )
	result_2 = simpsons(t, y0, euler_1, T, h )
	result_3 = simpsons(t, y0, rk_1, T, h )
	
	df2_extracted = result_2[['y', 'error']].rename(columns={'y': 'y_rk', 'error': 'error_rk'})
	df3_extracted = result_3[['y', 'error']].rename(columns={'y': 'y_eul', 'error': 'error_eul'})

	
	result = pd.concat([result_1, df2_extracted, df3_extracted], axis=1)
	
	print('''Results obtained with Simpsons method using different first value estimates:
	   		 y --> y1 computed as the exact solution at time t1
	   		 y_rk --> y1 computed with rk4
	   		 y_eul --> y1 computed with forward euler
	   ''')
	print(result)

	plt.plot(result.t , result.exact)
	result.plot(x='t', y=['exact','y','y_rk','y_eul'], kind='line')
	result.plot(x='t', y=[ 'error','error_rk','error_eul'], kind='line')
	plt.show()

