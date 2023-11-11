from math import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



""" 
	method to solve the problem dy/dt = -5y ; y(0) = 1.
	Simpson equation can be solved explicitly as the test function is linear.
	t --> initial time
	y0 --> initial value
	y1 --> second initial value
	T --> final time
	h --> stepsize
"""
def simpsons(t, y0, y1, T, h ):
		
		f = lambda t, y: -5*y
		sol = lambda t: exp(-5*t)
	
		# n: number of steps
		n = int((T-t)/h)

		# define array to store results data
		results = [[t, y0], [t+h, y1]]

		# loop to calculate results , t = 0, y0 = 1 
		for i in range(n-1):  

			tn1 = t + h

			f0 = f(t,y0)
			f1 = f(tn1, y1)

			y2 = (y0 + (h/3)*(f0 + 4*f1))/(1+(5/3)*h)
			
			actual = sol(t + 2*h)
			# error calculated as the difference between exact solution and estimate
			error = actual - y2
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

	# definition of the problem
	t=0
	y0=1
	f = lambda t, y: -5*y
	sol = lambda t: exp(-5*t)
	h=0.05
	T=6

	# estimate of the second initial value
	exact_1 = sol(t + h)
	euler_1 = euler_step(t,y0,h,f)
	rk_1 = rk4_step(t,y0,h,f)

	# compute solutions in the three different cases
	result_1 = simpsons(t, y0, exact_1, T, h )
	result_2 = simpsons(t, y0, euler_1, T, h )
	result_3 = simpsons(t, y0, rk_1, T, h )
	
	df2_extracted = result_2[['y', 'error']].rename(columns={'y': 'y_eul', 'error': 'error_eul'})
	df3_extracted = result_3[['y', 'error']].rename(columns={'y': 'y_rk', 'error': 'error_rk'})

	
	result = pd.concat([result_1, df2_extracted, df3_extracted], axis=1)
	
	print('''
	   Results obtained with Simpsons method using different first value estimates:
	   		 y --> y1 computed as the exact solution at time t1
	   		 y_rk --> y1 computed with rk4
	   		 y_eul --> y1 computed with forward euler
	   
	   ''')
	print(result)

	
	plt.subplot(1, 2, 1)
	plt.plot(result.t, result.exact,color='blue' ,label='Exact solution')
	plt.plot(result.t,  result.y, color='red', linewidth=1,label='Simpson:first value exact sol')
	plt.plot(result.t,result.y_rk,color='green', linewidth=0.7,label='Simpson:first value rk4')
	plt.xlabel("Time (s)")
	plt.grid()
	plt.legend()


	plt.subplot(1, 2, 2)
	plt.plot(result.t,result.y_eul,label='Simpson:first value eul')
	plt.xlabel("Time (s)")
	plt.legend()
	plt.grid()
	plt.show()


	plt.subplot(1, 2, 1)
	plt.plot(result.t,  result.error_rk, color='orange', linewidth=1,label='Error: first value rk4')
	plt.plot(result.t, result.error,color='blue' ,label='Error: first value exact sol')
	plt.xlabel("Time (s)")
	plt.grid()
	plt.legend()


	plt.subplot(1, 2, 2)
	plt.plot(result.t,result.error_eul,label='Error: first value eul')
	plt.xlabel("Time (s)")
	plt.legend()
	plt.grid()
	plt.show()
