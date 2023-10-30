from math import *

# Definition of Runge Kutta 4 stages method
def rk4(f, exact_sol, t, y, T, h, analytical = False, number_of_estimates = 0):

	# number_of_estimates is how many estimated solutions you want the function to return in an array,
	# for example if number_of_estimates == 2  --> [y1,y2] , if number_of_estimates == 1 --> [y1],
	# excluding y0, which is given by the initial problem.
	# if a number of estimates greater then zero is given the function returns the required number of 
	# estimates to be used somewhere else, otherwise it will print the estimate, exact solution
	# and error (if the analytical sol is given) for each step.

	# n: number of steps
	n = int((T-t)/h)

	if number_of_estimates > 0:
		estimates = []
		for i in range(number_of_estimates):
			k1 = h*f(t, y)
			k2 = h*f(t + h/2, y + k1/2)
			k3 = h*f(t + h/2, y + k2/2)
			k4 = h*f(t + h, y + k3)
			y = y + (k1 + 2*k2 + 2*k3 + k4)/6
			t += h
			estimates.append(y)
		print("estimates :", estimates)
		return estimates[1:]
	
	else:

		separator = "-"*100 if analytical else "-"*40

		# analytical == true --> compare RK-4 with the exact solution
		if analytical:
			print(f'\nt\t\t|\tEstimate\t\t|\tExact\t\t\t|\tError\n{separator}')
			print(f'{t:.3f}\t\t|\t{y:.9f}\t\t|\t{exact_sol(t):.9f}\t\t|\t{abs(exact_sol(t)-y):.9f}')
		# analytical == false --> estimate an ODE with RK-4
		else:
			print(f'\nt\t\t|\tEstimate\n{separator}')
			print(f'{t:.3f}\t\t|\t{y:.9f}')

		for i in range(n):
			k1 = h*f(t, y)
			k2 = h*f(t + h/2, y + k1/2)
			k3 = h*f(t + h/2, y + k2/2)
			k4 = h*f(t + h, y + k3)
			y = y + (k1 + 2*k2 + 2*k3 + k4)/6
			t += h
			actual = exact_sol(t)
			print(f'{t:.3f}\t\t|\t{y:.9f}', end = '')
			print(f'\t\t|\t{actual:.9f}\t\t|\t{abs(actual-y):.9f}' if analytical else '')

		print(separator)

# main function depending on the ODE to solve
def main():

	# Define these functions depending on the problem.
	f = lambda t, y: -5*y
	exact_sol= lambda t: exp(-5*t)

	choice = int(input(" To estimate with RK-4 enter [1] \n To compare analytical solution with RK-4 estimate enter [2] \n"))
	if choice == 2:
		print("Make sure you have defined the analytic function correctly.")

	t0 = float(input("t0 = "))
	y0 = float(input("y0 = "))
	T = float(input("T = "))
	h = float(input("Step-Size = "))
	rk4(f, exact_sol, t0, y0, T, h, choice == 2)

if __name__ == '__main__':
	main()
