from math import *

# Definition of Runge Kutta 4 stages method
def rk4(f, exact_sol, t, y, T, h, analytical = False):

	# n: number of steps
	n = int((T-t)/h)

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

	choice = int(input(" 1) Estimate with RK-4 \n 2) Compare analytical solution with RK-4 estimate \n"))
	if choice == 2:
		print("Make sure you have defined the analytic function correctly.")

	t0 = float(input("t0 = "))
	y0 = float(input("y0 = "))
	T = float(input("T = "))
	h = float(input("Step-Size = "))
	rk4(f, exact_sol, t0, y0, T, h, choice == 2)

if __name__ == '__main__':
	main()
