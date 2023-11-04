import numpy as np
import matplotlib.pyplot as plt

def R(z):
    return 1 + z + 0.5*z**2 + (1/6)*z**3 + (1/24)*z**4

# Create a grid of complex numbers
re = np.linspace(-5, 5, 500)
im = np.linspace(-5, 5, 500)
Re, Im = np.meshgrid(re, im)
Z = Re + 1j*Im

# Evaluate the stability function
R_Z = R(Z)

# Identify points in the stability region
stability_region = np.abs(R_Z) <= 1

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(Re, Im, stability_region, shading='auto',cmap='gray_r')
plt.title('Stability Region for 4th Order Runge-Kutta')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axvline(0, color='red', linestyle='--')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.colorbar(label='|R(z)|')
plt.axis('equal')
plt.show()
