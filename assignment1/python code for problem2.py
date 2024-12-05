# Let's generate Python code to plot the given polynomial and the compact formula.

import numpy as np
import matplotlib.pyplot as plt

# Define the original polynomial from the problem
def poly_expanded(x):
    return (x**8 - 24*x**7 + 252*x**6 - 1512*x**5 + 5670*x**4 - 13608*x**3 + 20412*x**2 - 17496*x + 6561)

# Define the compact polynomial formula (x - 3)^8
def poly_compact(x):
    return (x - 3)**8

# Generate x values from 2.920 to 3.080 with step size 0.001
x_values = np.arange(2.920, 3.081, 0.001)

# Compute y values for both polynomials
y_expanded = poly_expanded(x_values)
y_compact = poly_compact(x_values)

# Plot both polynomials
plt.figure(figsize=(10, 6))

# Plot the expanded polynomial
plt.plot(x_values, y_expanded, label="Expanded Polynomial", color='blue', linestyle='--')

# Plot the compact polynomial
plt.plot(x_values, y_compact, label="Compact Polynomial $(x-3)^8$", color='red', linestyle='-')

# Adding titles and labels
plt.title("Comparison of Expanded and Compact Polynomial Representations")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
