import numpy as np
import matplotlib.pyplot as plt

# Recurrence relation: x_k+1 = 9/4 * x_k - 1/2 * x_k-1
# Initial values
x1 = 1 / 3
x2 = 1 / 12
n = 60  # Number of elements to compute

# Initialize the sequence
x_values = np.zeros(n)
x_values[0] = x1
x_values[1] = x2

# Recurrence computation for the first 60 elements
for k in range(2, n):
    x_values[k] = (9 / 4) * x_values[k - 1] - (1 / 2) * x_values[k - 2]

# Create a semilog plot of the obtained values as a function of k
k_values = np.arange(1, n + 1)
plt.figure(figsize=(10, 6))
plt.semilogy(k_values, np.abs(x_values), label="Recursion Result", color='blue')

# Part b: Exact solution x_k = (1/3) * 4^(1-k)
exact_values = (1 / 3) * (4.0)** (1 - k_values)

# Plot the exact solution
plt.semilogy(k_values, np.abs(exact_values), label="Exact Solution $\\frac{1}{3}4^{1-k}$", color='red', linestyle='--')

# Adding titles and labels
plt.title("Recursion vs Exact Solution")
plt.xlabel("k")
plt.ylabel("$x_k$")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
