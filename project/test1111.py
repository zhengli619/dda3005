import numpy as np

# Define a 1D vector A
A = np.array([10, 20, 30, 40, 50])

# Define the index array P
P = [3, 0, 4, 1, 2]  # Reorder indices

# Use P to reorder the elements of A
A_reordered = A[P]

print("Original vector A:", A)
print("Reordered vector A:", A_reordered)
print(type(A))
print(type(P))
print(type(A_reordered))
print(np.__config__.show())
