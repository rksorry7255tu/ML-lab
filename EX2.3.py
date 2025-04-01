import numpy as np

# Create a square matrix
arr = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])  # Changed to a 3x3 matrix

# Perform operations
print("Array + 2:\n", arr + 2)
print("Array * 2:\n", arr * 2)
print("Matrix Multiplication:\n", np.dot(arr, arr.T))
print("Eigenvalue Decomposition:\n", np.linalg.eigvals(arr))