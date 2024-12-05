import numpy as np
from scipy.linalg import solve
from scipy.linalg import lu_factor, lu_solve

# Directly determining whether eigenvectors are nearly collinear (i.e., whether their directions are close) is an indirect approach.
# Eigenvalues and eigenvectors are related, but in numerical computations, eigenvectors can exhibit some instability, 
# especially when the eigenvalues do not change significantly. 
# Therefore, directly comparing the change in eigenvalues is a more stable and explicit criterion for convergence.

def power_iteration(A, x0, tol, max_iterations=1000):
    x_k = x0 / np.linalg.norm(x0)  #unit length
    lambda_old = 0

    for k in range(max_iterations):
        # compute A*x_k
        x_tilde = A @ x_k

        # normalize x_tilde
        x_k_next = x_tilde / np.linalg.norm(x_tilde)

        # compute eigenvalue
        lambda_k = x_k_next.T @ A @ x_k_next

        # judge whether converge
        if abs(lambda_k - lambda_old) < tol:
            print(f"Converged after {k + 1} iterations.")
            return lambda_k, x_k_next

        # update x_k
        x_k = x_k_next
        lambda_old=lambda_k

    print("Maximum number of power iterations reached.")
    return lambda_k, x_k



def inverse_iteration(A, x0, sigma, tol, max_iterations=10000):
    x_k = x0 / np.linalg.norm(x0)  # unit length
    lambda_old = 0
    for k in range(max_iterations):
       
        # Solve (A - sigma * I) * y = x_k
        x_tilde = solve(A - sigma * np.eye(A.shape[0]), x_k)

        # Normalize  to get x_k_next
        x_k_next = x_tilde / np.linalg.norm(x_tilde)

        # Compute the eigenvalue approximation
        lambda_k = x_k_next.T @ A @ x_k_next

        # judge converge or not
        if abs(lambda_k - lambda_old) < tol:
            print(f"Converged after {k + 1} iterations with shift {sigma}.")
            return lambda_k, x_k_next

        # update x_k
        x_k = x_k_next
        lambda_old=lambda_k

    print("Maximum of inverse power iterations with shift reached.")
    return lambda_k, x_k


def rayleith_quotient_iteration(A, x0, tol, max_iterations=10000):
    x_k = x0 / np.linalg.norm(x0)  # unit length
    lambda_old= x0.T @ A @ x0
    for k in range(max_iterations):
        # Solve (A - sigma * I) * y = x_k
        lu, piv = lu_factor(A - lambda_old * np.eye(A.shape[0]))
        x_tilde = lu_solve((lu, piv), x_k)
        # initially I USE:
        # x_tilde = solve(A - lambda_old * np.eye(A.shape[0]), x_k) to calculate x_tilde, but get warning:
        # LinAlgWarning: Ill-conditioned matrix (rcond=2.81961e-18): result may not be accurate.x_tilde = solve(A - lambda_k * np.eye(A.shape[0]), x_k)

        # Normalize to get x_k_next
        x_k_next = x_tilde / np.linalg.norm(x_tilde)

        # Compute the eigenvalue approximation
        lambda_k = x_k_next.T @ A @ x_k_next

        # judge converge or not
        if abs(lambda_k - lambda_old) < tol:
            print(f"Converged after {k + 1} iterations")
            return lambda_k, x_k_next

        # update x_k
        x_k = x_k_next
        lambda_old=lambda_k

    print("Maximum of inverse power iterations with shift reached.")
    return lambda_k, x_k


A = np.array([
    [46, 99, 45, 24, -27],
    [-6, -32,-6, 4, 18],
    [18, 18, 19, 6, 0],
    [-9, -90, -9, -47, 27],
    [12, 39, 12, 28, 19]
])

e1 = np.array([1, 0, 0, 0, 0])
e2 = np.array([0, 1, 0, 0, 0])
e3 = np.array([0, 0, 1, 0, 0])
e4 = np.array([0, 0, 0, 1, 0])
e5 = np.array([0, 0, 0, 0, 1])
e_list = [e1, e2, e3, e4, e5]


if __name__ == "__main__":
    # I want accuracy!
    tol=1e-12

    largest_eigenvalue, corresponding_eigenvector = power_iteration(A, e1, tol=1e-12)
    print("Largest Eigenvalue:", largest_eigenvalue)
    print("Corresponding Eigenvector:", corresponding_eigenvector,"\n")

    # Possible shifts for inverse iteration
    shifts = [0, 46, -32, 19, -47,]
    # Calculate remaining eigen-pairs using inverse iteration
    for sigma in shifts:
        eigenvalue, eigenvector = inverse_iteration(A, e1, sigma, tol=1e-12)
        print(f"Eigenvalue with shift {sigma}: {eigenvalue}")
        print(f"Corresponding Eigenvector with shift {sigma}: {eigenvector}\n")

    for initial_vector in e_list:
        eigenvalue, eigenvector = rayleith_quotient_iteration(A, initial_vector, tol=1e-12)
        print(f"Eigenvalue with initial vector {initial_vector}:{eigenvalue}")
        print(f"Corresponding Eigenvector: {eigenvector}\n")

